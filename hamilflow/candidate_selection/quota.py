"""Per-group quota selection.

Split a selection budget of ``n_total`` structures over user-defined groups
(e.g. one group per parent cell, composition, temperature, MD run, ...) and
pick that many diverse structures inside each group.

- :func:`allocate_quotas`: the quota arithmetic alone -- a proportional split
  of ``n_total`` by group weight, with a per-group minimum, fixed overrides
  and per-group capacities, rounded by largest remainder.
- :func:`select_by_group_quota`: allocate the quotas, optionally keep only
  the top-scoring candidates of each group (e.g. by model uncertainty), then
  run k-means (k = quota) on the descriptors of each group and pick one
  structure per cluster.

Groups are plain labels supplied by the caller (any hashable, one per
candidate); nothing here knows what a group means. Weights are supplied as a
mapping or a callable ``group -> weight``.
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

PICK_RULES = ("centroid", "max_score")
# warnings are attributed to the first caller outside this package
_SKIP = (os.path.dirname(__file__) + os.sep,)

GroupWeights = Mapping[Hashable, float] | Callable[[Hashable], float] | None


def _resolve_weights(groups: Sequence[Hashable], weights: GroupWeights) -> dict[Hashable, float]:
    if weights is None:
        out = {g: 1.0 for g in groups}
    elif isinstance(weights, Mapping):
        missing = [g for g in groups if g not in weights]
        if missing:
            raise ValueError(f"weights has no entry for group(s) {missing[:5]}")
        out = {g: float(weights[g]) for g in groups}
    elif callable(weights):
        out = {g: float(weights(g)) for g in groups}
    else:
        raise TypeError("weights must be None, a mapping group -> weight, or a callable")
    bad = {g: w for g, w in out.items() if not math.isfinite(w) or w < 0}
    if bad:
        raise ValueError(f"group weights must be finite and >= 0, got {bad}")
    return out


def _water_fill(n: int, w: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Real-valued shares ``clip(lam * w, lo, hi)`` summing to ``n``.

    Assumes ``lo.sum() <= n < hi.sum()``, ``hi = lo`` where ``w == 0`` and
    ``hi`` possibly infinite. The sum is a nondecreasing, piecewise-linear
    function of ``lam`` with breakpoints at ``lo/w`` and ``hi/w``, so ``lam``
    is found exactly by locating the bracketing breakpoints and
    interpolating (or extrapolating past the last finite breakpoint).
    """
    pos = w > 0

    def share(lam: float) -> np.ndarray:
        return np.clip(lam * w, lo, hi)

    bps = np.concatenate([lo[pos] / w[pos], hi[pos] / w[pos]])
    bps = np.unique(bps[np.isfinite(bps)])
    prev_lam, prev_sum = 0.0, float(share(0.0).sum())
    if prev_sum >= n:
        return share(0.0)
    for lam in bps:
        s = float(share(lam).sum())
        if s >= n:
            return share(prev_lam + (n - prev_sum) * (lam - prev_lam) / (s - prev_sum))
        prev_lam, prev_sum = lam, s
    slope = w[pos & (prev_lam * w < hi)].sum()
    return share(prev_lam + (n - prev_sum) / slope)


def allocate_quotas(
    n_total: int,
    groups: Sequence[Hashable],
    *,
    weights: GroupWeights = None,
    available: Mapping[Hashable, int] | None = None,
    fixed: Mapping[Hashable, int] | None = None,
    min_per_group: int = 0,
) -> dict[Hashable, int]:
    """Split ``n_total`` selections over ``groups`` in proportion to their weights.

    Parameters
    ----------
    n_total
        Total number of selections to distribute.
    groups
        Distinct group labels. Their order is the tie-breaking priority: when
        two groups have the same rounding remainder, the earlier one gets the
        extra selection.
    weights
        Mapping or callable ``group -> weight`` (finite, >= 0). ``None``
        weights all groups equally.
    available
        Capacity of each group (e.g. its number of candidates); no group is
        given more than its capacity. ``None`` means unlimited.
    fixed
        Quotas set by hand for some groups; these are taken out of
        ``n_total`` first and the rest is split over the other ("free")
        groups. A fixed quota above the group's capacity is capped, with a
        warning.
    min_per_group
        Minimum quota of every free group (or its capacity, if smaller).

    Returns
    -------
    dict
        ``group -> quota`` in the order of ``groups``.

    Notes
    -----
    The free budget ``n_free = n_total - sum(fixed)`` is first split into
    real-valued shares ``clip(lam * weight, min, capacity)`` with ``lam``
    chosen so that the shares sum to ``n_free`` (the proportional split,
    with groups at their minimum or capacity taken out and their surplus
    or deficit spread over the others in proportion to weight). The shares
    are then rounded down and the remaining units handed out one each by
    largest fractional remainder, ties going to the group listed first.
    A group of weight 0 gets only its minimum. If the free groups cannot
    absorb ``n_free`` (not enough capacity in the groups of positive
    weight), each gets its capacity (zero-weight ones their minimum) and the
    total falls short of ``n_total``, with a warning.

    Raises
    ------
    ValueError
        If ``groups`` has duplicates, a count is negative, ``fixed`` names an
        unknown group, the fixed quotas exceed ``n_total``, or the minima of
        the free groups do not fit in what is left.
    """
    groups = list(groups)
    if len(set(groups)) != len(groups):
        raise ValueError("groups must be distinct")
    if int(n_total) != n_total or int(min_per_group) != min_per_group or n_total < 0 or min_per_group < 0:
        raise ValueError(f"n_total and min_per_group must be integers >= 0, got {n_total}, {min_per_group}")
    fixed = dict(fixed or {})
    unknown = [g for g in fixed if g not in set(groups)]
    if unknown:
        raise ValueError(f"fixed quotas given for unknown group(s) {unknown[:5]}")
    if any(int(q) != q or q < 0 for q in fixed.values()):
        raise ValueError(f"fixed quotas must be non-negative integers, got {fixed}")
    if available is None:
        cap = {g: math.inf for g in groups}
    else:
        missing = [g for g in groups if g not in available]
        if missing:
            raise ValueError(f"available has no entry for group(s) {missing[:5]}")
        cap = {g: available[g] for g in groups}
        if any(c < 0 for c in cap.values()):
            raise ValueError("available counts must be >= 0")
    w = _resolve_weights(groups, weights)

    quotas: dict[Hashable, int] = {}
    for g, q in fixed.items():
        if q > cap[g]:
            warnings.warn(
                f"fixed quota {q} of group {g!r} exceeds its {cap[g]} candidates: capped", skip_file_prefixes=_SKIP
            )
            q = cap[g]
        quotas[g] = int(q)
    n_free = n_total - sum(quotas.values())
    if n_free < 0:
        raise ValueError(f"fixed quotas ({sum(quotas.values())}) exceed n_total ({n_total})")

    free = [g for g in groups if g not in fixed]
    if free:
        lo = np.array([min(min_per_group, cap[g]) for g in free], dtype=float)
        hi = np.array([cap[g] for g in free], dtype=float)
        wf = np.array([w[g] for g in free])
        if lo.sum() > n_free:
            raise ValueError(
                f"n_total ({n_total}) too small: {n_total - n_free} fixed + min_per_group ({min_per_group}) "
                f"for {len(free)} free group(s) need {n_total - n_free + int(lo.sum())}"
            )
        # zero-weight groups only get their minimum
        hi = np.where(wf > 0, hi, lo)
        if hi.sum() <= n_free:
            share = hi
            if hi.sum() < n_free:
                warnings.warn(
                    f"the free groups can take only {int(hi.sum())} of the {n_free} free selections "
                    f"(capacity, or zero weight): selecting {int(hi.sum()) + n_total - n_free} in total "
                    f"instead of {n_total}",
                    skip_file_prefixes=_SKIP,
                )
        else:
            share = _water_fill(n_free, wf, lo, hi)
        base = np.floor(share + 1e-9)
        base = np.minimum(base, hi)
        n_extra = round(float(min(n_free, hi.sum()) - base.sum()))
        frac = np.round(share - base, 9)
        # largest remainder first; stable sort keeps the group order on ties
        order = sorted(range(len(free)), key=lambda i: -frac[i])
        for i in order:
            if n_extra <= 0:
                break
            if base[i] < hi[i]:
                base[i] += 1
                n_extra -= 1
        quotas.update({g: int(b) for g, b in zip(free, base)})
    elif n_free > 0:
        warnings.warn(
            f"all groups have fixed quotas: selecting {n_total - n_free} in total instead of {n_total}",
            skip_file_prefixes=_SKIP,
        )
    return {g: quotas[g] for g in groups}


@dataclass
class GroupQuotaSelection:
    """Output of :func:`select_by_group_quota`.

    All index arrays index into the full candidate set passed in
    (``len(group_labels)`` entries). Per-group dicts are in group order.

    - ``selected_idx``: the selection, grouped by group (in ``groups``
      order) and ascending within each group.
    - ``cluster_size``: for each entry of ``selected_idx``, the number of
      candidates in the k-means cluster it represents (1 when a group's
      quota equals its number of candidates and no clustering was run).
    - ``candidate_idx``: all candidates that entered the per-group
      clustering, i.e. after the score filter (ascending).
    - ``quotas``: allocated quota per group; ``n_selected`` can be lower if
      k-means left clusters empty (degenerate descriptors).
    - ``n_available``: candidates per group before the score filter.
    - ``n_candidates``: candidates per group after the score filter.
    - ``weights``: the resolved group weights.
    """

    selected_idx: np.ndarray
    cluster_size: np.ndarray
    candidate_idx: np.ndarray
    groups: list
    quotas: dict
    n_available: dict
    n_candidates: dict
    weights: dict
    selected_by_group: dict = field(repr=False)

    @property
    def n_selected(self) -> dict:
        """Number of selected candidates per group."""
        return {g: len(self.selected_by_group[g]) for g in self.groups}

    def summary(self) -> list[dict]:
        """One row per group (``group``, ``weight``, ``quota``, ``selected``,
        ``available``, ``candidates``), handy for logging or a DataFrame."""
        return [
            {
                "group": g,
                "weight": self.weights[g],
                "quota": self.quotas[g],
                "selected": len(self.selected_by_group[g]),
                "available": self.n_available[g],
                "candidates": self.n_candidates[g],
            }
            for g in self.groups
        ]


def _cluster_pick(
    X: np.ndarray,
    k: int,
    *,
    scores: np.ndarray | None,
    sample_weight: np.ndarray | None,
    pick: str,
    seed: int,
    n_init: int,
) -> tuple[list[int], list[int]]:
    """k-means on rows of ``X``, one local index per non-empty cluster
    (closest to the centroid or highest score; ties -> lowest index)."""
    from sklearn.cluster import KMeans

    if k == len(X):
        return list(range(k)), [1] * k
    km = KMeans(n_clusters=k, random_state=seed, n_init=n_init).fit(X, sample_weight=sample_weight)
    picks, sizes = [], []
    for c in range(k):
        members = np.flatnonzero(km.labels_ == c)
        if len(members) == 0:
            continue
        if pick == "centroid":
            j = members[np.argmin(np.linalg.norm(X[members] - km.cluster_centers_[c], axis=1))]
        else:
            j = members[np.argmax(scores[members])]
        picks.append(int(j))
        sizes.append(len(members))
    return picks, sizes


def select_by_group_quota(
    descriptors: Sequence | np.ndarray,
    group_labels: Sequence[Hashable],
    n_total: int,
    *,
    group_order: Sequence[Hashable] | None = None,
    weights: GroupWeights = None,
    fixed: Mapping[Hashable, int] | None = None,
    min_per_group: int = 0,
    scores: Sequence[float] | np.ndarray | None = None,
    top_fraction: float | None = None,
    top_n: int | None = None,
    pick: str = "centroid",
    sample_weight: Sequence[float] | np.ndarray | None = None,
    candidate_idx: Sequence[int] | np.ndarray | None = None,
    seed: int = 42,
    n_init: int = 10,
) -> GroupQuotaSelection:
    """Select ``n_total`` candidates split over groups by quota, diverse within each group.

    1. Quotas: ``n_total`` is split over the groups with
       :func:`allocate_quotas` (``weights``, ``fixed``, ``min_per_group``;
       capacity = number of candidates in the group).
    2. Score filter (optional, needs ``scores``): within each group keep the
       ``top_n`` highest-scoring candidates, or the top ``top_fraction`` of
       them (rounded up), but never fewer than the group's quota.
    3. Diversity: within each group, k-means with k = quota on the
       descriptors of the remaining candidates; one candidate per cluster,
       the one closest to the centroid (``pick="centroid"``) or the
       highest-scoring one (``pick="max_score"``, needs ``scores``).

    Parameters
    ----------
    descriptors
        One feature vector per candidate, shape ``(n, d)`` (e.g. the AMDs
        from :meth:`CandidateSelector.compute_amds`).
    group_labels
        One group label (any hashable) per candidate.
    n_total
        Total selection budget.
    group_order
        Order of the groups, used for tie-breaking in the quota rounding
        (earlier wins) and for the order of the output. Must contain each
        group present exactly once. Default: order of first appearance in
        ``group_labels``.
    weights, fixed, min_per_group
        See :func:`allocate_quotas`.
    scores
        One score per candidate, higher = more wanted (e.g. an uncertainty).
        Needed by ``top_fraction``/``top_n`` and ``pick="max_score"``.
    top_fraction, top_n
        Per-group score filter (mutually exclusive). ``top_fraction`` in
        ``(0, 1]``; ``top_n >= 1``.
    pick
        ``"centroid"`` or ``"max_score"``.
    sample_weight
        Optional per-candidate k-means weight (e.g. to down-weight
        over-represented, near-duplicate configurations).
    candidate_idx
        Optional subset of candidates to select from (indices into
        ``descriptors``); the others are ignored entirely, including for the
        group sizes. Default: all.
    seed, n_init
        Passed to ``sklearn.cluster.KMeans`` (``random_state``, ``n_init``).

    Returns
    -------
    GroupQuotaSelection

    Notes
    -----
    Deterministic for fixed inputs and ``seed``. Ties are broken by the
    group order (quota rounding) and by the lower candidate index (equal
    scores in the filter, equal distances or scores inside a cluster).
    A group whose quota equals its number of candidates is taken whole,
    without clustering. A k-means cluster can come out empty for degenerate
    descriptors (fewer distinct points than k); the group then gets fewer
    selections than its quota, with a warning.
    """
    if pick not in PICK_RULES:
        raise ValueError(f"Unknown pick '{pick}' (expected one of {PICK_RULES})")
    n = len(group_labels)
    X = np.asarray(descriptors, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2 or len(X) != n:
        raise ValueError(f"descriptors must have shape (n, d) with n = len(group_labels) = {n}, got {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError("descriptors contain non-finite values")
    labels = list(group_labels)

    if scores is not None:
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (n,):
            raise ValueError(f"scores must have shape ({n},), got {scores.shape}")
        if not np.all(np.isfinite(scores)):
            raise ValueError("scores contain non-finite values")
    if top_fraction is not None and top_n is not None:
        raise ValueError("give top_fraction or top_n, not both")
    if (top_fraction is not None or top_n is not None or pick == "max_score") and scores is None:
        raise ValueError("top_fraction/top_n and pick='max_score' need scores")
    if top_fraction is not None and not 0 < top_fraction <= 1:
        raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}")
    if top_n is not None and top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if sample_weight.shape != (n,):
            raise ValueError(f"sample_weight must have shape ({n},), got {sample_weight.shape}")
        if not np.all(np.isfinite(sample_weight)) or np.any(sample_weight < 0):
            raise ValueError("sample_weight must be finite and >= 0")

    if candidate_idx is None:
        pool = np.arange(n)
    else:
        pool = np.unique(np.asarray(candidate_idx, dtype=int))
        if len(pool) and (pool[0] < 0 or pool[-1] >= n):
            raise ValueError(f"candidate_idx out of range [0, {n})")
    if len(pool) == 0:
        raise ValueError("no candidates to select from")

    present = list(dict.fromkeys(labels[i] for i in pool))
    if group_order is None:
        groups = present
    else:
        groups = list(group_order)
        if len(set(groups)) != len(groups):
            raise ValueError("group_order has duplicates")
        missing = [g for g in present if g not in set(groups)]
        if missing:
            raise ValueError(f"group_order misses group(s) {missing[:5]}")
    members = {g: [] for g in groups}
    for i in pool:
        members[labels[i]].append(int(i))
    members = {g: np.array(m, dtype=int) for g, m in members.items()}
    n_available = {g: len(members[g]) for g in groups}

    quotas = allocate_quotas(
        n_total, groups, weights=weights, available=n_available, fixed=fixed, min_per_group=min_per_group
    )
    w = _resolve_weights(groups, weights)

    selected_by_group, sizes_by_group, cands, n_candidates = {}, {}, [], {}
    for g in groups:
        idx, k = members[g], quotas[g]
        if top_fraction is not None or top_n is not None:
            n_keep = top_n if top_n is not None else math.ceil(top_fraction * len(idx) - 1e-9)
            n_keep = min(len(idx), max(n_keep, k))
            idx = np.sort(idx[np.argsort(-scores[idx], kind="stable")[:n_keep]])
        n_candidates[g] = len(idx)
        cands.append(idx)
        if k == 0:
            selected_by_group[g], sizes_by_group[g] = np.array([], dtype=int), np.array([], dtype=int)
            continue
        picks, sizes = _cluster_pick(
            X[idx],
            k,
            scores=None if scores is None else scores[idx],
            sample_weight=None if sample_weight is None else sample_weight[idx],
            pick=pick,
            seed=seed,
            n_init=n_init,
        )
        if len(picks) < k:
            warnings.warn(
                f"group {g!r}: k-means left {k - len(picks)} of {k} clusters empty, selecting {len(picks)}",
                skip_file_prefixes=_SKIP,
            )
        order = np.argsort(idx[picks])
        selected_by_group[g] = idx[picks][order]
        sizes_by_group[g] = np.asarray(sizes, dtype=int)[order]
        logger.info(f"group {g!r}: quota {k}, {len(idx)} candidate(s) of {n_available[g]}")

    return GroupQuotaSelection(
        selected_idx=np.concatenate([selected_by_group[g] for g in groups]).astype(int),
        cluster_size=np.concatenate([sizes_by_group[g] for g in groups]).astype(int),
        candidate_idx=np.sort(np.concatenate(cands)).astype(int),
        groups=groups,
        quotas=quotas,
        n_available=n_available,
        n_candidates=n_candidates,
        weights=w,
        selected_by_group=selected_by_group,
    )


__all__ = ["GroupQuotaSelection", "allocate_quotas", "select_by_group_quota"]
