"""K-fold ``dataset_split_{i}.json`` files for ensemble training, extendable across AL cycles.

Fold ``i`` validates on group ``i`` and trains on the other ``k - 1`` groups. Structures are
dealt into groups stratified by parent: variants of the same parent geometry (by default, names
that differ only after their last ``-``, e.g. ``tMoS2Bilayer_n1m5Ang6-003`` and ``-init``) are spread over
different folds, so every ensemble member sees every geometry. Names with no parent in common
fall back to a balanced random split.

In later AL cycles the previous splits are frozen: only the newly added structures are dealt
into the k groups, appended to each fold's validate set, and to every other fold's train set.
"""
import json
import random
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path


def parent_of(name: str) -> str:
    """Strip the variant suffix after the last ``-``; names without a ``-`` are their own parent."""
    return name.rsplit("-", 1)[0]


def _check_previous(previous: list[dict], k: int) -> set:
    if len(previous) != k:
        raise ValueError(f"Got {len(previous)} previous splits, expected k={k}")
    validates = [set(p["validate"]) for p in previous]
    old_names = set().union(*validates)
    if sum(len(v) for v in validates) != len(old_names):
        raise ValueError("Previous splits have overlapping validate sets")
    for i, p in enumerate(previous):
        if set(p["train"]) | validates[i] != old_names or set(p["train"]) & validates[i]:
            raise ValueError(f"Previous split {i + 1} is not a k-fold partition of the same structures")
    return old_names


def make_kfold_splits(
    names: Sequence[str],
    k: int,
    previous: list[dict] | None = None,
    group_key: Callable[[str], str] | None = parent_of,
    seed: int = 137,
) -> list[dict]:
    """Return ``k`` split dicts ``{"train", "validate", "test"}``.

    ``names`` is the full list of training structures for this cycle. With ``previous`` (the
    splits of the last AL cycle), every structure already in them keeps its assignment and only
    the names not seen before are distributed. ``group_key=None`` makes the split plain random.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    names = list(dict.fromkeys(names))
    key = group_key or (lambda n: n)

    if previous:
        old_names = _check_previous(previous, k)
        test = list(previous[0].get("test", []))
        missing = old_names - set(names)
        if missing:
            raise ValueError(f"{len(missing)} structures of the previous splits are not in names: {sorted(missing)[:5]}")
    else:
        previous = [{"train": [], "validate": [], "test": []} for _ in range(k)]
        old_names, test = set(), []
    new = [n for n in names if n not in old_names and n not in test]

    # Biggest groups first (random order among equal sizes), so they get spread before folds fill up.
    rng = random.Random(seed)
    groups: dict[str, list[str]] = {}
    for n in new:
        groups.setdefault(key(n), []).append(n)
    group_list = list(groups.values())
    rng.shuffle(group_list)
    group_list.sort(key=len, reverse=True)

    new_validate: list[list[str]] = [[] for _ in range(k)]
    parent_counts = [Counter(key(n) for n in p["validate"]) for p in previous]
    sizes = [len(p["validate"]) for p in previous]
    for members in group_list:
        rng.shuffle(members)
        for n in members:
            p = key(n)
            i = min(range(k), key=lambda j: (parent_counts[j][p], sizes[j], rng.random()))
            new_validate[i].append(n)
            parent_counts[i][p] += 1
            sizes[i] += 1

    splits = []
    for i, p in enumerate(previous):
        held_out = set(new_validate[i])
        splits.append({
            "train": list(p["train"]) + [n for n in new if n not in held_out],
            "validate": list(p["validate"]) + [n for n in new if n in held_out],
            "test": test,
        })
    return splits


def read_kfold_splits(split_dir, k: int, prefix: str = "dataset_split") -> list[dict]:
    splits = []
    for i in range(1, k + 1):
        with open(Path(split_dir) / f"{prefix}_{i}.json", encoding="utf-8") as f:
            splits.append(json.load(f))
    return splits


def write_kfold_splits(
    names: Sequence[str],
    k: int,
    out_dir,
    previous_dir=None,
    group_key: Callable[[str], str] | None = parent_of,
    seed: int = 137,
    prefix: str = "dataset_split",
) -> list[Path]:
    """Write ``{prefix}_{1..k}.json`` to ``out_dir``, extending the splits in ``previous_dir`` if given."""
    previous = read_kfold_splits(previous_dir, k, prefix) if previous_dir is not None else None
    splits = make_kfold_splits(names, k, previous=previous, group_key=group_key, seed=seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, split in enumerate(splits, start=1):
        path = out_dir / f"{prefix}_{i}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split, f, indent=2)
        paths.append(path)
    return paths
