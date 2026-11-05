import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np


@dataclass
class BandUncertaintyPostProcessor:
    """Summary statistics over the per-k, per-band values written by `BandUncertaintyCalculator`.

    Reads both `compute`/`compute_parallel` outputs (`key="sigma_eV_avg_ham"` or `"sigma_eV"`)
    and `compare_averaged_to_dft` outputs (`key="abs_err_eV"`), so the predicted uncertainty
    and the true error can be reduced over exactly the same bands.

    `refit_shift`: None (default) uses the stored values, aligned by the rigid shift fitted on
    the Hamiltonian matrices (`align_mode="matrix"`), which is dominated by the orbitals far from
    the gap. "mean" or "median" instead refits, for every statistic, one constant shift per
    spectrum on the states it reduces over, with the statistic's own state weights (the soft
    window of `weighted_stats`, the k-point weights over the selected bands in `structure_stats`):
    for `abs_err_eV` the shift of the average against DFT, from the signed `err_eV`
    (`|err - delta|`); for `sigma_eV_avg_ham` / `sigma_eV` one shift per member, from the signed
    `member_deviation_eV` (RMS / standard deviation over members of `dev_m - delta_m`). "mean"
    takes the weighted mean of the signed values (least squares), "median" the weighted median.
    A rigid shift moves every eigenvalue of a spectrum by the same amount, so this removes only
    the energy offset in the window; gap, band-shape and k-dependent errors are kept. Use the same
    `refit_shift` for uncertainty and error. Needs files that store the signed values.

    Example usage:
      pp = BandUncertaintyPostProcessor.from_json("band_uncertainty.json")
      stats = pp.summary(vbm_fraction=0.2)
      weighted = pp.weighted_summary(e_cut=1.0, tau=0.1)
      refit = BandUncertaintyPostProcessor.from_json("band_uncertainty.json", refit_shift="mean")
    """

    results: dict
    key: str = "sigma_eV_avg_ham"
    refit_shift: str | None = None

    _REFIT: ClassVar[tuple] = (None, "mean", "median")
    # signed per-k values each key is built from
    _SIGNED: ClassVar[dict] = {"abs_err_eV": "err_eV", "sigma_eV_avg_ham": "member_deviation_eV", "sigma_eV": "member_deviation_eV"}

    def __post_init__(self):
        if self.refit_shift not in self._REFIT:
            raise ValueError(f"`refit_shift` must be one of {self._REFIT}, got {self.refit_shift!r}")
        if self.refit_shift is not None and self.key not in self._SIGNED:
            raise ValueError(f"`refit_shift` needs `key` in {tuple(self._SIGNED)}, got {self.key!r}")

    @classmethod
    def from_json(cls, path: Path | str, key: str = "sigma_eV_avg_ham", refit_shift: str | None = None):
        with open(path) as f:
            return cls(json.load(f), key=key, refit_shift=refit_shift)

    @staticmethod
    def _per_k(res: dict) -> dict:
        # Per-k entries are under "kpoints"; "per_k" is the key in older `compare_averaged_to_dft` outputs.
        return res["kpoints"] if "kpoints" in res else res["per_k"]

    @staticmethod
    def vbm_index(res: dict) -> int:
        if "vbm_index" in res:
            return int(res["vbm_index"])
        # Outputs written before `vbm_index` was stored: assume spinless (two electrons per band).
        return int(res["occupation"]) // 2 - 1

    def values(self, structure: str) -> tuple[np.ndarray, np.ndarray]:
        """Return `(values, weights)`: `values[i_k, i_band]` for `self.key`, and the irreducible
        k-point weights."""
        per_k = self._per_k(self.results[structure])
        try:
            values = np.array([entry[self.key] for entry in per_k.values()], dtype=float)
        except KeyError as e:
            raise KeyError(f"{structure}: no {e} in per-k entries -- wrong `key` for this file?") from e
        weights = np.array([entry["weight"] for entry in per_k.values()], dtype=float)
        return values, weights

    def signed_values(self, structure: str) -> np.ndarray:
        """Signed values behind `self.key`, `d[i_spectrum, i_k, i_band]`: the members' deviations
        from the average (`member_deviation_eV`, one spectrum per member) or the average's error
        against DFT (`err_eV`, a single spectrum)."""
        signed = self._SIGNED[self.key]
        per_k = self._per_k(self.results[structure])
        try:
            d = np.array([entry[signed] for entry in per_k.values()], dtype=float)
        except KeyError as e:
            raise KeyError(
                f"{structure}: no {e} in per-k entries -- file written before signed values were "
                "stored, recompute it to use `refit_shift`"
            ) from e
        return d[None] if d.ndim == 2 else np.moveaxis(d, 1, 0)

    def refit(self, structure: str, idx: np.ndarray, state_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`(values[i_k, i], shifts)`: `self.key` on the bands `idx` after subtracting from each
        spectrum the shift `refit_shift` fitted on those states with weights `state_w[i_k, i]`
        (see the class docstring); `shifts` has one entry per spectrum."""
        d = self.signed_values(structure)[:, :, idx]
        if self.refit_shift == "mean":
            shifts = np.array([np.sum(state_w * dm) / np.sum(state_w) for dm in d])
        else:
            shifts = np.array([self.weighted_quantile(dm, state_w, 0.5) for dm in d])
        d = d - shifts[:, None, None]
        if self.key == "abs_err_eV":
            return np.abs(d[0]), shifts
        if self.key == "sigma_eV_avg_ham":
            return np.sqrt(np.mean(d**2, axis=0)), shifts
        return np.std(d, axis=0, ddof=1), shifts

    def _selected(self, structure: str, idx: np.ndarray, state_w: np.ndarray) -> tuple[np.ndarray, dict]:
        """`self.key` on the bands `idx` (refitted if `refit_shift`), and the refit shifts to report."""
        if self.refit_shift is None:
            return self.values(structure)[0][:, idx], {}
        sel, shifts = self.refit(structure, idx, state_w)
        return sel, {"refit_shift_eV": shifts.tolist()}

    def eigenvalues(self, structure: str) -> np.ndarray:
        """Midgap-aligned eigenvalues of the averaged Hamiltonian, `eigvals[i_k, i_band]`."""
        per_k = self._per_k(self.results[structure])
        try:
            return np.array([entry["eigvals_avg_ham_eV"] for entry in per_k.values()], dtype=float)
        except KeyError as e:
            raise KeyError(
                f"{structure}: no 'eigvals_avg_ham_eV' in per-k entries -- file written before "
                "eigenvalues were stored, recompute it"
            ) from e

    def band_edges(self, structure: str) -> tuple[float, float]:
        """`(VBM, CBM)` energies over the stored (irreducible) k-points, from the averaged
        Hamiltonian's eigenvalues."""
        eigvals = self.eigenvalues(structure)
        vbm = self.vbm_index(self.results[structure])
        if vbm + 1 >= eigvals.shape[1]:
            raise ValueError(f"{structure}: no conduction band stored (VBM index {vbm}, {eigvals.shape[1]} bands)")
        return float(eigvals[:, vbm].max()), float(eigvals[:, vbm + 1].min())

    def edge_distance(self, structure: str) -> np.ndarray:
        """Distance of each state from its band edge, `d[i_k, i_band] >= 0`: `VBM - e` for
        valence bands, `e - CBM` for conduction bands."""
        eigvals = self.eigenvalues(structure)
        vbm_e, cbm_e = self.band_edges(structure)
        valence = np.arange(eigvals.shape[1]) <= self.vbm_index(self.results[structure])
        return np.where(valence[None, :], vbm_e - eigvals, eigvals - cbm_e)

    @staticmethod
    def soft_window(d: np.ndarray, e_cut: float, tau: float) -> np.ndarray:
        """Fermi-like window `1 / (1 + exp((d - e_cut) / tau))`, rescaled so that `w(0) = 1`.

        Flat up to about `e_cut`, then drops to zero over a width of a few `tau`; `tau -> 0`
        recovers a hard window `d <= e_cut`.
        """
        if tau <= 0:
            raise ValueError(f"`tau` must be > 0, got {tau}")
        # 0.5 * (1 - tanh(x / 2)) == 1 / (1 + exp(x)), without overflow for large x.
        fermi = lambda x: 0.5 * (1.0 - np.tanh(x / 2.0))
        return fermi((d - e_cut) / tau) / fermi(-e_cut / tau)

    def band_indices(
        self,
        structure: str,
        bands: Sequence[int] | slice | None = None,
        vbm_fraction: float | None = None,
    ) -> np.ndarray:
        """Indices of the bands to reduce over (all bands if neither option is given).

        - `bands`: explicit band indices (list or slice, negative indices allowed).
        - `vbm_fraction`: take `n = round(vbm_fraction * n_occ)` (at least 1) bands ending at the
          VBM, plus the same number of bands starting at the CBM, clipped to the bands available.
        """
        if bands is not None and vbm_fraction is not None:
            raise ValueError("Pass either `bands` or `vbm_fraction`, not both")
        res = self.results[structure]
        n_bands = len(next(iter(self._per_k(res).values()))[self.key])
        all_bands = np.arange(n_bands)
        if bands is not None:
            return all_bands[bands]
        if vbm_fraction is None:
            return all_bands
        if not 0 < vbm_fraction <= 1:
            raise ValueError(f"`vbm_fraction` must be in (0, 1], got {vbm_fraction}")
        vbm = self.vbm_index(res)
        n_side = max(1, round(vbm_fraction * (vbm + 1)))
        return all_bands[vbm + 1 - n_side : vbm + 1 + n_side]

    def structure_stats(
        self,
        structure: str,
        bands: Sequence[int] | slice | None = None,
        vbm_fraction: float | None = None,
    ) -> dict:
        """Mean and max of `self.key` over the selected bands and all k-points of one structure.

        The mean weights each irreducible k-point by its symmetry weight, i.e. it is the mean over
        the full k-mesh.
        """
        weights = self.values(structure)[1]
        idx = self.band_indices(structure, bands, vbm_fraction)
        sel, refit = self._selected(structure, idx, np.repeat(weights[:, None], len(idx), axis=1))
        i_k, i_b = np.unravel_index(np.argmax(sel), sel.shape)
        vbm = self.vbm_index(self.results[structure])
        return {
            **refit,
            "mean_eV": float(np.average(sel.mean(axis=1), weights=weights)),
            "max_eV": float(sel[i_k, i_b]),
            "argmax_k": int(i_k),
            "argmax_band": int(idx[i_b]),
            "n_valence": int(np.sum(idx <= vbm)),
            "n_conduction": int(np.sum(idx > vbm)),
        }

    def summary(
        self,
        bands: Sequence[int] | slice | None = None,
        vbm_fraction: float | None = None,
    ) -> dict[str, dict]:
        """`structure_stats` for every structure in the file."""
        return {s: self.structure_stats(s, bands, vbm_fraction) for s in self.results}

    def weighted_stats(
        self,
        structure: str,
        e_cut: float = 1.0,
        tau: float = 0.1,
        bands: Sequence[int] | slice | None = None,
        vbm_fraction: float | None = None,
        quantile: float | None = None,
    ) -> dict:
        """Band-edge-weighted mean and max of `self.key` for one structure.

        Each state gets weight `w = soft_window(d, e_cut, tau)`, with `d` its distance from its
        band edge (see `edge_distance`), so states near the VBM/CBM count fully whatever the gap.
        `bands`/`vbm_fraction` (see `band_indices`) are applied first as a hard selection.

        - `weighted_mean_eV`: `sum(w_k * w * x) / sum(w_k * w)`, with `w_k` the k-point symmetry
          weights -- the typical value near the band edges.
        - `weighted_max_eV`: `max(w * x)` -- the worst state near the band edges.
        - `weighted_quantile_eV` (only if `quantile` is given): the `quantile` of `x` over the
          states, each with weight `w_k * w` (see `weighted_quantile`) -- e.g. 0.5 for the median,
          0.9 for a tail value less dominated by a single state than `weighted_max_eV`.
        - `n_eff_bands`: `sum(w_k * w) / sum(w_k)`, the number of bands per k-point the window
          effectively keeps.

        The unweighted `structure_stats` over the same bands are included alongside, for comparison.
        With `refit_shift`, the weighted statistics use the shifts refitted with the window
        weights (`weighted_refit_shift_eV`), the unweighted ones their own (`refit_shift_eV`).
        """
        weights = self.values(structure)[1]
        idx = self.band_indices(structure, bands, vbm_fraction)
        w = self.soft_window(self.edge_distance(structure)[:, idx], e_cut, tau)
        state_w = weights[:, None] * w
        sel, refit = self._selected(structure, idx, state_w)
        weighted = w * sel
        i_k, i_b = np.unravel_index(np.argmax(weighted), weighted.shape)
        vbm_e, cbm_e = self.band_edges(structure)
        out = {
            **self.structure_stats(structure, bands, vbm_fraction),
            **{f"weighted_{k}": v for k, v in refit.items()},
            "weighted_mean_eV": float(np.sum(state_w * sel) / np.sum(state_w)),
            "weighted_max_eV": float(weighted[i_k, i_b]),
            "weighted_argmax_k": int(i_k),
            "weighted_argmax_band": int(idx[i_b]),
            "n_eff_bands": float(np.sum(state_w) / np.sum(weights)),
            "vbm_eV": vbm_e,
            "cbm_eV": cbm_e,
            "gap_eV": cbm_e - vbm_e,
        }
        if quantile is not None:
            out["weighted_quantile_eV"] = self.weighted_quantile(sel, state_w, quantile)
        return out

    @staticmethod
    def weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
        """Smallest value of `x` whose cumulative weight reaches a fraction `q` of the total weight
        `sum(w)` (`q = 1` gives the maximum over the states with nonzero weight)."""
        if not 0 <= q <= 1:
            raise ValueError(f"`quantile` must be in [0, 1], got {q}")
        x, w = np.ravel(x), np.ravel(w)
        order = np.argsort(x)
        cdf = np.cumsum(w[order])
        # small tolerance so that rounding in the cumulative sum does not skip a value
        i = np.searchsorted(cdf, q * cdf[-1] * (1 - 1e-12))
        return float(x[order][min(i, len(x) - 1)])

    def weighted_summary(
        self,
        e_cut: float = 1.0,
        tau: float = 0.1,
        bands: Sequence[int] | slice | None = None,
        vbm_fraction: float | None = None,
        quantile: float | None = None,
    ) -> dict[str, dict]:
        """`weighted_stats` for every structure in the file."""
        return {s: self.weighted_stats(s, e_cut, tau, bands, vbm_fraction, quantile) for s in self.results}


Groups = Callable[[str], str] | Mapping[str, str] | None


def _group_of(groups: Groups, structure: str) -> str | None:
    if groups is None:
        return None
    return groups(structure) if callable(groups) else groups[structure]


def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (no tie correction)."""
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


@dataclass
class UncertaintyCalibrator:
    """Calibrate the band uncertainty of an ensemble against DFT, for a stopping criterion.

    Pairs, structure by structure, the uncertainty `u_i` (from a `compute` output, key
    `sigma_eV_avg_ham`) with the error `e_i` of the averaged Hamiltonian (from a
    `compare_averaged_to_dft` output of the same ensemble, key `abs_err_eV`), both reduced with
    the same statistic `stat` over the same states:

    - "weighted_max_eV" / "weighted_mean_eV": `BandUncertaintyPostProcessor.weighted_stats`
      with `e_cut`, `tau` (and `bands`/`vbm_fraction` as a hard preselection);
    - "weighted_quantile_eV": the same, the weighted `quantile` of the states (e.g. 0.9);
    - "max_eV" / "mean_eV": `BandUncertaintyPostProcessor.structure_stats` with `bands`/`vbm_fraction`.

    The calibration factor is a split-conformal quantile of the ratios `r_i = e_i / u_i`:
    `s_q` is the `ceil((n + 1) q)`-th smallest ratio of the `n` calibration structures, so that
    `e <= s_q u` holds for at least a fraction `q` of new structures exchangeable with them.
    The guarantee needs calibration structures chosen independently of `u` (e.g. a fixed test
    set), not structures selected for their high uncertainty, and structures no ensemble member
    was trained on. Refit for every new ensemble.

    Example usage:
      cal = UncertaintyCalibrator.from_json("uncertainty_test.json", "dft_compare_test.json",
                                            e_cut=1.0, tau=0.1)
      fit = cal.fit(q=0.9, groups=lambda s: "untwisted" if "untwisted" in s else "twisted")
      cal.coverage(fit["s"], structures=newly_labeled)          # check on the selected S_t
      pool = BandUncertaintyPostProcessor.from_json("uncertainty_pool.json")
      cal.pool_criterion(pool, fit["s"], eta_tol=0.025, percentile=95)
    """

    uncertainty: BandUncertaintyPostProcessor
    error: BandUncertaintyPostProcessor
    stat: str = "weighted_max_eV"
    e_cut: float = 1.0
    tau: float = 0.1
    bands: Sequence[int] | slice | None = None
    vbm_fraction: float | None = None
    quantile: float = 0.9

    _STATS = ("weighted_max_eV", "weighted_mean_eV", "weighted_quantile_eV", "max_eV", "mean_eV")
    _SETTINGS = ("align_mode", "ill_method", "ill_threshold")

    def __post_init__(self):
        if self.stat not in self._STATS:
            raise ValueError(f"`stat` must be one of {self._STATS}, got {self.stat!r}")
        if not 0 <= self.quantile <= 1:
            raise ValueError(f"`quantile` must be in [0, 1], got {self.quantile}")
        self._check_refit(self.error, "error")
        self.check_consistency()

    def _check_refit(self, pp: BandUncertaintyPostProcessor, name: str) -> None:
        if pp.refit_shift != self.uncertainty.refit_shift:
            raise ValueError(
                f"`refit_shift` of the {name} ({pp.refit_shift!r}) differs from that of the uncertainty "
                f"({self.uncertainty.refit_shift!r}): u and e would be aligned differently"
            )

    @classmethod
    def from_json(cls, uncertainty_path: Path | str, error_path: Path | str, refit_shift: str | None = None,
                  **kwargs):
        """`refit_shift`: see `BandUncertaintyPostProcessor`, applied to both files."""
        return cls(
            BandUncertaintyPostProcessor.from_json(uncertainty_path, key="sigma_eV_avg_ham", refit_shift=refit_shift),
            BandUncertaintyPostProcessor.from_json(error_path, key="abs_err_eV", refit_shift=refit_shift),
            **kwargs,
        )

    def check_consistency(self) -> None:
        """Raise if a structure present in both files was computed with different settings or
        k-points, so that `u` and `e` would not refer to the same states."""
        bad = []
        for s in set(self.uncertainty.results) & set(self.error.results):
            ru, re = self.uncertainty.results[s], self.error.results[s]
            settings_u = {k: ru.get(k) for k in self._SETTINGS}
            settings_e = {k: re.get(k) for k in self._SETTINGS}
            kpts_u = [e["k_frac"] for e in BandUncertaintyPostProcessor._per_k(ru).values()]
            kpts_e = [e["k_frac"] for e in BandUncertaintyPostProcessor._per_k(re).values()]
            if settings_u != settings_e or not np.allclose(kpts_u, kpts_e):
                bad.append(s)
        if bad:
            raise ValueError(
                f"{len(bad)} structures differ in settings ({', '.join(self._SETTINGS)}) or "
                f"k-points between the uncertainty and the error file, e.g. {sorted(bad)[:3]}"
            )

    def score(self, pp: BandUncertaintyPostProcessor, structure: str) -> float:
        """`stat` of `pp.key` for one structure."""
        if self.stat.startswith("weighted"):
            stats = pp.weighted_stats(structure, self.e_cut, self.tau, self.bands, self.vbm_fraction, self.quantile)
        else:
            stats = pp.structure_stats(structure, self.bands, self.vbm_fraction)
        return stats[self.stat]

    def paired(self, structures: Sequence[str] | None = None) -> tuple[list[str], np.ndarray, np.ndarray]:
        """`(names, u, e)` for the structures present in both files (restricted to `structures`
        if given; structures missing from either file raise)."""
        common = [s for s in self.uncertainty.results if s in self.error.results]
        if structures is not None:
            missing = [s for s in structures if s not in common]
            if missing:
                raise KeyError(f"not in both files: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
            common = list(structures)
        u = np.array([self.score(self.uncertainty, s) for s in common])
        e = np.array([self.score(self.error, s) for s in common])
        return common, u, e

    @staticmethod
    def conformal_factor(ratios: np.ndarray, q: float) -> float:
        """`ceil((n + 1) q)`-th smallest ratio; needs `n >= q / (1 - q)` ratios."""
        if not 0 < q < 1:
            raise ValueError(f"`q` must be in (0, 1), got {q}")
        # small tolerance so that e.g. (9 + 1) * 0.9 counts as 9, not 9.000000000000002
        rank = lambda n: math.ceil((n + 1) * q - 1e-9)
        n = len(ratios)
        if rank(n) > n:
            n_min = next(m for m in range(1, 10**6) if rank(m) <= m)
            raise ValueError(f"{n} calibration structures are too few for q={q}: need at least {n_min}")
        return float(np.sort(ratios)[rank(n) - 1])

    @staticmethod
    def _ratios(u: np.ndarray, e: np.ndarray) -> np.ndarray:
        return np.divide(e, u, out=np.full_like(e, np.inf), where=u > 0)

    def _s_values(self, names: Sequence[str], s: float | Mapping[str, float], groups: Groups) -> np.ndarray:
        if isinstance(s, Mapping):
            if groups is None:
                raise ValueError("per-group `s` needs `groups`")
            return np.array([s[_group_of(groups, n)] for n in names])
        return np.full(len(names), float(s))

    def fit(
        self,
        q: float = 0.9,
        structures: Sequence[str] | None = None,
        groups: Groups = None,
        min_group_size: int = 10,
    ) -> dict:
        """Fit the conformal factor on the calibration structures.

        Returns `s` (global factor), the ratio statistics and the rank correlation between `u`
        and `e`. With `groups` (callable or mapping structure -> group name), also
        `s_per_group`: the factor fitted on each group with at least `min_group_size` structures
        (and enough for `q`), the global factor otherwise (listed in `fallback_groups`). Pass
        `fit["s"]` or `fit["s_per_group"]` (with the same `groups`) to `coverage` and
        `pool_criterion`.
        """
        names, u, e = self.paired(structures)
        r = self._ratios(u, e)
        out = {
            "q": q,
            "n": len(names),
            "s": self.conformal_factor(r, q),
            "ratio_median": float(np.median(r)),
            "ratio_max": float(np.max(r)),
            "rank_correlation": _rank_correlation(u, e),
            "stat": self.stat,
            **({"quantile": self.quantile} if self.stat == "weighted_quantile_eV" else {}),
        }
        if groups is not None:
            per_group, fallback, sizes = {}, [], {}
            labels = np.array([_group_of(groups, n) for n in names])
            for g in dict.fromkeys(labels.tolist()):
                m = labels == g
                sizes[g] = int(m.sum())
                try:
                    if m.sum() < min_group_size:
                        raise ValueError
                    per_group[g] = self.conformal_factor(r[m], q)
                except ValueError:
                    per_group[g] = out["s"]
                    fallback.append(g)
            out.update(s_per_group=per_group, group_sizes=sizes, fallback_groups=fallback)
        return out

    def coverage(
        self,
        s: float | Mapping[str, float],
        structures: Sequence[str] | None = None,
        groups: Groups = None,
    ) -> dict:
        """Fraction of structures with `e <= s u` (e.g. on the newly labeled, high-`u`
        structures, or on a later test set), and the structures that violate it."""
        names, u, e = self.paired(structures)
        bound = self._s_values(names, s, groups) * u
        inside = e <= bound
        return {
            "n": len(names),
            "coverage": float(inside.mean()) if len(names) else float("nan"),
            "violations": {n: {"u": float(uu), "e": float(ee), "bound": float(b)}
                           for n, uu, ee, b, ok in zip(names, u, e, bound, inside) if not ok},
            "rank_correlation": _rank_correlation(u, e),
        }

    def pool_criterion(
        self,
        pool: BandUncertaintyPostProcessor,
        s: float | Mapping[str, float],
        eta_tol: float,
        percentile: float = 95.0,
        groups: Groups = None,
    ) -> dict:
        """Stopping criterion on the unlabeled pool: the `percentile` of the calibrated bounds
        `s u_i` over the pool must not exceed `eta_tol` (eV).

        A high percentile rather than the maximum is used, since over a large pool the maximum
        of a noisy `u` is dominated by its fluctuations. `pool` is a `compute` output of the same
        ensemble, with the same settings (key `sigma_eV_avg_ham`).
        """
        self._check_refit(pool, "pool")
        names = list(pool.results)
        u = np.array([self.score(pool, n) for n in names])
        bound = self._s_values(names, s, groups) * u
        value = float(np.percentile(bound, percentile))
        order = np.argsort(bound)[::-1]
        return {
            "n_pool": len(names),
            "percentile": percentile,
            "bound_percentile_eV": value,
            "bound_max_eV": float(bound.max()),
            "eta_tol_eV": eta_tol,
            "n_above_eta": int(np.sum(bound > eta_tol)),
            "stop": value <= eta_tol,
            "largest_bounds": {names[i]: float(bound[i]) for i in order[:10]},
        }


__all__ = ["BandUncertaintyPostProcessor", "UncertaintyCalibrator"]
