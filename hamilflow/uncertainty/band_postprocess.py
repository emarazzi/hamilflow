from pathlib import Path
from typing import Sequence
import numpy as np
from dataclasses import dataclass
import json


@dataclass
class BandUncertaintyPostProcessor:
    """Summary statistics over the per-k, per-band values written by `BandUncertaintyCalculator`.

    Reads both `compute`/`compute_parallel` outputs (`key="sigma_eV_avg_ham"` or `"sigma_eV"`)
    and `compare_averaged_to_dft` outputs (`key="abs_err_eV"`), so the predicted uncertainty
    and the true error can be reduced over exactly the same bands.

    Example usage:
      pp = BandUncertaintyPostProcessor.from_json("band_uncertainty.json")
      stats = pp.summary(vbm_fraction=0.2)
      weighted = pp.weighted_summary(e_cut=1.0, tau=0.1)
    """

    results: dict
    key: str = "sigma_eV_avg_ham"

    @classmethod
    def from_json(cls, path: Path | str, key: str = "sigma_eV_avg_ham"):
        with open(path) as f:
            return cls(json.load(f), key=key)

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
        values, weights = self.values(structure)
        idx = self.band_indices(structure, bands, vbm_fraction)
        sel = values[:, idx]
        i_k, i_b = np.unravel_index(np.argmax(sel), sel.shape)
        vbm = self.vbm_index(self.results[structure])
        return {
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
    ) -> dict:
        """Band-edge-weighted mean and max of `self.key` for one structure.

        Each state gets weight `w = soft_window(d, e_cut, tau)`, with `d` its distance from its
        band edge (see `edge_distance`), so states near the VBM/CBM count fully whatever the gap.
        `bands`/`vbm_fraction` (see `band_indices`) are applied first as a hard selection.

        - `weighted_mean_eV`: `sum(w_k * w * x) / sum(w_k * w)`, with `w_k` the k-point symmetry
          weights -- the typical value near the band edges.
        - `weighted_max_eV`: `max(w * x)` -- the worst state near the band edges.
        - `n_eff_bands`: `sum(w_k * w) / sum(w_k)`, the number of bands per k-point the window
          effectively keeps.

        The unweighted `structure_stats` over the same bands are included alongside, for comparison.
        """
        values, weights = self.values(structure)
        idx = self.band_indices(structure, bands, vbm_fraction)
        sel = values[:, idx]
        w = self.soft_window(self.edge_distance(structure)[:, idx], e_cut, tau)
        state_w = weights[:, None] * w
        weighted = w * sel
        i_k, i_b = np.unravel_index(np.argmax(weighted), weighted.shape)
        vbm_e, cbm_e = self.band_edges(structure)
        return {
            **self.structure_stats(structure, bands, vbm_fraction),
            "weighted_mean_eV": float(np.sum(state_w * sel) / np.sum(state_w)),
            "weighted_max_eV": float(weighted[i_k, i_b]),
            "weighted_argmax_k": int(i_k),
            "weighted_argmax_band": int(idx[i_b]),
            "n_eff_bands": float(np.sum(state_w) / np.sum(weights)),
            "vbm_eV": vbm_e,
            "cbm_eV": cbm_e,
            "gap_eV": cbm_e - vbm_e,
        }

    def weighted_summary(
        self,
        e_cut: float = 1.0,
        tau: float = 0.1,
        bands: Sequence[int] | slice | None = None,
        vbm_fraction: float | None = None,
    ) -> dict[str, dict]:
        """`weighted_stats` for every structure in the file."""
        return {s: self.weighted_stats(s, e_cut, tau, bands, vbm_fraction) for s in self.results}


__all__ = ["BandUncertaintyPostProcessor"]
