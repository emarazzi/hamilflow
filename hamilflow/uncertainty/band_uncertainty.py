from pathlib import Path
from typing import Iterable
import numpy as np
from dataclasses import dataclass, field
import concurrent.futures
import json
import os
import time

import threadpoolctl
from hamilflow.sparse_hamiltonian import SparseHamiltonianObj
from hamilflow.sparse_hamiltonian.eigensolver import eigh_generalized

from .alignment import fit_rigid_shifts, write_aligned_hamiltonian
from .hamiltonian_io import average_predicted_hamiltonians

import spglib

# Largest structural fill of H(k) for which method="projected" forms C_W^dag H_m(k) as a
# sparse-dense product (O(nnz n_w)) instead of a dense GEMM (O(n^2 n_w)). Measured
# crossover: sparse ~1.5x faster at 10% fill, slower above ~15%, independent of n.
SPARSE_PRODUCT_MAX_FILL = 0.1

@dataclass
class BandUncertaintyCalculator:
    """Compute band-energy uncertainty across an ensemble of models.

    Example usage:
      calc = BandUncertaintyCalculator()
      output = calc.compute(model_dirs, dft_dir)

    - `align_mode`: how spectra are put on a common energy reference before comparison.
      "matrix" (default): each member's Hamiltonian is aligned to the averaged Hamiltonian by the
      rigid shift c S that best explains their difference (`alignment.fit_rigid_shifts`); since
      H - c S has the eigenvalues of H shifted by exactly -c, this subtracts c from the member's
      eigenvalues, with no extra diagonalization. All spectra are then shifted by one common
      reference, the averaged Hamiltonian's midgap on the k-grid. In `compare_averaged_to_dft`
      the average is aligned to DFT the same way and DFT's midgap is the common reference.
      Unlike a per-spectrum midgap, this keeps errors of the band edges (e.g. a member whose CBM
      is too low) and is not moved by a single spurious state at an edge.
      "grid": each spectrum shifted by its own midgap between the VBM (max over the k-grid of the
      highest occupied band) and the CBM (min over the k-grid of the lowest unoccupied band).
      "gamma": each spectrum shifted by its own midgap between HOMO and LUMO at `anchor_k`.
    - `aligned_hamiltonian_dir`: with `align_mode="matrix"`, if given, the aligned Hamiltonians
      are also written there (`model_<k>/<structure>/` for the members in `compute`,
      `average_to_dft/<structure>/` for the average in `compare_averaged_to_dft`), with the shift
      stored as an attribute. Not needed for the uncertainty itself; off by default.
    - `ill_method`: None or "orbital_removal" -- handling of an ill-conditioned overlap,
      applied to every diagonalization (members, averaged Hamiltonian, DFT). Orbitals are removed
      (`sparse_hamiltonian.ill_conditioned.global_orbital_truncation`) until the smallest
      eigenvalue of S(k) on the k-grid exceeds `ill_threshold`. The choice depends on S only, so
      it is made once per structure and all spectra of a structure are computed in the same
      reduced basis; the removed eigenvalue slots are dropped. "window_regularization" (also
      offered by `IllConditionedHandler`) is not supported: its energy window would have to be
      placed relative to the Fermi energy in info.json, which is copied from the DFT calculation
      and is not the Fermi level of a predicted Hamiltonian.
    - `kpoints_settings`: alternative to the fixed `grid_mesh`. If given, it is passed as
      `kpoints_updates` to `dft_workflows.kpoints.get_ksampling` with each structure's cell, so
      the mesh is chosen per structure, e.g. {"grid_density": 1000},
      {"reciprocal_density": 100} or {"k_grid": [4, 4, 4]}. It must resolve to a k_grid (not a
      line path or explicit k-points). `grid_mesh` is then ignored; the mesh actually used is
      stored with every structure's output as "grid_mesh".
    - `kpoints_force_2d` / `kpoints_force_1d`: forwarded to `get_ksampling` as `force_2d` /
      `force_1d` (mutually exclusive): the resolved k_grid is set to 1 along c (2D) or along
      a and b (1D). Only used with `kpoints_settings`.
    - `method`: how the members' eigenvalues are obtained in `compute`/`compute_parallel`.
      "exact" (default): every member is diagonalized.
      "projected": only the averaged Hamiltonian is diagonalized; each member is projected onto
      the averaged Hamiltonian's eigenvectors `C_W` of a band window W around the gap (see
      `n_bands_below`/`n_bands_above`) and its window eigenvalues are those of
      `E_W + C_W^dag (H_m - H_avg - c_m S) C_W` (Rayleigh-Ritz in the window, with `c_m` the
      member's rigid shift), plus the downfolding correction below if `downfold`. Needs
      `align_mode="matrix"`. Accurate as long as the members' deviations are small compared to
      the separation from the bands outside the window; a member state entering the window from
      outside it (e.g. a spurious in-gap state) is not seen. With the window covering every band
      it reproduces "exact". The output has the same layout, restricted to the window bands:
      `vbm_index` is relative to the window, `first_band_index` is the window's first band and
      `n_bands_total` the number of bands of the averaged Hamiltonian; `per_model_gap_eV` is the
      gap of the members' window eigenvalues.
    - `n_bands_below` / `n_bands_above`: with `method="projected"`, the number of valence bands
      (counting the VBM band) and of conduction bands (counting the CBM band) in the window; None
      takes every band on that side (unless `window_e` is set).
    - `window_e`: with `method="projected"`, the window by energy instead of band count (eV,
      alternative to `n_bands_below`/`n_bands_above`): it takes every valence band that comes
      within `window_e` of the VBM, and every conduction band within `window_e` of the CBM, at
      some k-point of the grid (band edges over the grid, as for the gap). The number of bands
      then adapts to each structure's band density. One value for both sides, like the `e_cut`
      of the band-edge-weighted statistics in `BandUncertaintyPostProcessor`; use e_cut plus a
      few tau.
    - `downfold`: with `method="projected"`, fold the coupling to every band outside the window
      back into the window block (Loewdin partitioning, second order, `H_oo ~ E_o`): with
      `D = C^dag (H_m - H_avg - c_m S) C` in the averaged Hamiltonian's eigenbasis, the window
      block gets `Sigma_ij = sum_o D_io D_oj (1/(E_i - E_o) + 1/(E_j - E_o)) / 2`. Small
      admixtures of high-energy states, which no window around the gap reaches, otherwise leave
      the window eigenvalues too high. Costs one product of `C_W^dag H_m` with all eigenvectors
      per member and k-point, and the averaged Hamiltonian's eigenvectors of all bands are kept
      during the member loop instead of the window's only. Stops being valid where an outside
      band comes close to a window band, i.e. at the window edges.
    """

    grid_mesh: tuple[int, int, int] = (2, 2, 2)
    kpoints_settings: dict | None = None
    kpoints_force_2d: bool = False
    kpoints_force_1d: bool = False
    anchor_k: tuple[float, float, float] = (0.0, 0.0, 0.0)
    symprec: float = 1e-5
    species_number: dict[str, int] = field(default_factory=dict)  # overrides ase.data.atomic_numbers
    hamiltonian_name: str = "hamiltonian.h5"
    overlap_name: str = "overlap.h5"
    align_mode: str = "matrix"
    aligned_hamiltonian_dir: Path | None = None
    batch_entries: int = 4_000_000
    ill_method: str | None = None
    ill_threshold: float = 1e-3
    method: str = "exact"
    n_bands_below: int | None = None
    n_bands_above: int | None = None
    window_e: float | None = None
    downfold: bool = True

    _ILL_FILL_VALUE = 1e4

    def __post_init__(self):
        if self.align_mode not in ("matrix", "grid", "gamma"):
            raise ValueError(f"align_mode must be 'matrix', 'grid' or 'gamma', got {self.align_mode!r}")
        if self.ill_method == "none":
            self.ill_method = None
        if self.ill_method not in (None, "orbital_removal"):
            raise ValueError(f"ill_method must be None or 'orbital_removal', got {self.ill_method!r}")
        if self.method not in ("exact", "projected"):
            raise ValueError(f"method must be 'exact' or 'projected', got {self.method!r}")
        if self.method == "projected" and self.align_mode != "matrix":
            raise ValueError("method='projected' needs align_mode='matrix'")
        if self.window_e is not None:
            if self.n_bands_below is not None or self.n_bands_above is not None:
                raise ValueError("set either window_e or n_bands_below/n_bands_above, not both")
            if self.window_e < 0:
                raise ValueError(f"window_e must be >= 0, got {self.window_e}")

    def settings(self) -> dict:
        """Settings that change the results, stored with every structure's output."""
        settings = {
            "align_mode": self.align_mode,
            "ill_method": self.ill_method,
            "ill_threshold": self.ill_threshold if self.ill_method else None,
        }
        # only added for "projected", so outputs of the exact method keep their old settings
        if self.method == "projected":
            settings.update(method=self.method, n_bands_below=self.n_bands_below,
                            n_bands_above=self.n_bands_above, window_e=self.window_e,
                            downfold=self.downfold)
        return settings

    def make_ill_handler(self, h_obj, ks, n_jobs: int = -1, parallel_k: bool = True):
        """`IllConditionedHandler` for one structure, or None if `ill_method` is None.

        Only S(k) of `h_obj` is used (shared by all spectra of the structure); no energy
        information from info.json enters.
        """
        if self.ill_method is None:
            return None
        from hamilflow.sparse_hamiltonian.ill_conditioned import IllConditionedHandler

        handler = IllConditionedHandler(
            method=self.ill_method,
            ill_threshold=self.ill_threshold,
            fill_value=self._ILL_FILL_VALUE,
            verbose=False,
        )
        handler.prepare_orbital_truncation(h_obj.get_all_Sk(ks, n_jobs=n_jobs, parallel_k=parallel_k),
                                           n_jobs=n_jobs, parallel_k=parallel_k)
        return handler

    @staticmethod
    def n_kept_orbitals(ill_handler) -> int | None:
        if ill_handler is None or ill_handler.kept_orbitals is None:
            return None
        return len(ill_handler.kept_orbitals)

    def common_bands(self, *eigvals: np.ndarray) -> list[np.ndarray]:
        """Cut every `(n_bands, n_k)` array to the bands that are real (not an ill-conditioning
        fill value) at every k-point in all of them."""
        n_real = min(
            int(np.min(np.sum(e < 0.5 * self._ILL_FILL_VALUE, axis=0))) for e in eigvals
        )
        return [e[:n_real] for e in eigvals]

    def resolve_mesh(self, h_obj) -> tuple[int, int, int]:
        """`grid_mesh`, or the k_grid `get_ksampling` gives for `h_obj`'s cell if
        `kpoints_settings` is set."""
        if self.kpoints_settings is None:
            return tuple(self.grid_mesh)
        from pymatgen.core import Lattice, Structure

        from hamilflow.dft_workflows.kpoints import get_ksampling

        structure = Structure(Lattice(h_obj.lattice), list(h_obj.elements), h_obj.frac_coords)
        ksampling = get_ksampling(structure, kpoints_updates=self.kpoints_settings,
                                  force_2d=self.kpoints_force_2d, force_1d=self.kpoints_force_1d)
        if not ksampling or "k_grid" not in ksampling:
            raise ValueError(
                f"kpoints_settings {self.kpoints_settings!r} must resolve to a k_grid, got {ksampling!r}"
            )
        return tuple(ksampling["k_grid"])

    def build_irreducible_kpoints(self, h_obj, mesh, symprec):
        from ase.data import atomic_numbers

        try:
            species = [self.species_number.get(el) or atomic_numbers[el] for el in h_obj.elements]
        except KeyError as e:
            raise ValueError(f"No species_number tag for element {e}") from e

        cell = (h_obj.lattice, h_obj.frac_coords, species)
        mapping, grid = spglib.get_ir_reciprocal_mesh(mesh, cell, is_shift=[0, 0, 0], symprec=symprec)

        ir_indices = np.unique(mapping)
        weights = np.array([np.sum(mapping == idx) for idx in ir_indices])
        k_frac = grid[ir_indices] / np.array(mesh)

        anchor_idx = int(np.argmin(np.linalg.norm(k_frac - np.array(self.anchor_k), axis=1)))
        assert np.allclose(k_frac[anchor_idx], self.anchor_k, atol=1e-8)
        return k_frac, weights, anchor_idx

    def homo_lumo_indices(self, h_obj):
        if h_obj.occupation is None:
            raise ValueError(f"{h_obj.info_dir_path}: 'occupation' not set in info.json")
        n_elec = h_obj.occupation
        if not h_obj.spinful and n_elec % 2 != 0:
            raise ValueError(f"Odd electron count ({n_elec}) with spinful=False -- unexpected for closed shell")
        n_occ = n_elec // 2 if not h_obj.spinful else n_elec
        return n_occ - 1, n_occ

    def band_edges(self, eigvals, h_obj, anchor_k_idx=None):
        """`(VBM, CBM)` of `eigvals[band, k]`: at `anchor_k_idx` for `align_mode="gamma"`, over
        all k-points otherwise ("grid", and the common reference of "matrix")."""
        homo_idx, lumo_idx = self.homo_lumo_indices(h_obj)
        if self.align_mode == "gamma":
            return float(eigvals[homo_idx, anchor_k_idx]), float(eigvals[lumo_idx, anchor_k_idx])
        return float(eigvals[homo_idx].max()), float(eigvals[lumo_idx].min())

    def align_to_midgap(self, eigvals, h_obj, anchor_k_idx=None):
        vbm, cbm = self.band_edges(eigvals, h_obj, anchor_k_idx)
        shift = -(vbm + cbm) / 2
        return eigvals + shift, shift

    def gap(self, eigvals, h_obj) -> float:
        """Gap on the k-grid, `min_k CBM - max_k VBM` (whatever `align_mode` is)."""
        homo_idx, lumo_idx = self.homo_lumo_indices(h_obj)
        return float(eigvals[lumo_idx].min() - eigvals[homo_idx].max())

    def _resolve_average_hamiltonian_path(
        self,
        structure_name: str,
        model_paths: list[Path],
        average_hamiltonian_dir: Path,
    ) -> Path:
        avg_path = Path(average_hamiltonian_dir) / structure_name / self.hamiltonian_name
        if not avg_path.exists():
            average_predicted_hamiltonians(model_paths, avg_path)
        return avg_path

    def _avg_hamiltonian_obj(
        self,
        structure_name: str,
        model_dirs: list[Path],
        average_hamiltonian_dir: Path,
        share_from: SparseHamiltonianObj | None = None,
    ) -> SparseHamiltonianObj:
        """The models' averaged Hamiltonian as a `SparseHamiltonianObj`.

        The averaged real-space Hamiltonian (see `hamiltonian_io.average_predicted_hamiltonians`)
        is read from `average_hamiltonian_dir` if already there, otherwise computed and written
        there. `info.json`/`overlap.h5` are read from the first model dir, since those are
        structure-level (not per-model) and identical across the ensemble. `share_from` (an
        already loaded Hamiltonian of the structure) is passed on to `SparseHamiltonianObj`.
        """
        model_paths = [Path(d) / structure_name / self.hamiltonian_name for d in model_dirs]
        avg_path = self._resolve_average_hamiltonian_path(structure_name, model_paths, average_hamiltonian_dir)
        return SparseHamiltonianObj(Path(model_dirs[0]) / structure_name, H_file_path=avg_path,
                                    share_from=share_from)

    def _fit_and_save(self, h_paths, ref_path, overlap_dir, labels, structure_name, aligned_to):
        """Rigid shifts of `h_paths` relative to `ref_path` (overlap from `overlap_dir`); with
        `aligned_hamiltonian_dir` set, also write each aligned Hamiltonian under
        `<aligned_hamiltonian_dir>/<label>/<structure>/`."""
        s_path = Path(overlap_dir) / self.overlap_name
        rigid = fit_rigid_shifts(h_paths, ref_path, s_path, self.batch_entries)
        if self.aligned_hamiltonian_dir is not None:
            for h_path, label, r in zip(h_paths, labels, rigid):
                write_aligned_hamiltonian(
                    h_path, s_path, r["c_eV"],
                    Path(self.aligned_hamiltonian_dir) / label / structure_name / self.hamiltonian_name,
                    aligned_to=aligned_to, batch_entries=self.batch_entries,
                )
        return rigid

    @staticmethod
    def _map_k(fn, n_k: int, n_jobs: int = -1, parallel_k: bool = True) -> list:
        """`[fn(i_k) for i_k in range(n_k)]`, with the thread/BLAS split of
        `SparseHamiltonianObj.diag`."""
        if n_jobs < 0:
            n_jobs = os.cpu_count() or 1
        n_workers = max(1, min(n_jobs, n_k)) if parallel_k else 1
        n_blas_threads = max(1, n_jobs // n_workers)
        with threadpoolctl.threadpool_limits(limits=n_blas_threads, user_api="blas"):
            if n_workers <= 1:
                return [fn(i_k) for i_k in range(n_k)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
                return list(ex.map(fn, range(n_k)))

    @staticmethod
    def _eig_with_vecs(h_obj, k, ill_handler) -> tuple[np.ndarray, np.ndarray]:
        """Real eigenvalues and S-orthonormal eigenvectors (rows in the full orbital basis) at one
        k-point; with `ill_handler`, only the eigenvalues of the kept basis."""
        Hk, Sk = h_obj._dense_k(k), h_obj._s_obj._dense_k(k)
        if ill_handler is None:
            return eigh_generalized(Hk, Sk)
        vals, vecs = ill_handler.process_k(Hk, Sk, return_vecs=True)
        return vals[: vecs.shape[1]], vecs

    def _band_window(self, eigvals: np.ndarray, h_obj) -> tuple[int, int]:
        """`[first, stop)` band indices of the `method="projected"` window, from the averaged
        Hamiltonian's `eigvals[band, k]`."""
        n_bands = eigvals.shape[0]
        homo_idx, lumo_idx = self.homo_lumo_indices(h_obj)
        vbm, cbm = self.band_edges(eigvals, h_obj)
        if self.window_e is not None:
            # bands are sorted at every k, so the bands reaching the window are contiguous
            first = int(np.argmax(eigvals[: homo_idx + 1].max(axis=1) >= vbm - self.window_e))
            stop = lumo_idx + int(np.sum(eigvals[lumo_idx:].min(axis=1) <= cbm + self.window_e))
            return first, stop
        if self.n_bands_below is not None:
            first = max(0, homo_idx + 1 - self.n_bands_below)
        else:
            first = 0
        if self.n_bands_above is not None:
            stop = min(n_bands, lumo_idx + self.n_bands_above)
        else:
            stop = n_bands
        return first, stop

    @staticmethod
    def _downfold_correction(e_w: np.ndarray, d_wo: np.ndarray, e_o: np.ndarray) -> np.ndarray:
        """Second-order coupling of the window to the outside bands, symmetrized at the window
        energies: `Sigma_ij = sum_o D_io D_oj (1/(E_i - E_o) + 1/(E_j - E_o)) / 2`, with
        `d_wo[i, o] = D_io` and `D_oj = conj(D_jo)`."""
        a = d_wo / (e_w[:, None] - e_o[None, :])
        return 0.5 * (a @ d_wo.conj().T + d_wo @ a.conj().T)

    def _projected_structure_uncertainty(
        self,
        structure_name: str,
        model_dirs: list[Path],
        average_hamiltonian_dir: Path,
        n_jobs: int = -1,
        parallel_k: bool = True,
        log_models: bool = False,
    ) -> dict:
        """`_structure_uncertainty` for `method="projected"`."""
        ref_obj = SparseHamiltonianObj(model_dirs[0] / structure_name)
        mesh = self.resolve_mesh(ref_obj)
        ks, weights, anchor_k_idx = self.build_irreducible_kpoints(ref_obj, mesh, self.symprec)
        ill_handler = self.make_ill_handler(ref_obj, ks, n_jobs=n_jobs, parallel_k=parallel_k)

        avg_obj = self._avg_hamiltonian_obj(structure_name, model_dirs, average_hamiltonian_dir,
                                            share_from=ref_obj)
        eigs = self._map_k(lambda i_k: self._eig_with_vecs(avg_obj, ks[i_k], ill_handler),
                           len(ks), n_jobs, parallel_k)
        n_bands = min(len(vals) for vals, _ in eigs)
        avg_raw = np.stack([vals[:n_bands] for vals, _ in eigs], axis=1)
        first, stop = self._band_window(avg_raw, ref_obj)
        e_win = [vals[first:stop] for vals, _ in eigs]
        c_win = [vecs[:, first:stop] for _, vecs in eigs]
        if self.downfold:
            # every band outside the window, at every k-point (all real bands of that k-point)
            outside = [np.r_[0:first, stop:len(vals)] for vals, _ in eigs]
            e_out = [vals[o] for (vals, _), o in zip(eigs, outside)]
            c_out = [vecs[:, o] for (_, vecs), o in zip(eigs, outside)]
        # without downfolding only the window's eigenvectors are kept, not the full
        # (n_orb, n_orb) set per k-point
        del eigs
        avg_ham_aligned, avg_ham_shift = self.align_to_midgap(avg_raw, ref_obj, anchor_k_idx)

        member_paths = [Path(d) / structure_name / self.hamiltonian_name for d in model_dirs]
        rigid = self._fit_and_save(member_paths, avg_obj.matrix_path, model_dirs[0] / structure_name,
                                   [f"model_{k + 1}" for k in range(len(member_paths))], structure_name,
                                   aligned_to="ensemble average")
        del avg_obj

        # deviations[m, band, k] of member m's aligned window eigenvalues from the average's
        deviations = []
        for i_model, model in enumerate(model_dirs):
            t_model = time.monotonic()
            # member 0 is the reference; the others share its block layout and overlap
            h_obj = ref_obj if i_model == 0 else SparseHamiltonianObj(model / structure_name, share_from=ref_obj)
            c_m = rigid[i_model]["c_eV"]
            # the sparse C^dag H product beats dense GEMM only for a sparse enough H(k)
            build_k = h_obj._sparse_k if h_obj.fill_fraction < SPARSE_PRODUCT_MAX_FILL else h_obj._dense_k

            # C^dag S C = 1 and C^dag H_avg C = E, so C^dag (H_m - H_avg - c_m S) C = C^dag H_m C - E - c_m
            # on the window block and C_W^dag H_m C_o off it
            def window_deviation(i_k):
                c, e = c_win[i_k], e_win[i_k]
                y = c.conj().T @ build_k(ks[i_k])
                p = y @ c
                p = 0.5 * (p + p.conj().T)
                p[np.diag_indices(len(e))] -= e + c_m
                if self.downfold:
                    p = p + self._downfold_correction(e, y @ c_out[i_k], e_out[i_k])
                return np.linalg.eigvalsh(np.diag(e) + p) - e

            deviations.append(np.stack(self._map_k(window_deviation, len(ks), n_jobs, parallel_k), axis=1))
            del h_obj
            if log_models:
                print(
                    f"[{structure_name}] model {i_model + 1}/{len(model_dirs)} projected "
                    f"in {time.monotonic() - t_model:.1f}s",
                    flush=True,
                )
        deviations = np.stack(deviations, axis=0)

        # the averaged Hamiltonian's eigenvalues are a constant per (band, k), so the spread of the
        # members' eigenvalues is the spread of their deviations
        sigma_eigvals = np.std(deviations, axis=0, ddof=1)
        sigma_eigvals_avg_ham = np.sqrt(np.mean(deviations ** 2, axis=0))
        avg_window = avg_ham_aligned[first:stop]
        homo_w = self.homo_lumo_indices(ref_obj)[0] - first
        member_window = avg_raw[first:stop][None] + deviations
        member_gaps = member_window[:, homo_w + 1].min(axis=1) - member_window[:, homo_w].max(axis=1)

        result_per_k = {}
        for i_k in range(len(ks)):
            result_per_k[f"k{i_k}"] = {
                "k_frac": ks[i_k].tolist(),
                "weight": int(weights[i_k]),
                "sigma_eV": sigma_eigvals[:, i_k].tolist(),
                "sigma_eV_avg_ham": sigma_eigvals_avg_ham[:, i_k].tolist(),
                "eigvals_avg_ham_eV": avg_window[:, i_k].tolist(),
            }

        return {
            **self.settings(),
            "grid_mesh": list(mesh),
            "n_irreducible_kpoints": len(ks),
            "n_bands": stop - first,
            "n_bands_total": n_bands,
            "first_band_index": first,
            "n_kept_orbitals": self.n_kept_orbitals(ill_handler),
            "occupation": ref_obj.occupation,
            "vbm_index": homo_w,
            "per_model_shift_eV": [float(avg_ham_shift - r["c_eV"]) for r in rigid],
            "avg_hamiltonian_shift_eV": float(avg_ham_shift),
            "per_model_rigid_shift_eV": [r["c_eV"] for r in rigid],
            "per_model_rigid_shift_residual": [r["residual"] for r in rigid],
            "per_model_gap_eV": [float(g) for g in member_gaps],
            "avg_hamiltonian_gap_eV": self.gap(avg_raw, ref_obj),
            "kpoints": result_per_k,
        }

    def _structure_uncertainty(
        self,
        structure_name: str,
        model_dirs: list[Path],
        average_hamiltonian_dir: Path,
        n_jobs: int = -1,
        parallel_k: bool = True,
        log_models: bool = False,
    ) -> dict:
        """Band uncertainty of one structure (shared by `compute` and `compute_parallel`)."""
        if self.method == "projected":
            return self._projected_structure_uncertainty(
                structure_name, model_dirs, average_hamiltonian_dir, n_jobs, parallel_k, log_models
            )
        ref_obj = SparseHamiltonianObj(model_dirs[0] / structure_name)
        mesh = self.resolve_mesh(ref_obj)
        ks, weights, anchor_k_idx = self.build_irreducible_kpoints(ref_obj, mesh, self.symprec)
        ill_handler = self.make_ill_handler(ref_obj, ks, n_jobs=n_jobs, parallel_k=parallel_k)
        diag_kwargs = dict(bands_only=True, n_jobs=n_jobs, parallel_k=parallel_k, ill_handler=ill_handler)

        raws = []
        for i_model, model in enumerate(model_dirs):
            t_model = time.monotonic()
            # member 0 is the reference; the others share its block layout and overlap
            h_obj = ref_obj if i_model == 0 else SparseHamiltonianObj(model / structure_name, share_from=ref_obj)
            raws.append(h_obj.diag(ks, **diag_kwargs))
            del h_obj
            if log_models:
                print(
                    f"[{structure_name}] model {i_model + 1}/{len(model_dirs)} done "
                    f"in {time.monotonic() - t_model:.1f}s",
                    flush=True,
                )
        avg_obj = self._avg_hamiltonian_obj(structure_name, model_dirs, average_hamiltonian_dir,
                                            share_from=ref_obj)
        avg_raw = avg_obj.diag(ks, **diag_kwargs)
        *raws, avg_raw = self.common_bands(*raws, avg_raw)

        # occupation/spinful come from info.json, identical for all members and the average
        avg_ham_aligned, avg_ham_shift = self.align_to_midgap(avg_raw, ref_obj, anchor_k_idx)
        rigid = None
        if self.align_mode == "matrix":
            member_paths = [Path(d) / structure_name / self.hamiltonian_name for d in model_dirs]
            rigid = self._fit_and_save(member_paths, avg_obj.matrix_path, model_dirs[0] / structure_name,
                                       [f"model_{k + 1}" for k in range(len(member_paths))], structure_name,
                                       aligned_to="ensemble average")
            shifts = [avg_ham_shift - r["c_eV"] for r in rigid]
            aligned_eigvals = [raw + s for raw, s in zip(raws, shifts)]
        else:
            aligned_eigvals, shifts = zip(*(self.align_to_midgap(raw, ref_obj, anchor_k_idx) for raw in raws))

        aligned_stack = np.stack(aligned_eigvals, axis=0)
        sigma_eigvals = np.std(aligned_stack, axis=0, ddof=1)
        # Deviation from the averaged Hamiltonian's own eigenvalues (a fixed
        # reference, not estimated from this sample) -- no ddof correction needed.
        sigma_eigvals_avg_ham = np.sqrt(np.mean((aligned_stack - avg_ham_aligned[None, :, :]) ** 2, axis=0))

        n_irr = len(ks)
        result_per_k = {}
        for i_k in range(n_irr):
            result_per_k[f"k{i_k}"] = {
                "k_frac": ks[i_k].tolist(),
                "weight": int(weights[i_k]),
                "sigma_eV": sigma_eigvals[:, i_k].tolist(),
                "sigma_eV_avg_ham": sigma_eigvals_avg_ham[:, i_k].tolist(),
                "eigvals_avg_ham_eV": avg_ham_aligned[:, i_k].tolist(),
            }

        return {
            **self.settings(),
            "grid_mesh": list(mesh),
            "n_irreducible_kpoints": n_irr,
            "n_bands": int(avg_raw.shape[0]),
            "n_kept_orbitals": self.n_kept_orbitals(ill_handler),
            "occupation": ref_obj.occupation,
            "vbm_index": self.homo_lumo_indices(ref_obj)[0],
            "per_model_shift_eV": [float(s) for s in shifts],
            "avg_hamiltonian_shift_eV": float(avg_ham_shift),
            **({"per_model_rigid_shift_eV": [r["c_eV"] for r in rigid],
                "per_model_rigid_shift_residual": [r["residual"] for r in rigid]} if rigid else {}),
            "per_model_gap_eV": [self.gap(raw, ref_obj) for raw in raws],
            "avg_hamiltonian_gap_eV": self.gap(avg_raw, ref_obj),
            "kpoints": result_per_k,
        }

    @staticmethod
    def _write_output_atomic(output: dict, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(output, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, output_path)

    @staticmethod
    def _load_existing_output(output_path: Path | None) -> dict:
        if output_path is None or not output_path.exists():
            return {}
        with open(output_path) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} finished structures from {output_path}", flush=True)
        return existing

    @classmethod
    def _init_output(cls, output_path: Path | None, skip_existing: bool) -> dict:
        """Return the starting result dict. When not resuming, an existing output file is
        atomically reset to `{}` so an interrupted run cannot leave stale results behind."""
        if skip_existing:
            return cls._load_existing_output(output_path)
        if output_path is not None:
            cls._write_output_atomic({}, output_path)
        return {}

    def _drop_done(self, structures: list[str], output: dict) -> list[str]:
        todo = [s for s in structures if s not in output]
        if len(todo) < len(structures):
            print(f"Skipping {len(structures) - len(todo)} already-computed structures", flush=True)
        settings = self.settings()
        other = [s for s, res in output.items() if any(res.get(k) != v for k, v in settings.items())]
        if other:
            print(
                f"WARNING: {len(other)} stored structures were computed with settings other than "
                f"{settings} (or before settings were stored) and are kept as-is; "
                "pass skip_existing=False to recompute them",
                flush=True,
            )
        return todo

    def compute(
        self,
        model_dirs: Iterable[Path],
        average_hamiltonian_dir: Path,
        structure_pattern: str | None = None,
        exclude_structures: Iterable[str] | None = None,
        n_jobs: int = -1,
        parallel_k: bool = True,
        output_path: Path | str | None = None,
        skip_existing: bool = True,
    ):
        """
        - `output_path`: if given, the accumulated result dict is written to this path
          (atomically) after every structure completes, so a killed/timed-out job still
          leaves the results computed so far on disk.
        - `skip_existing`: if True and `output_path` already exists, structures already in it
          are not recomputed and their stored results are kept in the returned dict. Set to
          False to recompute everything (the file is then overwritten). Note that stored
          results are reused as-is, even if `grid_mesh`/`kpoints_settings` or the models changed (a warning is
          printed when they were computed with different `settings()`).
        - `average_hamiltonian_dir`: root containing (or to receive) each structure's averaged
          `hamiltonian.h5` (see `hamiltonian_io.average_predicted_hamiltonians`), diagonalized to
          get the `sigma_eV_avg_ham` reference below. If a structure's average is already there
          it is read as-is; otherwise it is computed and written there. Its midgap-aligned
          eigenvalues are also stored per k-point as `eigvals_avg_ham_eV` (same band order as
          `sigma_eV_avg_ham`), for energy-dependent weighting in post-processing.
        - `n_jobs`: CPU budget for every diagonalization (-1 = all cores). Passed to
          `SparseHamiltonianObj.diag`.
        - `parallel_k`: if True, k-points are spread over threads (leftover budget goes to each
          thread's BLAS calls); if False, k-points run serially with `n_jobs` BLAS threads each.
        """
        model_dirs = [Path(p) for p in model_dirs]
        average_hamiltonian_dir = Path(average_hamiltonian_dir)
        structures = self._list_structures(model_dirs[0], structure_pattern, exclude_structures)

        if output_path is not None:
            output_path = Path(output_path)
        output = self._init_output(output_path, skip_existing)
        structures = self._drop_done(structures, output)

        for structure_name in structures:
            output[structure_name] = self._structure_uncertainty(
                structure_name, model_dirs, average_hamiltonian_dir, n_jobs=n_jobs, parallel_k=parallel_k
            )

            if output_path is not None:
                self._write_output_atomic(output, output_path)
                print(f"[{structure_name}] wrote {len(output)} structures to {output_path}", flush=True)

        return output

    def _compute_structure(
        self,
        structure_name: str,
        model_dirs: list[Path],
        average_hamiltonian_dir: Path,
        blas_threads_per_worker: int = 1,
    ):
        """Compute uncertainty for a single structure (helper for parallel runs)."""
        t_start = time.monotonic()
        print(f"[{structure_name}] starting ({len(model_dirs)} models, pid={os.getpid()})", flush=True)

        # compute_parallel already parallelizes over structures at the process
        # level (one worker per structure, capped at max_workers), so k-point
        # threading is disabled here to avoid oversubscribing on top of that.
        # Each worker still gets a fair share of the machine's cores for its
        # own BLAS calls (see max_workers sizing in compute_parallel) instead
        # of being pinned to 1 thread -- otherwise a structure whose diagonalization
        # dominates the runtime (a large Hamiltonian, or one outlier after its
        # siblings finish) leaves the rest of the machine idle.
        result = self._structure_uncertainty(
            structure_name, model_dirs, average_hamiltonian_dir,
            n_jobs=blas_threads_per_worker, parallel_k=False, log_models=True,
        )
        print(f"[{structure_name}] finished in {time.monotonic() - t_start:.1f}s", flush=True)
        return structure_name, result

    def compute_parallel(
        self,
        model_dirs: Iterable[Path],
        average_hamiltonian_dir: Path,
        structure_pattern: str | None = None,
        max_workers: int | None = None,
        output_path: Path | str | None = None,
        exclude_structures: Iterable[str] | None = None,
        skip_existing: bool = True,
    ):
        """Parallelized version of `compute` that runs per-structure work in separate processes.

        - `max_workers`: number of worker processes (defaults to number of CPU cores).
        - `output_path`: if given, the accumulated result dict is written to this path
          (atomically) after every structure completes, so a killed/timed-out job still
          leaves the results computed so far on disk.
        - `exclude_structures`: structure names to skip.
        - `skip_existing`: see `compute`.
        - `average_hamiltonian_dir`: see `compute`.
        """
        model_dirs = [Path(p) for p in model_dirs]
        average_hamiltonian_dir = Path(average_hamiltonian_dir)
        structures = self._list_structures(model_dirs[0], structure_pattern, exclude_structures)

        if output_path is not None:
            output_path = Path(output_path)
        output = self._init_output(output_path, skip_existing)
        structures = self._drop_done(structures, output)
        if not structures:
            return output

        if max_workers is None:
            max_workers = min(len(structures), os.cpu_count() or 1)

        # Split the machine's cores evenly across worker processes so each
        # process's (sequential, parallel_k=False) diagonalizations still use
        # more than one BLAS thread when there are fewer workers than cores
        # (e.g. few structures, or large structures relative to core count).
        blas_threads_per_worker = max(1, (os.cpu_count() or 1) // max_workers)

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(
                    self._compute_structure, s, model_dirs, average_hamiltonian_dir, blas_threads_per_worker
                ): s
                for s in structures
            }
            for fut in concurrent.futures.as_completed(futures):
                name, res = fut.result()
                output[name] = res
                if output_path is not None:
                    self._write_output_atomic(output, output_path)
                    print(f"[{name}] wrote {len(output)} structures to {output_path}", flush=True)

        return output

    @staticmethod
    def _list_structures(
        root: Path,
        structure_pattern: str | None = None,
        exclude_structures: Iterable[str] | None = None,
    ) -> list[str]:
        if structure_pattern:
            structures = [p.name for p in (root / structure_pattern).parent.glob(structure_pattern)]
        else:
            structures = [p.name for p in root.glob("*") if p.is_dir()]
        if exclude_structures:
            excluded = set(exclude_structures)
            structures = [s for s in structures if s not in excluded]
        return structures

    def _compare_structure(
        self,
        structure_name: str,
        model_dirs: list[Path],
        average_hamiltonian_dir: Path,
        dft_root: Path,
        n_jobs: int = -1,
        parallel_k: bool = True,
    ):
        """Compare the models' averaged Hamiltonian to DFT for a single structure."""
        t_start = time.monotonic()
        print(f"[{structure_name}] starting comparison to DFT (pid={os.getpid()})", flush=True)

        dft_obj = SparseHamiltonianObj(dft_root / structure_name)

        mesh = self.resolve_mesh(dft_obj)
        ks, weights, anchor_k_idx = self.build_irreducible_kpoints(dft_obj, mesh, self.symprec)
        ill_handler = self.make_ill_handler(dft_obj, ks, n_jobs=n_jobs, parallel_k=parallel_k)
        diag_kwargs = dict(bands_only=True, n_jobs=n_jobs, parallel_k=parallel_k, ill_handler=ill_handler)

        avg_obj = self._avg_hamiltonian_obj(structure_name, model_dirs, average_hamiltonian_dir,
                                            share_from=dft_obj)
        avg_raw, dft_raw = self.common_bands(avg_obj.diag(ks, **diag_kwargs), dft_obj.diag(ks, **diag_kwargs))
        dft_aligned, dft_shift = self.align_to_midgap(dft_raw, dft_obj, anchor_k_idx)
        rigid = None
        if self.align_mode == "matrix":
            rigid = self._fit_and_save([avg_obj.matrix_path], dft_root / structure_name / self.hamiltonian_name,
                                       dft_root / structure_name, ["average_to_dft"], structure_name,
                                       aligned_to="DFT")[0]
            avg_aligned = avg_raw - rigid["c_eV"] + dft_shift
        else:
            avg_aligned, _ = self.align_to_midgap(avg_raw, dft_obj, anchor_k_idx)

        abs_err = np.abs(avg_aligned - dft_aligned)

        per_k = {}
        mae_values = []
        for i_k in range(len(ks)):
            vals = abs_err[:, i_k].tolist()
            per_k[f"k{i_k}"] = {
                "k_frac": ks[i_k].tolist(),
                "weight": int(weights[i_k]),
                "abs_err_eV": vals,
                "eigvals_avg_ham_eV": avg_aligned[:, i_k].tolist(),
            }
            mae_values.append(float(np.mean(vals)))

        overall_mae = float(np.mean(mae_values))

        print(f"[{structure_name}] finished in {time.monotonic() - t_start:.1f}s", flush=True)
        return structure_name, {
            **self.settings(),
            "grid_mesh": list(mesh),
            "overall_mae_eV": overall_mae,
            **({"avg_vs_dft_rigid_shift_eV": rigid["c_eV"],
                "avg_vs_dft_rigid_shift_residual": rigid["residual"]} if rigid else {}),
            "n_bands": int(avg_raw.shape[0]),
            "n_kept_orbitals": self.n_kept_orbitals(ill_handler),
            "gap_avg_ham_eV": self.gap(avg_raw, dft_obj),
            "gap_dft_eV": self.gap(dft_raw, dft_obj),
            "occupation": dft_obj.occupation,
            "vbm_index": self.homo_lumo_indices(dft_obj)[0],
            "kpoints": per_k,
        }

    def compare_averaged_to_dft(
        self,
        model_dirs: Iterable[Path],
        average_hamiltonian_dir: Path,
        dft_root: Path,
        structure_pattern: str | None = None,
        exclude_structures: Iterable[str] | None = None,
        n_jobs: int = -1,
        parallel_k: bool = True,
        output_path: Path | str | None = None,
        skip_existing: bool = True,
    ):
        """Compare the models' averaged Hamiltonian to the DFT reference under `dft_root`.

        Returns a dict keyed by structure with MAE and per-k error lists similar to `compute`.

        - `average_hamiltonian_dir`: see `compute`. Each structure's averaged `hamiltonian.h5`
          is read from there if present, otherwise averaged from `model_dirs` and written there.
          `info.json`/`overlap.h5` are taken from `model_dirs[0]`.
        - `output_path`, `skip_existing`, `n_jobs`, `parallel_k`: see `compute`.
        - `exclude_structures`: structure names to skip.
        """
        model_dirs = [Path(p) for p in model_dirs]
        average_hamiltonian_dir = Path(average_hamiltonian_dir)
        dft_root = Path(dft_root)
        structures = self._list_structures(model_dirs[0], structure_pattern, exclude_structures)

        if output_path is not None:
            output_path = Path(output_path)
        output = self._init_output(output_path, skip_existing)
        structures = self._drop_done(structures, output)

        for structure_name in structures:
            name, res = self._compare_structure(
                structure_name, model_dirs, average_hamiltonian_dir, dft_root,
                n_jobs=n_jobs, parallel_k=parallel_k,
            )
            output[name] = res
            if output_path is not None:
                self._write_output_atomic(output, output_path)
                print(f"[{name}] wrote {len(output)} structures to {output_path}", flush=True)

        return output

    def compare_averaged_to_dft_parallel(
        self,
        model_dirs: Iterable[Path],
        average_hamiltonian_dir: Path,
        dft_root: Path,
        structure_pattern: str | None = None,
        max_workers: int | None = None,
        output_path: Path | str | None = None,
        exclude_structures: Iterable[str] | None = None,
        skip_existing: bool = True,
    ):
        """Parallelized version of `compare_averaged_to_dft` that runs per-structure work in
        separate processes. Arguments as in `compare_averaged_to_dft` / `compute_parallel`.
        """
        model_dirs = [Path(p) for p in model_dirs]
        average_hamiltonian_dir = Path(average_hamiltonian_dir)
        dft_root = Path(dft_root)
        structures = self._list_structures(model_dirs[0], structure_pattern, exclude_structures)

        if output_path is not None:
            output_path = Path(output_path)
        output = self._init_output(output_path, skip_existing)
        structures = self._drop_done(structures, output)
        if not structures:
            return output

        if max_workers is None:
            max_workers = min(len(structures), os.cpu_count() or 1)

        # See compute_parallel: k-point threading off inside workers, cores split evenly
        # across processes for BLAS.
        blas_threads_per_worker = max(1, (os.cpu_count() or 1) // max_workers)

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(
                    self._compare_structure, s, model_dirs, average_hamiltonian_dir, dft_root,
                    blas_threads_per_worker, False,
                ): s
                for s in structures
            }
            for fut in concurrent.futures.as_completed(futures):
                name, res = fut.result()
                output[name] = res
                if output_path is not None:
                    self._write_output_atomic(output, output_path)
                    print(f"[{name}] wrote {len(output)} structures to {output_path}", flush=True)

        return output

__all__ = ["BandUncertaintyCalculator"]
