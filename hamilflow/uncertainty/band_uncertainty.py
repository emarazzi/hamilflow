from pathlib import Path
from typing import Iterable
import numpy as np
from dataclasses import dataclass, field
import concurrent.futures
import json
import os
import time

from hamilflow.sparse_hamiltonian import SparseHamiltonianObj

from .hamiltonian_io import average_predicted_hamiltonians

import spglib

@dataclass
class BandUncertaintyCalculator:
    """Compute band-energy uncertainty across an ensemble of models.

    Example usage:
      calc = BandUncertaintyCalculator()
      output = calc.compute(model_dirs, dft_dir)

    - `align_mode`: reference energy every spectrum is shifted to zero before comparison.
      "grid" (default): midgap between the VBM (max over the k-grid of the highest occupied
      band) and the CBM (min over the k-grid of the lowest unoccupied band), i.e. the true
      gap as resolved by `grid_mesh`. "gamma": midgap between HOMO and LUMO at `anchor_k`
      (the behavior before `align_mode` existed).
    - `ill_method`: None or "orbital_removal" -- handling of an ill-conditioned overlap,
      applied to every diagonalization (members, averaged Hamiltonian, DFT). Orbitals are removed
      (`deepx_dock`'s global orbital truncation) until the smallest eigenvalue of S(k) on the
      k-grid exceeds `ill_threshold`. The choice depends on S only, so it is made once per
      structure and all spectra of a structure are computed in the same reduced basis; the
      removed eigenvalue slots are dropped. "window_regularization" (also offered by
      `IllConditionedHandler`) is not supported: its energy window would have to be placed
      relative to the Fermi energy in info.json, which is copied from the DFT calculation and
      is not the Fermi level of a predicted Hamiltonian.
    """

    grid_mesh: tuple[int, int, int] = (2, 2, 2)
    anchor_k: tuple[float, float, float] = (0.0, 0.0, 0.0)
    symprec: float = 1e-5
    species_number: dict[str, int] = field(default_factory=lambda: {"Mo": 42, "S": 16})
    hamiltonian_name: str = "hamiltonian.h5"
    align_mode: str = "grid"
    ill_method: str | None = None
    ill_threshold: float = 1e-3

    _ILL_FILL_VALUE = 1e4

    def __post_init__(self):
        if self.align_mode not in ("grid", "gamma"):
            raise ValueError(f"align_mode must be 'grid' or 'gamma', got {self.align_mode!r}")
        if self.ill_method == "none":
            self.ill_method = None
        if self.ill_method not in (None, "orbital_removal"):
            raise ValueError(f"ill_method must be None or 'orbital_removal', got {self.ill_method!r}")

    def settings(self) -> dict:
        """Settings that change the results, stored with every structure's output."""
        return {
            "align_mode": self.align_mode,
            "ill_method": self.ill_method,
            "ill_threshold": self.ill_threshold if self.ill_method else None,
        }

    def make_ill_handler(self, h_obj, ks, n_jobs: int = -1, parallel_k: bool = True):
        """`IllConditionedHandler` for one structure, or None if `ill_method` is None.

        Only S(k) of `h_obj` is used (shared by all spectra of the structure); no energy
        information from info.json enters.
        """
        if self.ill_method is None:
            return None
        from deepx_dock.compute.eigen.ill_conditioned import IllConditionedHandler

        handler = IllConditionedHandler(
            method=self.ill_method,
            ill_threshold=self.ill_threshold,
            fill_value=self._ILL_FILL_VALUE,
            verbose=False,
        )
        handler.prepare_orbital_truncation(h_obj.get_all_Sk(ks, n_jobs=n_jobs, parallel_k=parallel_k))
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

    def build_irreducible_kpoints(self, h_obj, mesh, symprec):
        try:
            species = [self.species_number[el] for el in h_obj.elements]
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
        """`(VBM, CBM)` of `eigvals[band, k]`: over all k-points for `align_mode="grid"`, at
        `anchor_k_idx` for `align_mode="gamma"`."""
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
    ) -> SparseHamiltonianObj:
        """The models' averaged Hamiltonian as a `SparseHamiltonianObj`.

        The averaged real-space Hamiltonian (see `hamiltonian_io.average_predicted_hamiltonians`)
        is read from `average_hamiltonian_dir` if already there, otherwise computed and written
        there. `info.json`/`overlap.h5` are read from the first model dir, since those are
        structure-level (not per-model) and identical across the ensemble.
        """
        model_paths = [Path(d) / structure_name / self.hamiltonian_name for d in model_dirs]
        avg_path = self._resolve_average_hamiltonian_path(structure_name, model_paths, average_hamiltonian_dir)
        return SparseHamiltonianObj(Path(model_dirs[0]) / structure_name, H_file_path=avg_path)

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
        ref_obj = SparseHamiltonianObj(model_dirs[0] / structure_name)
        ks, weights, anchor_k_idx = self.build_irreducible_kpoints(ref_obj, self.grid_mesh, self.symprec)
        ill_handler = self.make_ill_handler(ref_obj, ks, n_jobs=n_jobs, parallel_k=parallel_k)
        diag_kwargs = dict(bands_only=True, n_jobs=n_jobs, parallel_k=parallel_k, ill_handler=ill_handler)

        raws = []
        for i_model, model in enumerate(model_dirs):
            t_model = time.monotonic()
            raws.append(SparseHamiltonianObj(model / structure_name).diag(ks, **diag_kwargs))
            if log_models:
                print(
                    f"[{structure_name}] model {i_model + 1}/{len(model_dirs)} done "
                    f"in {time.monotonic() - t_model:.1f}s",
                    flush=True,
                )
        avg_obj = self._avg_hamiltonian_obj(structure_name, model_dirs, average_hamiltonian_dir)
        avg_raw = avg_obj.diag(ks, **diag_kwargs)
        *raws, avg_raw = self.common_bands(*raws, avg_raw)

        # occupation/spinful come from info.json, identical for all members and the average
        aligned_eigvals, shifts = zip(*(self.align_to_midgap(raw, ref_obj, anchor_k_idx) for raw in raws))
        avg_ham_aligned, avg_ham_shift = self.align_to_midgap(avg_raw, ref_obj, anchor_k_idx)

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
            "grid_mesh": list(self.grid_mesh),
            "n_irreducible_kpoints": n_irr,
            "n_bands": int(avg_raw.shape[0]),
            "n_kept_orbitals": self.n_kept_orbitals(ill_handler),
            "occupation": ref_obj.occupation,
            "vbm_index": self.homo_lumo_indices(ref_obj)[0],
            "per_model_shift_eV": [float(s) for s in shifts],
            "avg_hamiltonian_shift_eV": float(avg_ham_shift),
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
          results are reused as-is, even if `grid_mesh` or the models changed (a warning is
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

        ks, weights, anchor_k_idx = self.build_irreducible_kpoints(dft_obj, self.grid_mesh, self.symprec)
        ill_handler = self.make_ill_handler(dft_obj, ks, n_jobs=n_jobs, parallel_k=parallel_k)
        diag_kwargs = dict(bands_only=True, n_jobs=n_jobs, parallel_k=parallel_k, ill_handler=ill_handler)

        avg_obj = self._avg_hamiltonian_obj(structure_name, model_dirs, average_hamiltonian_dir)
        avg_raw, dft_raw = self.common_bands(avg_obj.diag(ks, **diag_kwargs), dft_obj.diag(ks, **diag_kwargs))
        avg_aligned, _ = self.align_to_midgap(avg_raw, dft_obj, anchor_k_idx)
        dft_aligned, _ = self.align_to_midgap(dft_raw, dft_obj, anchor_k_idx)

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
            "overall_mae_eV": overall_mae,
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
