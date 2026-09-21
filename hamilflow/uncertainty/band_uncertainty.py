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
    """

    grid_mesh: tuple[int, int, int] = (2, 2, 2)
    anchor_k: tuple[float, float, float] = (0.0, 0.0, 0.0)
    symprec: float = 1e-5
    window_ev: float | None = None
    species_number: dict[str, int] = field(default_factory=lambda: {"Mo": 42, "S": 16})
    hamiltonian_name: str = "hamiltonian.h5"

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

    def align_to_midgap(self, eigvals, h_obj, anchor_k_idx):
        homo_idx, lumo_idx = self.homo_lumo_indices(h_obj)
        mid_gap = (eigvals[homo_idx, anchor_k_idx] + eigvals[lumo_idx, anchor_k_idx]) / 2
        shift = -mid_gap
        return eigvals + shift, shift

    def band_window_mask(self, eigvals, window_ev):
        if window_ev is None:
            return np.ones_like(eigvals, dtype=bool)
        return np.abs(eigvals) <= window_ev

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

    def _average_hamiltonian_aligned_eigvals(
        self,
        structure_name: str,
        model_dirs: list[Path],
        ks: np.ndarray,
        anchor_k_idx: int,
        average_hamiltonian_dir: Path,
        n_jobs: int = -1,
        parallel_k: bool = True,
    ):
        """Diagonalize the models' averaged Hamiltonian and align it to midgap.

        The averaged real-space Hamiltonian (see `hamiltonian_io.average_predicted_hamiltonians`)
        is read from `average_hamiltonian_dir` if already there, otherwise computed and written
        there. `info.json`/`overlap.h5` are read from the first model dir, since those are
        structure-level (not per-model) and identical across the ensemble.
        """
        model_paths = [Path(d) / structure_name / self.hamiltonian_name for d in model_dirs]
        avg_path = self._resolve_average_hamiltonian_path(structure_name, model_paths, average_hamiltonian_dir)
        avg_h_obj = SparseHamiltonianObj(Path(model_dirs[0]) / structure_name, H_file_path=avg_path)
        avg_raw = avg_h_obj.diag(ks, bands_only=True, n_jobs=n_jobs, parallel_k=parallel_k)
        return self.align_to_midgap(avg_raw, avg_h_obj, anchor_k_idx)

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

    @staticmethod
    def _drop_done(structures: list[str], output: dict) -> list[str]:
        todo = [s for s in structures if s not in output]
        if len(todo) < len(structures):
            print(f"Skipping {len(structures) - len(todo)} already-computed structures", flush=True)
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
          results are reused as-is, even if `grid_mesh`, `window_ev` or the models changed.
        - `average_hamiltonian_dir`: root containing (or to receive) each structure's averaged
          `hamiltonian.h5` (see `hamiltonian_io.average_predicted_hamiltonians`), diagonalized to
          get the `sigma_eV_avg_ham` reference below. If a structure's average is already there
          it is read as-is; otherwise it is computed and written there.
        - `n_jobs`: CPU budget for every diagonalization (-1 = all cores). Passed to
          `SparseHamiltonianObj.diag`.
        - `parallel_k`: if True, k-points are spread over threads (leftover budget goes to each
          thread's BLAS calls); if False, k-points run serially with `n_jobs` BLAS threads each.
        """
        model_dirs = [Path(p) for p in model_dirs]
        average_hamiltonian_dir = Path(average_hamiltonian_dir)
        if structure_pattern:
            structures = [p.name for p in (model_dirs[0] / structure_pattern).parent.glob(structure_pattern)]
        else:
            structures = [p.name for p in (model_dirs[0] / "*").parent.glob("*") if (model_dirs[0] / p).is_dir()]
        if exclude_structures:
            excluded = set(exclude_structures)
            structures = [s for s in structures if s not in excluded]

        if output_path is not None:
            output_path = Path(output_path)
        output = self._init_output(output_path, skip_existing)
        structures = self._drop_done(structures, output)

        for structure_name in structures:
            h_obj = SparseHamiltonianObj(model_dirs[0] / structure_name)
            ks, weights, anchor_k_idx = self.build_irreducible_kpoints(h_obj, self.grid_mesh, self.symprec)
            occupation = h_obj.occupation

            n_irr = len(ks)

            aligned_eigvals = []
            shifts = []
            for model in model_dirs:
                h_obj = SparseHamiltonianObj(model / structure_name)
                raw = h_obj.diag(ks, bands_only=True, n_jobs=n_jobs, parallel_k=parallel_k)
                aligned, shift = self.align_to_midgap(raw, h_obj, anchor_k_idx)
                aligned_eigvals.append(aligned)
                shifts.append(shift)

            avg_ham_aligned, avg_ham_shift = self._average_hamiltonian_aligned_eigvals(
                structure_name, model_dirs, ks, anchor_k_idx, average_hamiltonian_dir,
                n_jobs=n_jobs, parallel_k=parallel_k,
            )

            window_mask = self.band_window_mask(aligned_eigvals[0], self.window_ev)

            aligned_stack = np.stack(aligned_eigvals, axis=0)
            #sigma_eigvals = np.std(aligned_stack, axis=0, ddof=1)
            # Deviation from the averaged Hamiltonian's own eigenvalues (a fixed
            # reference, not estimated from this sample) -- no ddof correction needed.
            sigma_eigvals_avg_ham = np.sqrt(np.mean((aligned_stack - avg_ham_aligned[None, :, :]) ** 2, axis=0))

            result_per_k = {}
            for i_k in range(n_irr):
                mask_k = window_mask[:, i_k]
                result_per_k[f"k{i_k}"] = {
                    "k_frac": ks[i_k].tolist(),
                    "weight": int(weights[i_k]),
                    #"sigma_eV": sigma_eigvals[mask_k, i_k].tolist(),
                    "sigma_eV_avg_ham": sigma_eigvals_avg_ham[mask_k, i_k].tolist(),
                    "n_bands_in_window": int(mask_k.sum()),
                }

            output[structure_name] = {
                "grid_mesh": list(self.grid_mesh),
                "n_irreducible_kpoints": n_irr,
                "occupation": occupation,
                "per_model_shift_eV": shifts,
                "avg_hamiltonian_shift_eV": avg_ham_shift,
                "kpoints": result_per_k,
            }

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

        ks_obj = SparseHamiltonianObj(model_dirs[0] / structure_name)
        ks, weights, anchor_k_idx = self.build_irreducible_kpoints(ks_obj, self.grid_mesh, self.symprec)
        occupation = ks_obj.occupation

        aligned_eigvals = []
        shifts = []
        for i_model, model in enumerate(model_dirs):
            t_model = time.monotonic()
            h_obj = SparseHamiltonianObj(model / structure_name)
            # compute_parallel already parallelizes over structures at the process
            # level (one worker per structure, capped at max_workers), so k-point
            # threading is disabled here to avoid oversubscribing on top of that.
            # Each worker still gets a fair share of the machine's cores for its
            # own BLAS calls (see max_workers sizing in compute_parallel) instead
            # of being pinned to 1 thread -- otherwise a structure whose diagonalization
            # dominates the runtime (a large Hamiltonian, or one outlier after its
            # siblings finish) leaves the rest of the machine idle.
            raw = h_obj.diag(ks, bands_only=True, n_jobs=blas_threads_per_worker, parallel_k=False)
            aligned, shift = self.align_to_midgap(raw, h_obj, anchor_k_idx)
            aligned_eigvals.append(aligned)
            shifts.append(shift)
            print(
                f"[{structure_name}] model {i_model + 1}/{len(model_dirs)} done "
                f"in {time.monotonic() - t_model:.1f}s",
                flush=True,
            )

        avg_ham_aligned, avg_ham_shift = self._average_hamiltonian_aligned_eigvals(
            structure_name, model_dirs, ks, anchor_k_idx, average_hamiltonian_dir
        )

        window_mask = self.band_window_mask(aligned_eigvals[0], self.window_ev)

        aligned_stack = np.stack(aligned_eigvals, axis=0)
        sigma_eigvals = np.std(aligned_stack, axis=0, ddof=1)
        # Deviation from the averaged Hamiltonian's own eigenvalues (a fixed
        # reference, not estimated from this sample) -- no ddof correction needed.
        sigma_eigvals_avg_ham = np.sqrt(np.mean((aligned_stack - avg_ham_aligned[None, :, :]) ** 2, axis=0))

        n_irr = len(ks)
        result_per_k = {}
        for i_k in range(n_irr):
            mask_k = window_mask[:, i_k]
            result_per_k[f"k{i_k}"] = {
                "k_frac": ks[i_k].tolist(),
                "weight": int(weights[i_k]),
                "sigma_eV": sigma_eigvals[mask_k, i_k].tolist(),
                "sigma_eV_avg_ham": sigma_eigvals_avg_ham[mask_k, i_k].tolist(),
                "n_bands_in_window": int(mask_k.sum()),
            }

        print(f"[{structure_name}] finished in {time.monotonic() - t_start:.1f}s", flush=True)
        return structure_name, {
            "grid_mesh": list(self.grid_mesh),
            "n_irreducible_kpoints": n_irr,
            "occupation": occupation,
            "per_model_shift_eV": shifts,
            "avg_hamiltonian_shift_eV": avg_ham_shift,
            "kpoints": result_per_k,
        }

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
        if structure_pattern:
            structures = [p.name for p in (model_dirs[0] / structure_pattern).parent.glob(structure_pattern)]
        else:
            structures = [p.name for p in (model_dirs[0] / "*").parent.glob("*") if (model_dirs[0] / p).is_dir()]
        if exclude_structures:
            excluded = set(exclude_structures)
            structures = [s for s in structures if s not in excluded]

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

    def compare_averaged_to_dft(self, averaged_model_root: Path, dft_root: Path, structure_pattern: str | None = None):
        """Compare averaged Hamiltonian (stored per-structure under `averaged_model_root`) to DFT reference.

        Expects `averaged_model_root/<structure_name>/hamiltonian.h5` and supporting files
        (info.json, overlap.h5) to be present so `SparseHamiltonianObj` can read the averaged result.

        Returns a dict keyed by structure with MAE and per-k error lists similar to `compute`.
        """
        averaged_model_root = Path(averaged_model_root)
        dft_root = Path(dft_root)

        if structure_pattern:
            structures = [p.name for p in (averaged_model_root / structure_pattern).parent.glob(structure_pattern)]
        else:
            structures = [p.name for p in averaged_model_root.glob("*") if (averaged_model_root / p.name).is_dir()]

        results = {}
        for structure_name in structures:
            avg_obj = SparseHamiltonianObj(averaged_model_root / structure_name)
            dft_obj = SparseHamiltonianObj(dft_root / structure_name)

            ks, weights, anchor_k_idx = self.build_irreducible_kpoints(dft_obj, self.grid_mesh, self.symprec)

            avg_raw = avg_obj.diag(ks, bands_only=True)
            dft_raw = dft_obj.diag(ks, bands_only=True)

            avg_aligned, _ = self.align_to_midgap(avg_raw, avg_obj, anchor_k_idx)
            dft_aligned, _ = self.align_to_midgap(dft_raw, dft_obj, anchor_k_idx)

            window_mask = self.band_window_mask(avg_aligned, self.window_ev)

            abs_err = np.abs(avg_aligned - dft_aligned)

            per_k = {}
            mae_values = []
            for i_k in range(len(ks)):
                mask_k = window_mask[:, i_k]
                vals = abs_err[mask_k, i_k].tolist()
                per_k[f"k{i_k}"] = {
                    "k_frac": ks[i_k].tolist(),
                    "weight": int(weights[i_k]),
                    "abs_err_eV": vals,
                    "n_bands_in_window": int(mask_k.sum()),
                }
                if vals:
                    mae_values.append(float(np.mean(vals)))

            overall_mae = float(np.mean(mae_values)) if mae_values else 0.0

            results[structure_name] = {
                "overall_mae_eV": overall_mae,
                "occupation": dft_obj.occupation,
                "per_k": per_k,
            }

        return results


__all__ = ["BandUncertaintyCalculator"]
