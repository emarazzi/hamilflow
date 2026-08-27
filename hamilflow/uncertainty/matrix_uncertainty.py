from pathlib import Path
from typing import Iterable
import concurrent.futures
import json
import os
import tempfile

import h5py
import numpy as np
from dataclasses import dataclass

from .hamiltonian_io import average_predicted_hamiltonians


@dataclass
class MatrixUncertaintyCalculator:
    """Compute Hamiltonian matrix-element uncertainty across an ensemble of models.

    For each structure: obtain the average Hamiltonian (read it from
    ``average_hamiltonian_dir`` if a file is already there, otherwise compute
    it on the fly with :func:`hamiltonian_io.average_predicted_hamiltonians`),
    then stream each model's real-space sparse ``entries`` dataset against the
    average's, one chunk at a time, computing the per-matrix-element absolute
    difference and its std across models. The structure's uncertainty is the
    mean of these per-element stds.

    ``entries`` (the on-disk flat array of real-space sparse Hamiltonian
    values, see ``hamiltonian_io``) is never loaded in full for any model or
    the average -- only ``chunk_size`` elements at a time, for every model and
    the average alike, keeping peak memory at ``O(n_models * chunk_size)``
    regardless of structure size.

    Example usage:
      calc = MatrixUncertaintyCalculator()
      output = calc.compute(model_dirs)
    """

    hamiltonian_name: str = "hamiltonian.h5"
    chunk_size: int = 2_000_000
    ddof: int = 1

    def _discover_structures(
        self,
        model_dirs: list[Path],
        structure_pattern: str | None = None,
        exclude_structures: Iterable[str] | None = None,
    ) -> list[str]:
        root = Path(model_dirs[0])
        if structure_pattern:
            structures = [p.name for p in root.glob(structure_pattern)]
        else:
            structures = [p.name for p in root.iterdir() if p.is_dir()]
        if exclude_structures:
            excluded = set(exclude_structures)
            structures = [s for s in structures if s not in excluded]
        return structures

    def _iter_chunk_ranges(self, n_entries: int):
        for start in range(0, n_entries, self.chunk_size):
            yield start, min(start + self.chunk_size, n_entries)

    @staticmethod
    def _read_structure_meta(h5_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
        """Read only the small layout arrays (never ``entries``) needed to
        find the chunking range and to check that a model's sparse layout
        matches the average's."""
        with h5py.File(h5_path, "r") as f:
            atom_pairs = np.array(f["atom_pairs"][:], dtype=np.int64)
            chunk_shapes = np.array(f["chunk_shapes"][:], dtype=np.int64)
            n_entries = f["entries"].shape[0]
        return atom_pairs, chunk_shapes, n_entries

    def _resolve_average_path(
        self,
        structure_name: str,
        model_paths: list[Path],
        average_hamiltonian_dir: Path | None,
        tmp_root: Path,
    ) -> Path:
        root = Path(average_hamiltonian_dir) if average_hamiltonian_dir is not None else Path(tmp_root)
        avg_path = root / structure_name / self.hamiltonian_name
        if not avg_path.exists():
            average_predicted_hamiltonians(model_paths, avg_path)
        return avg_path

    def _compute_structure(
        self,
        structure_name: str,
        model_dirs: list[Path],
        average_hamiltonian_dir: Path | None,
        tmp_root: Path,
        n_jobs: int = 1,
    ) -> tuple[str, dict]:
        model_paths = [Path(d) / structure_name / self.hamiltonian_name for d in model_dirs]
        avg_path = self._resolve_average_path(structure_name, model_paths, average_hamiltonian_dir, tmp_root)

        avg_atom_pairs, avg_chunk_shapes, n_entries = self._read_structure_meta(avg_path)
        for p in model_paths:
            atom_pairs, chunk_shapes, n_model_entries = self._read_structure_meta(p)
            if (
                n_model_entries != n_entries
                or not np.array_equal(atom_pairs, avg_atom_pairs)
                or not np.array_equal(chunk_shapes, avg_chunk_shapes)
            ):
                raise ValueError(
                    f"{structure_name}: sparse layout of {p} does not match the average "
                    f"Hamiltonian ({avg_path}) -- atom_pairs/chunk_shapes/entries length "
                    "must be identical across models"
                )

        sum_std = 0.0
        n_total = 0

        avg_file = h5py.File(avg_path, "r")
        model_files = [h5py.File(p, "r") for p in model_paths]
        try:
            avg_dset = avg_file["entries"]
            model_dsets = [f["entries"] for f in model_files]

            def read_chunk(dset, start, end):
                return np.asarray(dset[start:end])

            for start, end in self._iter_chunk_ranges(n_entries):
                avg_chunk = read_chunk(avg_dset, start, end)
                if n_jobs > 1 and len(model_dsets) > 1:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=min(n_jobs, len(model_dsets))) as tex:
                        model_chunks = list(tex.map(lambda d: read_chunk(d, start, end), model_dsets))
                else:
                    model_chunks = [read_chunk(d, start, end) for d in model_dsets]

                diffs = np.abs(np.stack(model_chunks, axis=0) - avg_chunk[None, :])
                std_chunk = np.std(diffs, axis=0, ddof=self.ddof)
                sum_std += float(std_chunk.sum())
                n_total += std_chunk.size
        finally:
            avg_file.close()
            for f in model_files:
                f.close()

        uncertainty = sum_std / n_total if n_total else 0.0
        return structure_name, {
            "uncertainty": uncertainty,
            "n_matrix_elements": n_total,
            "n_models": len(model_paths),
        }

    def compute(
        self,
        model_dirs: Iterable[Path],
        structure_pattern: str | None = None,
        exclude_structures: Iterable[str] | None = None,
        average_hamiltonian_dir: Path | None = None,
    ) -> dict:
        """Sequential version -- one structure at a time.

        - ``average_hamiltonian_dir``: root containing (or to receive) each
          structure's averaged ``hamiltonian.h5``. If a structure's average is
          already there it is read as-is; otherwise it is computed via
          ``average_predicted_hamiltonians`` and written there. If ``None``,
          averages are computed into a throwaway temporary directory.
        """
        model_dirs = [Path(p) for p in model_dirs]
        structures = self._discover_structures(model_dirs, structure_pattern, exclude_structures)

        output = {}
        with tempfile.TemporaryDirectory() as tmp:
            for structure_name in structures:
                name, res = self._compute_structure(structure_name, model_dirs, average_hamiltonian_dir, tmp)
                output[name] = res
        return output

    def compare_averaged_to_dft(
        self,
        model_dirs: Iterable[Path],
        dft_root: Path,
        structure_pattern: str | None = None,
        exclude_structures: Iterable[str] | None = None,
        average_hamiltonian_dir: Path | None = None,
    ) -> dict:
        """Compare each structure's averaged Hamiltonian (from ``model_dirs``, resolved the
        same way as `compute`) to the DFT reference Hamiltonian under `dft_root`.

        Like `_compute_structure`, entries are streamed in `chunk_size` chunks rather than
        loaded in full, so peak memory stays O(chunk_size) regardless of structure size.

        Returns a dict keyed by structure with summary statistics (mean/max absolute
        difference across matrix elements) similar in spirit to `compute`.
        """
        model_dirs = [Path(p) for p in model_dirs]
        dft_root = Path(dft_root)
        structures = self._discover_structures(model_dirs, structure_pattern, exclude_structures)

        results = {}
        with tempfile.TemporaryDirectory() as tmp:
            for structure_name in structures:
                model_paths = [Path(d) / structure_name / self.hamiltonian_name for d in model_dirs]
                avg_path = self._resolve_average_path(structure_name, model_paths, average_hamiltonian_dir, tmp)
                dft_path = dft_root / structure_name / self.hamiltonian_name

                avg_atom_pairs, avg_chunk_shapes, n_entries = self._read_structure_meta(avg_path)
                dft_atom_pairs, dft_chunk_shapes, n_dft_entries = self._read_structure_meta(dft_path)
                if (
                    n_dft_entries != n_entries
                    or not np.array_equal(dft_atom_pairs, avg_atom_pairs)
                    or not np.array_equal(dft_chunk_shapes, avg_chunk_shapes)
                ):
                    raise ValueError(
                        f"{structure_name}: sparse layout of {dft_path} does not match the "
                        f"averaged Hamiltonian ({avg_path}) -- atom_pairs/chunk_shapes/entries "
                        "length must be identical"
                    )

                sum_abs_diff = 0.0
                max_abs_diff = 0.0
                n_total = 0

                with h5py.File(avg_path, "r") as avg_file, h5py.File(dft_path, "r") as dft_file:
                    avg_dset = avg_file["entries"]
                    dft_dset = dft_file["entries"]
                    for start, end in self._iter_chunk_ranges(n_entries):
                        avg_chunk = np.asarray(avg_dset[start:end])
                        dft_chunk = np.asarray(dft_dset[start:end])
                        abs_diff = np.abs(avg_chunk - dft_chunk)
                        sum_abs_diff += float(abs_diff.sum())
                        max_abs_diff = max(max_abs_diff, float(abs_diff.max()) if abs_diff.size else 0.0)
                        n_total += abs_diff.size

                mean_abs_diff = sum_abs_diff / n_total if n_total else 0.0

                results[structure_name] = {
                    "mean_abs_diff": mean_abs_diff,
                    "max_abs_diff": max_abs_diff,
                    "n_matrix_elements": n_total,
                }

        return results

    def compute_parallel(
        self,
        model_dirs: Iterable[Path],
        structure_pattern: str | None = None,
        max_workers: int | None = None,
        output_path: Path | str | None = None,
        exclude_structures: Iterable[str] | None = None,
        average_hamiltonian_dir: Path | None = None,
    ) -> dict:
        """Parallelized version of `compute` that runs per-structure work in separate processes.

        - `max_workers`: number of worker processes (defaults to number of CPU cores).
        - `output_path`: if given, the accumulated result dict is written to this path
          (atomically) after every structure completes, so a killed/timed-out job still
          leaves the results computed so far on disk.
        - `exclude_structures`: structure names to skip.
        - `average_hamiltonian_dir`: see `compute`.
        """
        model_dirs = [Path(p) for p in model_dirs]
        structures = self._discover_structures(model_dirs, structure_pattern, exclude_structures)

        if max_workers is None:
            max_workers = min(len(structures), os.cpu_count() or 1)

        # Split the machine's cores evenly across worker processes so each
        # process's model-reads within a structure can still use more than
        # one thread when there are fewer workers than cores.
        read_threads_per_worker = max(1, (os.cpu_count() or 1) // max_workers)

        if output_path is not None:
            output_path = Path(output_path)

        output = {}
        with tempfile.TemporaryDirectory() as tmp:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = {
                    ex.submit(
                        self._compute_structure,
                        s,
                        model_dirs,
                        average_hamiltonian_dir,
                        tmp,
                        read_threads_per_worker,
                    ): s
                    for s in structures
                }
                for fut in concurrent.futures.as_completed(futures):
                    name, res = fut.result()
                    output[name] = res
                    if output_path is not None:
                        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
                        with open(tmp_path, "w") as f:
                            json.dump(output, f, indent=4)
                        os.replace(tmp_path, output_path)

        return output


__all__ = ["MatrixUncertaintyCalculator"]
