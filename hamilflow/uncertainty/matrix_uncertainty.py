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
