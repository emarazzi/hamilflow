import concurrent.futures
import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..cpus import available_cpu_count
from .alignment import (
    SparseStream,
    fit_rigid_shifts,
    pair_distances,
    write_aligned_hamiltonian,
)
from .band_postprocess import BandUncertaintyPostProcessor
from .hamiltonian_io import average_predicted_hamiltonians


@dataclass
class MatrixUncertaintyCalculator:
    """Compute Hamiltonian matrix-element uncertainty across an ensemble of models.

    For each structure: obtain the average Hamiltonian (read it from ``average_hamiltonian_dir``
    if a file is already there, otherwise compute it with
    :func:`hamiltonian_io.average_predicted_hamiltonians`), align every member to it, and stream
    the real-space sparse ``entries`` of the members and the average one batch of atom-pair
    blocks at a time, so peak memory stays bounded regardless of structure size.

    - ``align``: remove from each member the rigid shift ``c_m S`` that best explains its
      difference from the average (see :mod:`alignment`). Predicted Hamiltonians carry an absolute
      energy offset that differs from model to model; without alignment it dominates the
      matrix-element differences. The shifts and their residuals are reported.
    - ``statistic``: ``"rms"`` (default) -- per matrix element, the RMS deviation of the members
      from the average, ``sigma_e = sqrt(mean_m |h_m - c_m s - h_avg|^2)`` (the same definition as
      the band uncertainty); ``"std_abs"`` -- the earlier statistic, the std over members of
      ``|h_m - h_avg|`` (with ``ddof``), kept for comparison.
    - Structure scores: ``uncertainty_rms`` (RMS of ``sigma_e`` over all elements),
      ``uncertainty_mean`` (mean of ``sigma_e``), ``uncertainty_max`` (max of ``sigma_e``).
    - ``weighting``: optional distance weighting, ``None`` (default), ``"fermi"`` or ``"exp"``.
      Each atom pair ``p`` (R, i, j) gets the RMS of ``sigma_e`` over its block, ``sigma_p``, and a
      weight ``w_p`` from its distance ``r_p = |R + x_j - x_i|`` (0 on-site): ``"fermi"`` is the
      soft window of the band post-processing, ``w = F((r - r_cut)/tau) / F(-r_cut/tau)`` (1
      on-site, flat up to ``r_cut``, dropping over a few ``tau``); ``"exp"`` is
      ``exp(-r / decay_length)``. Adds ``uncertainty_weighted_mean`` (``sum w_p sigma_p / sum w_p``)
      and ``uncertainty_weighted_max`` (``max w_p sigma_p``). Distances in Angstrom, from POSCAR.
    - ``aligned_hamiltonian_dir``: if given (and ``align``), the aligned member Hamiltonians are
      also written there, ``model_<k>/<structure>/``. Off by default.

    Example usage:
      calc = MatrixUncertaintyCalculator()
      output = calc.compute(model_dirs)
    """

    hamiltonian_name: str = "hamiltonian.h5"
    overlap_name: str = "overlap.h5"
    chunk_size: int = 2_000_000
    ddof: int = 1
    align: bool = True
    statistic: str = "rms"
    weighting: str | None = None
    r_cut: float = 5.0
    tau: float = 0.5
    decay_length: float = 2.0
    aligned_hamiltonian_dir: Path | None = None

    def __post_init__(self):
        if self.statistic not in ("rms", "std_abs"):
            raise ValueError(f"statistic must be 'rms' or 'std_abs', got {self.statistic!r}")
        if self.weighting == "none":
            self.weighting = None
        if self.weighting not in (None, "fermi", "exp"):
            raise ValueError(f"weighting must be None, 'fermi' or 'exp', got {self.weighting!r}")

    def settings(self) -> dict:
        """Settings that change the results, stored with every structure's output."""
        out = {"align": self.align, "statistic": self.statistic, "weighting": self.weighting}
        if self.weighting == "fermi":
            out.update(r_cut=self.r_cut, tau=self.tau)
        elif self.weighting == "exp":
            out.update(decay_length=self.decay_length)
        return out

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

    def _pair_weights(self, structure_dir: Path, atom_pairs: np.ndarray) -> np.ndarray | None:
        if self.weighting is None:
            return None
        r = pair_distances(structure_dir, atom_pairs)
        if self.weighting == "fermi":
            return BandUncertaintyPostProcessor.soft_window(r, self.r_cut, self.tau)
        return np.exp(-r / self.decay_length)

    def _shifts(self, h_paths, ref_path, s_path, labels, structure_name, aligned_to, n_threads=1):
        """Rigid shifts of `h_paths` relative to `ref_path` (zero when not aligning), writing the
        aligned files if requested."""
        if not self.align:
            return None
        rigid = fit_rigid_shifts(h_paths, ref_path, s_path, self.chunk_size, n_threads)
        if self.aligned_hamiltonian_dir is not None:
            for h_path, label, r in zip(h_paths, labels, rigid):
                write_aligned_hamiltonian(
                    h_path, s_path, r["c_eV"],
                    Path(self.aligned_hamiltonian_dir) / label / structure_name / self.hamiltonian_name,
                    aligned_to=aligned_to, batch_entries=self.chunk_size,
                )
        return rigid

    def _deviation_stats(self, stream: SparseStream, n_dev: int, ref_index: int, shifts: list[float],
                         weights: np.ndarray | None) -> dict:
        """Stream `stream`, whose paths are the `n_dev` Hamiltonians to compare followed by the
        reference at `ref_index`, and reduce the per-element deviations `h_k - c_k s - h_ref`."""
        sum_sq = sum_val = 0.0
        max_val = 0.0
        n_total = 0
        pair_sq = np.zeros(len(stream.atom_pairs)) if weights is not None else None
        for p0, p1, hs, s in stream:
            ref = hs[ref_index]
            devs = np.stack([hs[k] - shifts[k] * s - ref for k in range(n_dev)], axis=0)
            if self.statistic == "rms":
                sigma = np.sqrt(np.mean(np.abs(devs) ** 2, axis=0))
            else:
                sigma = np.std(np.abs(devs), axis=0, ddof=self.ddof if n_dev > 1 else 0)
            sum_sq += float(np.sum(sigma ** 2))
            sum_val += float(np.sum(sigma))
            max_val = max(max_val, float(sigma.max(initial=0.0)))
            n_total += sigma.size
            if pair_sq is not None:
                offsets = stream.bounds[p0:p1] - stream.bounds[p0]
                pair_sq[p0:p1] = np.add.reduceat(sigma ** 2, offsets) if sigma.size else 0.0
        out = {
            "uncertainty_rms": float(np.sqrt(sum_sq / n_total)) if n_total else 0.0,
            "uncertainty_mean": sum_val / n_total if n_total else 0.0,
            "uncertainty_max": max_val,
            "n_matrix_elements": n_total,
        }
        if pair_sq is not None:
            sizes = np.diff(stream.bounds)
            sigma_p = np.sqrt(pair_sq / np.maximum(sizes, 1))
            out["uncertainty_weighted_mean"] = float(np.sum(weights * sigma_p) / np.sum(weights))
            out["uncertainty_weighted_max"] = float(np.max(weights * sigma_p))
        return out

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
        structure_dir = Path(model_dirs[0]) / structure_name
        s_path = structure_dir / self.overlap_name

        rigid = self._shifts(model_paths, avg_path, s_path, [f"model_{k + 1}" for k in range(len(model_paths))],
                             structure_name, "ensemble average", n_jobs)
        shifts = [r["c_eV"] for r in rigid] if rigid else [0.0] * len(model_paths)

        stream = SparseStream(model_paths + [avg_path], s_path, self.chunk_size, n_jobs)
        weights = self._pair_weights(structure_dir, stream.atom_pairs)
        stats = self._deviation_stats(stream, len(model_paths), len(model_paths), shifts, weights)
        return structure_name, {
            **self.settings(),
            **stats,
            **({"per_model_rigid_shift_eV": shifts,
                "per_model_rigid_shift_residual": [r["residual"] for r in rigid]} if rigid else {}),
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

        With ``align``, the average is first aligned to DFT by the rigid shift ``c S`` that best
        explains their difference (reported as ``avg_vs_dft_rigid_shift_eV`` with its residual),
        and written to ``aligned_hamiltonian_dir/average_to_dft/<structure>/`` if that is set.
        Reports the mean, max and RMS absolute difference over matrix elements, and with
        ``weighting`` the distance-weighted mean and max of the per-pair RMS difference.
        Entries are streamed, so peak memory stays bounded regardless of structure size.
        """
        model_dirs = [Path(p) for p in model_dirs]
        dft_root = Path(dft_root)
        structures = self._discover_structures(model_dirs, structure_pattern, exclude_structures)

        results = {}
        with tempfile.TemporaryDirectory() as tmp:
            for structure_name in structures:
                model_paths = [Path(d) / structure_name / self.hamiltonian_name for d in model_dirs]
                avg_path = self._resolve_average_path(structure_name, model_paths, average_hamiltonian_dir, tmp)
                dft_dir = dft_root / structure_name
                dft_path = dft_dir / self.hamiltonian_name
                s_path = dft_dir / self.overlap_name

                rigid = self._shifts([avg_path], dft_path, s_path, ["average_to_dft"], structure_name, "DFT")
                c = rigid[0]["c_eV"] if rigid else 0.0

                stream = SparseStream([avg_path, dft_path], s_path, self.chunk_size)
                weights = self._pair_weights(dft_dir, stream.atom_pairs)
                stats = self._deviation_stats(stream, 1, 1, [c], weights)
                sum_abs = max_abs = 0.0
                for _, _, (h_avg, h_dft), s in SparseStream([avg_path, dft_path], s_path, self.chunk_size):
                    a = np.abs(h_avg - c * s - h_dft)
                    sum_abs += float(a.sum())
                    max_abs = max(max_abs, float(a.max(initial=0.0)))
                n_total = stats["n_matrix_elements"]

                results[structure_name] = {
                    **self.settings(),
                    "mean_abs_diff": sum_abs / n_total if n_total else 0.0,
                    "max_abs_diff": max_abs,
                    "rms_diff": stats["uncertainty_rms"],
                    **({"weighted_mean_rms_diff": stats["uncertainty_weighted_mean"],
                        "weighted_max_rms_diff": stats["uncertainty_weighted_max"]} if weights is not None else {}),
                    **({"avg_vs_dft_rigid_shift_eV": c,
                        "avg_vs_dft_rigid_shift_residual": rigid[0]["residual"]} if rigid else {}),
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
            max_workers = min(len(structures), available_cpu_count())

        # Split the machine's cores evenly across worker processes so each
        # process's model-reads within a structure can still use more than
        # one thread when there are fewer workers than cores.
        read_threads_per_worker = max(1, available_cpu_count() // max_workers)

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
