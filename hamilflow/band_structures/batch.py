"""
Band structures of every structure in a folder, from DeepH-format inputs, with sparse Hamiltonians
and GPU diagonalization.

Each subfolder of the parent folder is one structure in DeepH format (``POSCAR``, ``info.json``,
``overlap.h5``, ``hamiltonian.h5``). For each, H(R)/S(R) are kept as sparse blocks
(`SparseHamiltonianObj`), H(k)/S(k) are built as sparse matrices along a high-symmetry path, and
diagonalized by the solver chosen with ``solver``:

* ``"gpu"``: full diagonalization on the GPU (CuPy/cuSOLVER). Only the nonzeros of H(k)/S(k) are
  sent to the device, where they are densified; the host never holds a dense (Nb, Nb) matrix.
* ``"full"``: the same, on the GPU if one is available (see `dense_backend`), else on the CPU.
* ``"sparse"``: ``num_band`` eigenvalues above E_F + ``emin`` with shift-invert Lanczos (scipy
  ``eigsh``) on the CPU, never densifying anything.
* ``"auto"`` (default): ``"full"`` for matrices smaller than `GPU_MIN_DIM` or when a GPU is
  available, ``"sparse"`` otherwise.

The path is SeeK-path's for each structure unless one is given (see `kpath.resolve_kpath`). The
bands (shifted so that E_F = 0) go to ``<subfolder>/<output_filename>`` in ``BandDataGenerator``'s
HDF5 format, and optionally a plot to ``<subfolder>/band.png``.

Command line: ``hamilflow-bands PARENT_DIR [--kpath ...]``, see ``hamilflow-bands --help``.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
from deepx_dock.CONSTANT import DEEPX_INFO_FILENAME, DEEPX_OVERLAP_FILENAME, DEEPX_POSCAR_FILENAME

from .kpath import DEFAULT_DENSITY, resolve_kpath, structure_cell

logger = logging.getLogger(__name__)

SOLVERS = ("auto", "gpu", "full", "sparse")
REQUIRED_FILES = (DEEPX_INFO_FILENAME, DEEPX_POSCAR_FILENAME, DEEPX_OVERLAP_FILENAME)


@dataclass(frozen=True)
class BandOptions:
    """Settings shared by every structure of a batch (see the module docstring)."""

    kpath: str | None = None
    density: float = DEFAULT_DENSITY
    in_plane: bool = False
    symprec: float = 1e-5
    solver: str = "auto"
    num_band: int | None = None
    emin: float = -0.5
    maxiter: int = 300
    n_jobs: int = -1
    gpu_threads: int = 2
    hamiltonian_filename: str = "hamiltonian.h5"
    output_filename: str = "band_data.h5"
    plot: bool = False
    plot_emin: float = -5.0
    plot_emax: float = 5.0

    def __post_init__(self):
        if self.solver not in SOLVERS:
            raise ValueError(f"solver must be one of {SOLVERS}, got {self.solver!r}")
        if self.num_band is not None and self.num_band < 1:
            raise ValueError(f"num_band must be positive, got {self.num_band}")


def _choose_solver(solver: str, mat_dim: int) -> str:
    """The concrete solver (``"full"`` or ``"sparse"``) for ``solver`` on an Nb = ``mat_dim`` problem.
    Initializes CUDA in this process when it asks for the GPU."""
    from hamilflow.sparse_hamiltonian.dense_backend import GPU_MIN_DIM, gpu_backend

    if solver == "gpu":
        if gpu_backend() is None:
            raise RuntimeError("solver='gpu' but no GPU backend is available (CuPy with a visible CUDA device); "
                               "use solver='full' or 'sparse' to run on the CPU")
        if mat_dim < GPU_MIN_DIM:
            logger.info(f"Matrix dimension {mat_dim} < {GPU_MIN_DIM}: diagonalized on the CPU")
        return "full"
    if solver == "auto":
        return "full" if mat_dim < GPU_MIN_DIM or gpu_backend() is not None else "sparse"
    return solver


def _keep_band_window(bd_gen, num_band: int, emin: float) -> None:
    """Keep ``num_band`` bands of a full diagonalization (already shifted to E_F = 0), starting from
    the lowest band reaching above ``emin``: the same band indices at every k-point."""
    data = bd_gen.band_data
    above = np.nonzero(data.max(axis=1) >= emin)[0]
    start = int(above[0]) if len(above) else data.shape[0]
    start = max(0, min(start, data.shape[0] - num_band))
    bd_gen.band_data = data[start:start + num_band]
    bd_gen.band_quantity = bd_gen.band_data.shape[0]


def compute_band_structure(case_dir: str | Path, options: BandOptions | None = None):
    """
    Band structure of one DeepH-format folder: returns the `SparseBandDataGenerator`, with
    ``band_data`` shifted to E_F = 0, after writing it to ``case_dir / options.output_filename``.
    """
    from hamilflow.band_structures.band_data_generator import SparseBandDataGenerator
    from hamilflow.sparse_hamiltonian import SparseHamiltonianObj
    from hamilflow.sparse_hamiltonian.dense_backend import GPU_MIN_DIM, gpu_in_use

    options = BandOptions() if options is None else options
    case_dir = Path(case_dir)
    t0 = time.perf_counter()
    h_path = case_dir / options.hamiltonian_filename
    obj_H = SparseHamiltonianObj(case_dir, H_file_path=h_path)
    cell = structure_cell(obj_H.lattice, obj_H.frac_coords, obj_H.elements)
    k_list_spell = resolve_kpath(cell, options.kpath, density=options.density, in_plane=options.in_plane,
                                 symprec=options.symprec)

    solver = _choose_solver(options.solver, obj_H._mat_dim)
    num_band = options.num_band
    if solver == "sparse" and num_band is None:
        num_band = min(50, obj_H._mat_dim - 2)
    band_conf = {"k_list_spell": k_list_spell, "num_band": num_band, "lowest_band_energy": options.emin,
                 "maxiter": options.maxiter}
    bd_gen = SparseBandDataGenerator(obj_H, band_conf)
    logger.info(f"{case_dir}: Nb = {obj_H._mat_dim}, {bd_gen.kpoints_quantity} k-points, solver '{solver}'"
                + (f" ({num_band} bands)" if num_band else " (all bands)"))

    if solver == "sparse":
        bd_gen.calc_band_data(n_jobs=options.n_jobs, parallel_k=True, sparse_calc=True)
    else:
        n_jobs = options.n_jobs
        if gpu_in_use() and obj_H._mat_dim >= GPU_MIN_DIM:
            # each thread holds ~5 dense Nb x Nb matrices on the device: a few threads are enough to
            # overlap the host-side sparse assembly with the GPU work
            n_jobs = options.gpu_threads if n_jobs < 0 else min(n_jobs, options.gpu_threads)
        bd_gen.calc_band_data(n_jobs=n_jobs, parallel_k=True, sparse_assembly=True)
        if num_band is not None and num_band < bd_gen.band_quantity:
            _keep_band_window(bd_gen, num_band, options.emin)

    output = case_dir / options.output_filename
    bd_gen.dump_band_data(str(output))
    logger.info(f"{case_dir}: wrote {output} in {time.perf_counter() - t0:.1f} s "
                f"(gap {bd_gen.band_gap:.3f} eV, metal={bd_gen._is_metal})")
    if options.plot:
        _plot(output, options.plot_emin, options.plot_emax)
    return bd_gen


def _plot(band_file: Path, emin: float, emax: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from deepx_dock.compute.eigen.band import BandPlotter

    BandPlotter(band_file).plot(Emin=emin, Emax=emax, plot_format="png")
    plt.close("all")


def find_structure_dirs(parent_dir: str | Path, pattern: str = "*",
                        hamiltonian_filename: str = "hamiltonian.h5") -> list[Path]:
    """The subfolders of ``parent_dir`` matching ``pattern`` that hold DeepH inputs, sorted by name
    (natural order: structure_2 before structure_10). Subfolders missing a file are skipped with a
    warning."""
    parent_dir = Path(parent_dir)
    if not parent_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {parent_dir}")

    def natural_key(p: Path):
        import re

        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]

    dirs = []
    for d in sorted((p for p in parent_dir.glob(pattern) if p.is_dir()), key=natural_key):
        missing = [f for f in (*REQUIRED_FILES, hamiltonian_filename) if not (d / f).is_file()]
        if missing:
            logger.warning(f"Skipping {d}: missing {', '.join(missing)}")
        else:
            dirs.append(d)
    if not dirs:
        raise FileNotFoundError(f"No subfolder of {parent_dir} matching '{pattern}' holds DeepH inputs")
    return dirs


def _run_one(case_dir: Path, options: BandOptions) -> tuple[Path, str | None]:
    from hamilflow.sparse_hamiltonian.dense_backend import release_gpu_memory

    try:
        compute_band_structure(case_dir, options)
        return case_dir, None
    except Exception as exc:  # reported in the summary, the batch goes on
        logger.exception(f"{case_dir}: failed")
        return case_dir, f"{type(exc).__name__}: {exc}"
    finally:
        release_gpu_memory()


def run_folder(
    parent_dir: str | Path,
    options: BandOptions | None = None,
    pattern: str = "*",
    workers: int = 1,
    skip_existing: bool = False,
) -> dict[Path, str | None]:
    """
    Band structures of every structure subfolder of ``parent_dir`` (see `find_structure_dirs`).
    Returns ``{subfolder: None if it succeeded, else the error}``; a failure does not stop the others.

    ``workers > 1`` runs that many structures at once in separate processes, which can share one
    GPU; each then gets ``n_jobs // workers`` threads (with ``options.n_jobs = -1``). This process
    does not initialize CUDA, so that the forked workers can.
    """
    from hamilflow.cpus import available_cpu_count
    from hamilflow.sparse_hamiltonian.dense_backend import gpu_use_reported, workers_gpu_memory

    options = BandOptions() if options is None else options
    dirs = find_structure_dirs(parent_dir, pattern, options.hamiltonian_filename)
    if skip_existing:
        done = [d for d in dirs if (d / options.output_filename).is_file()]
        if done:
            logger.info(f"Skipping {len(done)} structure(s) with an existing {options.output_filename}")
        dirs = [d for d in dirs if d not in done]
    logger.info(f"{len(dirs)} structure(s) in {parent_dir}: " + ", ".join(d.name for d in dirs[:10])
                + (", ..." if len(dirs) > 10 else ""))
    if not dirs:
        return {}

    workers = max(1, min(workers, len(dirs)))
    if workers == 1:
        return dict(_run_one(d, options) for d in dirs)

    if options.n_jobs < 0:
        options = BandOptions(**{**asdict(options), "n_jobs": max(1, available_cpu_count() // workers)})
    reported = contextlib.nullcontext()
    if workers_gpu_memory() is not None:
        logger.info(f"{workers} workers sharing the GPU (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
        reported = gpu_use_reported()
    with reported, ProcessPoolExecutor(max_workers=workers) as ex:
        return dict(ex.map(partial(_run_one, options=options), dirs))


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="hamilflow-bands",
        description="Band structures along a high-symmetry path for every DeepH-format subfolder of a folder, "
                    "with sparse Hamiltonians and GPU diagonalization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("parent_dir", type=Path, help="Folder whose subfolders are the structures")
    p.add_argument("--pattern", default="*", help="Glob selecting the subfolders")
    p.add_argument("--kpath", default=None,
                   help="k-path: a label path with SeeK-path's coordinates (e.g. 'GAMMA-M-K-GAMMA|A-L'), "
                        "a K_PATH file, or a JSON file {'points': {...}, 'path': ...}. Default: SeeK-path's path "
                        "of each structure")
    p.add_argument("--density", type=float, default=DEFAULT_DENSITY,
                   help="k-points per 1/Angstrom of path (reciprocal lattice with the 2 pi factor)")
    p.add_argument("--in-plane", action="store_true",
                   help="Keep only the k_z = 0 segments of the SeeK-path path (slabs, 2D materials)")
    p.add_argument("--symprec", type=float, default=1e-5, help="Symmetry tolerance of SeeK-path (Angstrom)")
    p.add_argument("--solver", choices=SOLVERS, default="auto",
                   help="gpu: full diagonalization on the GPU; full: on the GPU if available, else the CPU; "
                        "sparse: shift-invert Lanczos on the CPU; auto: gpu/full if possible, else sparse")
    p.add_argument("--num-band", type=int, default=None,
                   help="Bands kept, from the first reaching above E_F + emin (default: all for full/gpu, 50 for "
                        "sparse)")
    p.add_argument("--emin", type=float, default=-0.5, help="Lowest energy of the kept bands, relative to E_F (eV)")
    p.add_argument("--maxiter", type=int, default=300, help="Maximum Lanczos iterations (sparse solver)")
    p.add_argument("--n-jobs", type=int, default=-1, help="Threads per structure (-1: all cores / workers)")
    p.add_argument("--gpu-threads", type=int, default=2,
                   help="Threads per structure sharing the GPU in the full/gpu solver (each holds ~5 Nb x Nb "
                        "matrices on the device)")
    p.add_argument("--workers", type=int, default=1, help="Structures computed at once (processes)")
    p.add_argument("--hamiltonian-filename", default="hamiltonian.h5", help="Hamiltonian file in each subfolder")
    p.add_argument("--output-filename", default="band_data.h5", help="Output HDF5 file in each subfolder")
    p.add_argument("--skip-existing", action="store_true", help="Skip subfolders that already have the output")
    p.add_argument("--plot", action="store_true", help="Also write band.png in each subfolder")
    p.add_argument("--plot-emin", type=float, default=-5.0, help="Plot energy range minimum (eV)")
    p.add_argument("--plot-emax", type=float, default=5.0, help="Plot energy range maximum (eV)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(args.parent_dir / "band_structures.log", delay=True)],
    )
    logging.captureWarnings(True)
    options = BandOptions(
        kpath=args.kpath, density=args.density, in_plane=args.in_plane, symprec=args.symprec, solver=args.solver,
        num_band=args.num_band, emin=args.emin, maxiter=args.maxiter, n_jobs=args.n_jobs,
        gpu_threads=args.gpu_threads,
        hamiltonian_filename=args.hamiltonian_filename, output_filename=args.output_filename, plot=args.plot,
        plot_emin=args.plot_emin, plot_emax=args.plot_emax,
    )
    results = run_folder(args.parent_dir, options, pattern=args.pattern, workers=args.workers,
                         skip_existing=args.skip_existing)
    failed = {d: err for d, err in results.items() if err is not None}
    for d, err in failed.items():
        logger.error(f"Failed {d}: {err}")
    logger.info(f"Finished {len(results)} structure(s): {len(results) - len(failed)} succeeded, {len(failed)} failed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
