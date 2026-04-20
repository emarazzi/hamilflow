from __future__ import annotations

import argparse
import importlib
import logging
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from pymatgen.core import Structure

from hamilflow.band_structure import get_band_conf_from_struc, get_hamiltonian

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BandStructurePaths:
    """Filesystem locations for one band-structure calculation."""

    structure_path: Path
    workdir: Path
    output_path: Path


@dataclass(frozen=True)
class BatchOptions:
    """Configuration for a batch run over multiple independent directories."""

    parent_dir: Path
    pattern: str
    structure_filename: str
    output_filename: str
    workers: int


def load_band_data_generator() -> type:
    """Import BandDataGenerator lazily to keep the example import-light."""
    module = importlib.import_module("deepx_dock.compute.eigen.band")
    return module.BandDataGenerator


def build_band_structure(paths: BandStructurePaths):
    """Build the band structure object for one input directory."""
    if not paths.structure_path.exists():
        raise FileNotFoundError(f"Structure file not found: {paths.structure_path}")
    if not paths.workdir.exists():
        raise FileNotFoundError(f"Working directory not found: {paths.workdir}")

    structure = Structure.from_file(paths.structure_path)
    band_conf = get_band_conf_from_struc(structure)
    hamiltonian = get_hamiltonian(paths.workdir)

    band_data_generator = load_band_data_generator()
    band_data = band_data_generator(hamiltonian, band_conf)
    band_data.calc_band_data()
    return band_data


def parse_args() -> BatchOptions:
    parser = argparse.ArgumentParser(
        description=(
            "Run band-structure calculations for a parent directory containing "
            "independent structure_* subdirectories."
        )
    )
    parser.add_argument(
        "parent_dir",
        type=Path,
        help="Parent directory containing subdirectories such as structure_0, structure_1, ...",
    )
    parser.add_argument(
        "--pattern",
        default="structure_*",
        help="Glob pattern used to select child directories under parent_dir.",
    )
    parser.add_argument(
        "--structure-filename",
        default="POSCAR",
        help="Structure filename expected inside each child directory.",
    )
    parser.add_argument(
        "--output-filename",
        default="band_data.h5",
        help="Output filename written inside each child directory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes to use. Keep this at 1 for pure serial execution.",
    )
    args = parser.parse_args()
    return BatchOptions(
        parent_dir=args.parent_dir,
        pattern=args.pattern,
        structure_filename=args.structure_filename,
        output_filename=args.output_filename,
        workers=args.workers,
    )


def iter_case_directories(options: BatchOptions) -> list[Path]:
    if not options.parent_dir.exists():
        raise FileNotFoundError(f"Parent directory not found: {options.parent_dir}")

    child_dirs = [path for path in sorted(options.parent_dir.glob(options.pattern)) if path.is_dir()]
    if not child_dirs:
        raise FileNotFoundError(
            f"No subdirectories matched pattern '{options.pattern}' in {options.parent_dir}"
        )
    return child_dirs


def run_single_case(case_dir: Path, options: BatchOptions) -> tuple[Path, bool, str | None]:
    paths = BandStructurePaths(
        structure_path=case_dir / options.structure_filename,
        workdir=case_dir,
        output_path=case_dir / options.output_filename,
    )

    try:
        band_data = build_band_structure(paths)
        paths.output_path.parent.mkdir(parents=True, exist_ok=True)
        band_data.dump_band_data(str(paths.output_path))
        return case_dir, True, None
    except Exception as exc:  # pragma: no cover - surfaced to the user in summary
        return case_dir, False, f"{type(exc).__name__}: {exc}"


def run_batch(options: BatchOptions) -> list[tuple[Path, bool, str | None]]:
    case_dirs = iter_case_directories(options)

    if options.workers <= 1:
        return [run_single_case(case_dir, options) for case_dir in case_dirs]

    with ProcessPoolExecutor(max_workers=options.workers) as executor:
        worker = partial(run_single_case, options=options)
        return list(executor.map(worker, case_dirs))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    options = parse_args()
    results = run_batch(options)

    succeeded = [case_dir for case_dir, ok, _ in results if ok]
    failed = [(case_dir, error) for case_dir, ok, error in results if not ok]

    for case_dir in succeeded:
        LOGGER.info("Completed %s", case_dir)
    for case_dir, error in failed:
        LOGGER.error("Failed %s: %s", case_dir, error)

    LOGGER.info("Finished %d case(s): %d succeeded, %d failed.", len(results), len(succeeded), len(failed))

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
