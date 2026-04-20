from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from pathlib import Path

from pymatgen.core import Structure

from hamilflow.band_structure import get_band_conf_from_struc, get_hamiltonian


@dataclass(frozen=True)
class TaskPaths:
    """Filesystem locations for one band-structure task."""

    case_dir: Path
    structure_filename: str
    output_filename: str


def load_band_data_generator() -> type:
    """Import BandDataGenerator lazily."""
    module = importlib.import_module("deepx_dock.compute.eigen.band")
    return module.BandDataGenerator


def build_band_data(paths: TaskPaths):
    structure_path = paths.case_dir / paths.structure_filename
    workdir = paths.case_dir
    output_path = paths.case_dir / paths.output_filename

    if not paths.case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {paths.case_dir}")
    if not structure_path.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_path}")

    structure = Structure.from_file(structure_path)
    band_conf = get_band_conf_from_struc(structure)
    hamiltonian = get_hamiltonian(workdir)

    band_data_generator = load_band_data_generator()
    band_data = band_data_generator(hamiltonian, band_conf)
    band_data.calc_band_data()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    band_data.dump_band_data(str(output_path))
    return output_path


def parse_args() -> TaskPaths:
    parser = argparse.ArgumentParser(
        description="Run one independent band-structure task for a single directory."
    )
    parser.add_argument(
        "case_dir",
        type=Path,
        help="Directory containing the structure and Hamiltonian inputs for one case.",
    )
    parser.add_argument(
        "--structure-filename",
        default="POSCAR",
        help="Structure filename inside the case directory.",
    )
    parser.add_argument(
        "--output-filename",
        default="band_data.h5",
        help="HDF5 filename written inside the case directory.",
    )
    args = parser.parse_args()
    return TaskPaths(
        case_dir=args.case_dir,
        structure_filename=args.structure_filename,
        output_filename=args.output_filename,
    )


def main() -> None:
    paths = parse_args()
    output_path = build_band_data(paths)
    print(f"Saved band data to {output_path.resolve()}")


if __name__ == "__main__":
    main()
