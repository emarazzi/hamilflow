from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from pymatgen.core import Structure

from hamilflow.band_structures import SparseBandDataGenerator
from hamilflow.band_structures.band_calculation import (
    get_band_conf_from_struc,
    get_hamiltonian,
    plot_band,
)


@dataclass(frozen=True)
class ExamplePaths:
    """Input and output paths used by the example script."""

    structure_path: Path
    workdir: Path
    output_path: Path


def parse_args() -> ExamplePaths:
    parser = argparse.ArgumentParser(
        description="Generate and plot a band structure using hamilflow.band_structures.band_calculation."
    )
    parser.add_argument(
        "workdir",
        type=Path,
        help="DeepH-format directory containing the Hamiltonian inputs (e.g. /path/to/your/structure/folder).",
    )
    parser.add_argument(
        "--structure",
        type=Path,
        default=None,
        help="Path to a structure file readable by pymatgen. Default: <workdir>/POSCAR.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path written by plot_band. Default: <workdir>/band_structure.png.",
    )
    args = parser.parse_args()
    return ExamplePaths(
        structure_path=args.structure or args.workdir / "POSCAR",
        workdir=args.workdir,
        output_path=args.output or args.workdir / "band_structure.png",
    )


def build_band_structure(paths: ExamplePaths):
    if not paths.structure_path.exists():
        raise FileNotFoundError(f"Structure file not found: {paths.structure_path}")
    if not paths.workdir.exists():
        raise FileNotFoundError(f"Working directory not found: {paths.workdir}")

    structure = Structure.from_file(paths.structure_path)
    band_conf = get_band_conf_from_struc(structure)
    hamiltonian = get_hamiltonian(paths.workdir)

    band_data = SparseBandDataGenerator(hamiltonian, band_conf)
    band_data.calc_band_data()
    return band_data


def main() -> None:
    paths = parse_args()
    # hamilflow's messages and warnings on stdout and in <workdir>/band_structure.log
    # (created at the first message)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(paths.workdir / "band_structure.log", delay=True),
        ],
    )
    logging.captureWarnings(True)
    band_data = build_band_structure(paths)

    paths.output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_band(band_data, show_fig=False, save_path=paths.output_path)

    print(f"Band structure saved to {paths.output_path.resolve()}")


if __name__ == "__main__":
    main()
