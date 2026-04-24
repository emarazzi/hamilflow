from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from pathlib import Path

from pymatgen.core import Structure

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


def load_band_data_generator() -> type:
    """Import BandDataGenerator lazily to keep editor diagnostics clean."""
    module = importlib.import_module("deepx_dock.compute.eigen.band")
    return module.BandDataGenerator


def parse_args() -> ExamplePaths:
    parser = argparse.ArgumentParser(
        description="Generate and plot a band structure using hamilflow.band_structures.band_calculation."
    )
    parser.add_argument(
        "--structure",
        type=Path,
        default=Path("./Silicon/POSCAR"),
        help="Path to a structure file readable by pymatgen.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("./Silicon"),
        help="Working directory containing the Hamiltonian inputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./Silicon/band_structure.png"),
        help="Output image path written by plot_band.",
    )
    args = parser.parse_args()
    return ExamplePaths(
        structure_path=args.structure,
        workdir=args.workdir,
        output_path=args.output,
    )


def build_band_structure(paths: ExamplePaths):
    if not paths.structure_path.exists():
        raise FileNotFoundError(f"Structure file not found: {paths.structure_path}")
    if not paths.workdir.exists():
        raise FileNotFoundError(f"Working directory not found: {paths.workdir}")

    structure = Structure.from_file(paths.structure_path)
    band_conf = get_band_conf_from_struc(structure)
    hamiltonian = get_hamiltonian(paths.workdir)

    BandDataGenerator = load_band_data_generator()
    band_data = BandDataGenerator(hamiltonian, band_conf)
    band_data.calc_band_data()
    return band_data


def main() -> None:
    paths = parse_args()
    band_data = build_band_structure(paths)

    paths.output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_band(band_data, show_fig=False, save_path=paths.output_path)

    print(f"Band structure saved to {paths.output_path.resolve()}")


if __name__ == "__main__":
    main()
