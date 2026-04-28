from pathlib import Path
from typing import Any

from pymatgen.core.structure import FileFormats, Structure

__all__ = ["generate_perturbed_population"]


def _normalize_supercell_size(supercell_size: list[int] | tuple[int, int, int]) -> tuple[int, int, int]:
    if len(supercell_size) != 3:
        raise ValueError(
            "supercell_size must contain exactly 3 integers, "
            f"got {len(supercell_size)} entries: {supercell_size}"
        )
    normalized = (int(supercell_size[0]), int(supercell_size[1]), int(supercell_size[2]))
    if any(value <= 0 for value in normalized):
        raise ValueError(f"supercell_size values must be positive integers, got: {normalized}")
    return normalized


def generate_perturbed_population(
    structure: Structure,
    input_structures_path: str | Path,
    num_structures: int,
    distance: float,
    min_distance: float | None = None,
    supercell_size: list[int] | tuple[int, int, int] | None = None,
    file_format: FileFormats = "poscar",
) -> dict[str, Any]:
    """
    Generate ``structure_0..structure_{N-1}`` folders from a base structure.

    ``structure_0`` is the unperturbed reference; all subsequent structures are
    generated with ``Structure.perturb`` using the requested parameters.
    """
    if num_structures < 1:
        raise ValueError(f"num_structures must be >= 1, got: {num_structures}")
    if distance <= 0:
        raise ValueError(f"distance must be > 0, got: {distance}")
    if min_distance is not None and min_distance < 0:
        raise ValueError(f"min_distance must be >= 0 when provided, got: {min_distance}")

    normalized_format = str(file_format).lower()

    def _output_filename() -> str:
        if normalized_format == "poscar":
            return "POSCAR"
        if normalized_format == "aims":
            return "geometry.in"
        return f"structure.{normalized_format}"

    base_structure = structure.copy()
    if supercell_size is not None:
        base_structure.make_supercell(_normalize_supercell_size(supercell_size))

    input_structure_path = Path(input_structures_path)
    input_structure_path.mkdir(parents=True, exist_ok=True)

    output_filename = _output_filename()
    written_dirs: list[str] = []

    structure_0_path = input_structure_path / "structure_0"
    structure_0_path.mkdir(parents=True, exist_ok=True)
    base_structure.to(filename=structure_0_path / output_filename)
    written_dirs.append(str(structure_0_path.resolve()))

    for i in range(1, num_structures):
        perturbed_structure = base_structure.copy()
        perturbed_structure.perturb(distance=distance, min_distance=min_distance)
        perturbed_structure_path = input_structure_path / f"structure_{i}"
        perturbed_structure_path.mkdir(parents=True, exist_ok=True)
        perturbed_structure.to(filename=perturbed_structure_path / output_filename)
        written_dirs.append(str(perturbed_structure_path.resolve()))

    return {
        "input_structures_path": str(input_structure_path.resolve()),
        "num_structures": num_structures,
        "distance": distance,
        "min_distance": min_distance,
        "supercell_size": list(supercell_size) if supercell_size is not None else None,
        "file_format": normalized_format,
        "written_structure_dirs": written_dirs,
    }
