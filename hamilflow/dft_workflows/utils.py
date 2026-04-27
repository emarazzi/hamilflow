from pathlib import Path
from typing import Any

from pymatgen.core.structure import FileFormats, Structure

__all__ = ["resolve_structure_path"]

_KNOWN_STRUCTURE_SUFFIXES = {
    ".cif",
    ".cssr",
    ".json",
    ".yaml",
    ".yml",
    ".xsf",
    ".mcsqs",
    ".res",
    ".pwmat",
    ".aims",
}
_KNOWN_STRUCTURE_FILENAMES = {
    "poscar",
    "contcar",
    "geometry.in",
}


def _matches_structure_file(path: Path, structure_file_format: FileFormats) -> bool:
    name = path.name.lower()
    suffix = path.suffix.lower()

    if structure_file_format == "poscar":
        return name in {"poscar", "contcar"}

    if structure_file_format == "aims":
        return name == "geometry.in"

    if structure_file_format:
        return suffix == f".{structure_file_format}"

    return name in _KNOWN_STRUCTURE_FILENAMES or suffix in _KNOWN_STRUCTURE_SUFFIXES


def _resolve_structure_file(directory: Path, structure_file_format: FileFormats) -> Path:
    candidates = [
        child
        for child in sorted(directory.iterdir())
        if child.is_file() and _matches_structure_file(child, structure_file_format)
    ]
    if not candidates:
        raise ValueError(
            f"No structure file found in {directory} for format {structure_file_format!r}"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple structure files found in {directory} for format {structure_file_format!r}: "
            + ", ".join(str(candidate) for candidate in candidates)
        )
    return candidates[0]


def resolve_structure_path(
    structures_path: str | Path,
    structure_pattern: str = "*",
    structure_file_format: FileFormats = "poscar",
) -> list[Path]:
    path = Path(structures_path)
    if not path.is_dir():
        raise ValueError(f"The provided structures_path is not a directory: {structures_path}")

    structure_dirs = sorted(candidate for candidate in path.glob(structure_pattern) if candidate.is_dir())
    if not structure_dirs:
        raise ValueError(f"No structure directories found matching pattern: {structure_pattern}")

    structures_filenames = [
        _resolve_structure_file(directory, str(structure_file_format).lower())
        for directory in structure_dirs
    ]

    return structures_filenames


def generate_perturbed_population(
    structure: Structure,
    input_structures_path: str | Path,
    num_structures: int,
    distance: float,
    min_distance: float | None = None,
    supercell_size: list[int] | None = None,
    file_format: FileFormats = "poscar"
) -> dict[str, Any]:

    normalized_format = str(file_format).lower()

    def _output_filename(index: int) -> str:
        if normalized_format == "poscar":
            return "POSCAR"
        if normalized_format == "aims":
            return "geometry.in"
        return f"structure_{index}.{normalized_format}"

    if supercell_size is not None:
        structure.make_supercell(supercell_size)
    input_structure_path = Path(input_structures_path)
    input_structure_path.mkdir(parents=True, exist_ok=True)

    structure_0_path = input_structure_path / "structure_0"
    structure_0_path.mkdir(parents=True, exist_ok=True)
    structure.to(filename=structure_0_path / _output_filename(0))

    for i in range(1, num_structures):
        perturbed_structure = structure.copy()
        perturbed_structure.perturb(distance=distance, min_distance=min_distance)
        perturbed_structure_path = input_structure_path / f"structure_{i}"
        perturbed_structure_path.mkdir(parents=True, exist_ok=True)
        perturbed_structure.to(filename=perturbed_structure_path / _output_filename(i))
    output = {
        "input_structures_path": str(input_structure_path.resolve()),
        "num_structures": num_structures,
        "distance": distance,
        "min_distance": min_distance,
        "supercell_size": supercell_size,
    }
    return output
    
