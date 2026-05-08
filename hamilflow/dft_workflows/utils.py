from pathlib import Path
from typing import Sequence

from ase.io import read as ase_read
from pymatgen.core.structure import FileFormats

__all__ = ["resolve_structure_path", "get_structure_names_from_path"]

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
) -> Path | list[Path]:
    path = Path(structures_path)
    
    # Handle single trajectory file (e.g., xyz, extxyz)
    if path.is_file():
        return path
    
    if not path.is_dir():
        raise ValueError(f"The provided structures_path is not a directory or file: {structures_path}")

    structure_dirs = sorted(candidate for candidate in path.glob(structure_pattern) if candidate.is_dir())
    if not structure_dirs:
        raise ValueError(f"No structure directories found matching pattern: {structure_pattern}")

    structures_filenames = [
        _resolve_structure_file(directory, str(structure_file_format).lower())
        for directory in structure_dirs
    ]

    return structures_filenames


def get_structure_names_from_path(
    structures_path: str | Path,
    structures_filenames: Path | Sequence[Path],
) -> list[str]:
    """
    Extract structure names from either trajectory file or directory structure.
    
    If structures_path is a file (trajectory file), returns indexed names like
    structure_0000, structure_0001, etc. for each structure in the file.
    
    If structures_path is a directory, returns parent directory names for each file.
    """
    path = Path(structures_path)

    if path.is_file():
        # Trajectory file: determine number of frames using ASE and return indexed names
        ase_structures = ase_read(str(path), index=":")
        if not isinstance(ase_structures, list):
            ase_structures = [ase_structures]
        return [f"structure_{i:04d}" for i in range(len(ase_structures))]

    # Directory structure: return parent directory names
    return [file_path.parent.name for file_path in structures_filenames]



