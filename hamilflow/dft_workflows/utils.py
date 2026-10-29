from collections.abc import Sequence
from pathlib import Path

from ase import Atoms
from ase.io import read as ase_read
from pymatgen.core import Structure
from pymatgen.core.structure import FileFormats
from pymatgen.io.ase import AseAtomsAdaptor

__all__ = ["resolve_structure_path", "get_structure_names_from_path", "load_structures"]

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


def _resolve_structure_file(
    directory: Path, structure_file_format: FileFormats
) -> Path:
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
    structure_pattern: str | list[str] = "*",
    structure_file_format: FileFormats = "poscar",
) -> Path | list[Path]:
    path = Path(structures_path)

    # Handle single trajectory file (e.g., xyz, extxyz)
    if path.is_file():
        return path

    if not path.is_dir():
        raise ValueError(
            f"The provided structures_path is not a directory or file: {structures_path}"
        )

    patterns = (
        [structure_pattern] if isinstance(structure_pattern, str) else structure_pattern
    )
    structure_dirs = sorted({c for p in patterns for c in path.glob(p) if c.is_dir()})
    if not structure_dirs:
        raise ValueError(
            f"No structure directories found matching pattern: {structure_pattern}"
        )

    structures_filenames = [
        _resolve_structure_file(directory, structure_file_format)
        for directory in structure_dirs
    ]

    return structures_filenames


def _read_trajectory(path: Path) -> list[Atoms]:
    ase_structures = ase_read(str(path), index=":")
    if not isinstance(ase_structures, list):
        ase_structures = [ase_structures]
    return ase_structures


def _trajectory_names(ase_structures: Sequence[Atoms]) -> list[str]:
    """
    Name the frames of a trajectory file.

    Frames are named after their ``index`` info entry when every frame carries
    one, whatever the file format; otherwise frames are named structure_0000,
    structure_0001, ...
    """
    structure_indices = [atoms.info.get("index") for atoms in ase_structures]
    if all(index is not None for index in structure_indices):
        return [str(index) for index in structure_indices]

    return [f"structure_{i:04d}" for i in range(len(ase_structures))]


def get_structure_names_from_path(
    structures_path: str | Path,
    structures_filenames: Path | Sequence[Path],
) -> list[str]:
    """
    Extract structure names from either trajectory file or directory structure.

    If structures_path is a file (trajectory file), frames are named as described
    in ``_trajectory_names``.

    If structures_path is a directory, returns parent directory names for each file.
    """
    path = Path(structures_path)

    if path.is_file():
        return _trajectory_names(_read_trajectory(path))

    # Directory structure: return parent directory names
    if isinstance(structures_filenames, Path):
        return [structures_filenames.parent.name]

    return [file_path.parent.name for file_path in structures_filenames]


def load_structures(
    structures_path: str | Path,
    structure_pattern: str | list[str] = "*",
    structure_file_format: FileFormats = "poscar",
) -> list[tuple[str, Structure]]:
    """
    Resolve, read and name every structure under ``structures_path``.

    This is the single source of truth for structure names: the same name is used
    for the AIMS job, the collected run directory, the DeepH directory and the
    projected directory of each structure.
    """
    path = Path(structures_path)

    if path.is_file():
        ase_structures = _read_trajectory(path)
        names = _trajectory_names(ase_structures)
        entries = [
            (name, AseAtomsAdaptor.get_structure(atoms))
            for name, atoms in zip(names, ase_structures, strict=True)
        ]
    else:
        structures_filenames = resolve_structure_path(
            path, structure_pattern, structure_file_format
        )
        if isinstance(structures_filenames, Path):
            structures_filenames = [structures_filenames]
        entries = [
            (file_path.parent.name, Structure.from_file(file_path))
            for file_path in structures_filenames
        ]

    names = [name for name, _ in entries]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate structure names under {structures_path}: {duplicates}")

    return entries
