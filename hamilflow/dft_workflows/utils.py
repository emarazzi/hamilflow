from pathlib import Path

from pymatgen.core.structure import FileFormats

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
