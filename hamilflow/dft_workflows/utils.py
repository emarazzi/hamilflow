from pathlib import Path


def resolve_structure_path(
    structures_path: str | Path, structure_pattern: str = "structure_*"
) -> list[Path]:
    path = Path(structures_path)
    if not path.is_dir():
        raise ValueError(f"The provided structures_path is not a directory: {structures_path}")

    structure_dirs = sorted(path.glob(structure_pattern))
    if not structure_dirs:
        raise ValueError(f"No structure directories found matching pattern: {structure_pattern}")

    structures_filenames = [directory / "POSCAR" for directory in structure_dirs]
    missing_poscars = [str(poscar) for poscar in structures_filenames if not poscar.is_file()]
    if missing_poscars:
        raise ValueError(
            "Missing POSCAR in structure directories: " + ", ".join(sorted(missing_poscars))
        )

    return structures_filenames
