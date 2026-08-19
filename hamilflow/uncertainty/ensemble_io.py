from glob import glob
import shutil
from pathlib import Path
from typing import Iterable, Sequence


def discover_structures(pattern: str) -> list[Path]:
    """Return absolute Paths matching a glob pattern for structures.

    Example pattern: '/path/to/dft/str*' or './input_dir/dft/str*'
    """
    return [Path(p).absolute() for p in glob(pattern)]


def link_ensemble_files(
    structures: Iterable[Path],
    ensemble_dir: Path,
    dst_parent: Path,
    hamiltonian_pred_name: str = "hamiltonian_pred.h5",
    hamiltonian_name: str = "hamiltonian.h5",
    symlink_files: Sequence[str] = ("POSCAR", "info.json", "overlap.h5"),
):
    """For each structure path, ensure ensemble files are placed under dst_parent/ensemble_dir/<structure.name>.

    - If <dst>/hamiltonian.h5 is missing but <dst>/hamiltonian_pred.h5 exists, copy it.
    - Create symlinks for the files in `symlink_files` from the source structure into the dst.
    """
    ensemble_dir = Path(ensemble_dir)
    dst_parent = Path(dst_parent)

    for s in structures:
        s = Path(s)
        dst = dst_parent / ensemble_dir / s.name
        if dst.is_dir():
            dst_file = dst / hamiltonian_name
            src_file = dst / hamiltonian_pred_name
            if not dst_file.exists() and src_file.exists():
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)

            for fname in symlink_files:
                src_file = s / fname
                dst_file = dst / fname
                if src_file.exists() and not dst_file.exists():
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        dst_file.symlink_to(src_file)
                    except Exception:
                        # Fall back to copying if symlinks are unavailable
                        shutil.copy2(src_file, dst_file)


__all__ = ["discover_structures", "link_ensemble_files"]
