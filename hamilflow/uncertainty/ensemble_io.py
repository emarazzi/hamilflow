import logging
import shutil
from collections.abc import Iterable, Sequence
from glob import glob
from pathlib import Path

logger = logging.getLogger(__name__)


def discover_structures(pattern: str) -> list[Path]:
    """Return absolute Paths matching a glob pattern for structures.

    Example pattern: '/path/to/dft/str*' or './input_dir/dft/str*'
    """
    paths = [Path(p).absolute() for p in glob(pattern)]
    if not paths:
        logger.warning(f"No structures match {pattern!r}")
    return paths


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

    missing = []
    for s in structures:
        s = Path(s)
        dst = dst_parent / ensemble_dir / s.name
        if not dst.is_dir():
            missing.append(s.name)
        else:
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
                    except Exception:  # noqa: BLE001
                        # Fall back to copying if symlinks are unavailable
                        logger.debug(f"could not symlink {dst_file}, copying {src_file} instead")
                        shutil.copy2(src_file, dst_file)
    if missing:
        logger.warning(f"{len(missing)} structures have no folder in {dst_parent / ensemble_dir} and were skipped: "
                       f"{missing[:5]}{' ...' if len(missing) > 5 else ''}")


__all__ = ["discover_structures", "link_ensemble_files"]
