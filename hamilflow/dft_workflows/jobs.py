from pathlib import Path
from shutil import move, rmtree
from typing import Sequence

from jobflow.core.flow import Flow
from jobflow.core.job import Job, job
from jobflow.core.maker import Maker
from pymatgen.core import Structure


def build_aims_dft_jobs(
    structures_filenames: Sequence[Path], aims_maker: Maker
) -> list[Flow | Job]:
    jobs: list[Flow | Job] = []
    for structure_file in structures_filenames:
        structure = Structure.from_file(structure_file)
        structure_name = structure_file.parent.name
        aims_job = aims_maker.make(structure)
        aims_job.name = f"aims_static_{structure_name}"
        jobs.append(aims_job)
    return jobs


@job
def collect_aims_outputs(
    source_run_dirs: list[str | Path],
    structure_names: list[str],
    collected_runs_root: str | Path,
) -> dict[str, str | list[str]]:
    if not source_run_dirs:
        raise ValueError("No source_run_dirs were provided for output collection.")

    if len(source_run_dirs) != len(structure_names):
        raise ValueError(
            "source_run_dirs and structure_names must have the same length for "
            "deterministic output naming."
        )

    output_root = Path(collected_runs_root)
    output_root.mkdir(parents=True, exist_ok=True)

    collected_dirs: list[str] = []
    for source, structure_name in zip(source_run_dirs, structure_names, strict=True):
        source_dir = Path(source)
        if not source_dir.is_dir():
            raise ValueError(f"AIMS output directory does not exist: {source_dir}")

        target_dir = output_root / structure_name
        if target_dir.exists():
            rmtree(target_dir)

        moved_dir = Path(move(str(source_dir), str(target_dir)))
        collected_dirs.append(str(moved_dir.resolve()))

    return {
        "collected_runs_root": str(output_root.resolve()),
        "collected_dirs": collected_dirs,
    }
