from pathlib import Path
from shutil import move, rmtree
from typing import Sequence

from jobflow.core.flow import Flow
from jobflow.core.job import Job, job
from jobflow.core.maker import Maker
from pymatgen.core import Structure
from deepx_dock.convert.fhi_aims.aims_to_deeph import PeriodicAimsDataTranslator


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
        source_path = str(source)
        # Strip cluster name if path is remote (e.g., "cluster.host.com:/path/to/dir")
        if ":" in source_path and not source_path.startswith("/"):
            source_path = source_path.split(":", 1)[1]

        source_dir = Path(source_path)
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


@job
def convert_aims_to_deeph(
    input_root: str | Path,
    output_dirs: str | Path,
    jobs_num: int = 1,
    tier_num: int = 1,
) -> dict[str, str]:
    input_root = Path(input_root)
    if not input_root.is_dir():
        raise ValueError(f"Input directory does not exist: {input_root}")

    output_dirs = Path(output_dirs)
    output_dirs.mkdir(parents=True, exist_ok=True)

    translator = PeriodicAimsDataTranslator(
        input_root,
        output_dirs,
        export_rho=False,
        export_r=False,
        n_jobs=jobs_num,
        n_tier=tier_num,
    )

    translator.transfer_all_aims_to_deeph()

    return {
        "deeph_inputs_root": str(output_dirs.resolve()),
    }
