from pathlib import Path
from fnmatch import fnmatch
from shutil import move, rmtree
from typing import Any, Sequence

from jobflow.core.flow import Flow
from jobflow.core.job import Job, job
from jobflow.core.maker import Maker
from pymatgen.core import Structure
from deepx_dock.convert.fhi_aims.aims_to_deeph import PeriodicAimsDataTranslator

from ..projection import ProjectionConfig, RemovalPlanLike, ReductionMode, run_projection


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
    output_dir: str | Path,
    jobs_num: int = 1,
    tier_num: int = 0,
) -> dict[str, str]:
    input_root = Path(input_root)
    if not input_root.is_dir():
        raise ValueError(f"Input directory does not exist: {input_root}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    translator = PeriodicAimsDataTranslator(
        input_root,
        output_dir,
        export_rho=False,
        export_r=False,
        n_jobs=jobs_num,
        n_tier=tier_num,
    )

    translator.transfer_all_aims_to_deeph()

    return {
        "deeph_inputs_root": str(output_dir.resolve()),
    }


def resolve_structure_removal_plan(
    structure_name: str,
    default_plan: RemovalPlanLike,
    per_structure: dict[str, RemovalPlanLike] | None = None,
    pattern_overrides: dict[str, RemovalPlanLike] | None = None,
) -> RemovalPlanLike:
    """
    Resolve the removal plan for one structure directory.

    Precedence is deterministic and intentionally simple:
    1) exact per_structure match
    2) first matching glob pattern (sorted by pattern key)
    3) default_plan
    """
    if per_structure and structure_name in per_structure:
        return per_structure[structure_name]

    if pattern_overrides:
        for pattern in sorted(pattern_overrides):
            if fnmatch(structure_name, pattern):
                return pattern_overrides[pattern]

    return default_plan


@job
def run_projection_for_structure(
    structure_name: str,
    deeph_inputs_root: str | Path,
    projected_root: str | Path,
    removal_plan: RemovalPlanLike,
    kgrid: tuple[int, int, int] = (4, 4, 4),
    reduction_mode: ReductionMode = "schur",
    deeph_conversion_output: dict[str, str] | None = None,
    upstream_projection_output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run projection for one converted DeepH subdirectory.

    ``deeph_conversion_output`` is optional and mainly used to create an explicit
    dependency on an upstream conversion job in chained flows.

    ``upstream_projection_output`` is an optional dependency-only payload used
    to force ordering between projection stages in multi-step workflows.
    """
    # Kept as a dependency token for chained projection stages.
    _ = upstream_projection_output

    if deeph_conversion_output is not None:
        if "deeph_inputs_root" not in deeph_conversion_output:
            raise ValueError("deeph_conversion_output is missing 'deeph_inputs_root'.")
        deeph_inputs_root = deeph_conversion_output["deeph_inputs_root"]

    if reduction_mode not in ("schur", "truncate"):
        raise ValueError(
            f"Unsupported reduction_mode '{reduction_mode}'. Expected 'schur' or 'truncate'."
        )

    deeph_inputs_root = Path(deeph_inputs_root)
    input_dir = deeph_inputs_root / structure_name
    if not input_dir.is_dir():
        raise ValueError(f"DeepH structure directory does not exist: {input_dir}")

    projected_root = Path(projected_root)
    output_dir = projected_root / structure_name

    if len(kgrid) != 3:
        raise ValueError(f"kgrid must contain exactly three integers, got: {kgrid}")
    kgrid_3: tuple[int, int, int] = (int(kgrid[0]), int(kgrid[1]), int(kgrid[2]))

    result = run_projection(
        config=ProjectionConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            kgrid=kgrid_3,
            reduction_mode=reduction_mode,
        ),
        removal_plan=removal_plan,
    )

    payload = result.to_dict()
    payload["structure_name"] = structure_name
    return payload
