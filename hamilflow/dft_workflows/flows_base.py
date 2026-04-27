from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast, get_args

__all__ = [
    "ConvertAimsToDeephConfig",
    "DEFAULT_AIMS_KWARGS",
    "GenerateAimsDFTData",
    "GenerateProjectedDeephInputs",
    "ProjectDeephInputsConfig",
    "ProjectionRemovalPlanConfig",
    "resolve_projection_removal_plan",
]

from atomate2.aims.jobs.core import StaticMaker
from jobflow.core.flow import Flow
from jobflow.core.job import Job
from jobflow.core.maker import Maker
from pymatgen.core.structure import FileFormats
from pymatgen.io.aims.sets.core import StaticSetGenerator

from ..projection.models import ReductionMode, RemovalPlanLike
from .jobs import (
    build_aims_dft_jobs,
    collect_aims_outputs,
    convert_aims_to_deeph,
    resolve_structure_removal_plan,
    run_projection_for_structure,
)
from .utils import resolve_structure_path

DEFAULT_AIMS_KWARGS: dict[str, Any] = {"output_rs_matrices": "plain"}


@dataclass(frozen=True, slots=True)
class ConvertAimsToDeephConfig:
    """Configuration for the optional AIMS-to-DeepH conversion step."""

    output_dir: str | Path
    jobs_num: int = 1
    tier_num: int = 0


@dataclass(frozen=True, slots=True)
class ProjectionRemovalPlanConfig:
    """
    Optional per-structure removal-plan configuration for projection workflows.

    Precedence is:
    1) ``per_structure`` exact match
    2) ``pattern_overrides`` glob match (deterministic sorted iteration)
    3) ``default_plan``
    """

    default_plan: RemovalPlanLike
    per_structure: dict[str, RemovalPlanLike] = field(default_factory=dict)
    pattern_overrides: dict[str, RemovalPlanLike] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProjectDeephInputsConfig:
    """Configuration for projecting a DeepH-formatted directory tree."""

    output_root: str | Path
    removal_plan: RemovalPlanLike | ProjectionRemovalPlanConfig
    structure_pattern: str = "*"
    kgrid: tuple[int, int, int] = (4, 4, 4)
    reduction_mode: ReductionMode = "schur"


def resolve_projection_removal_plan(
    structure_name: str,
    removal_plan: RemovalPlanLike | ProjectionRemovalPlanConfig,
) -> RemovalPlanLike:
    if isinstance(removal_plan, ProjectionRemovalPlanConfig):
        return resolve_structure_removal_plan(
            structure_name=structure_name,
            default_plan=removal_plan.default_plan,
            per_structure=removal_plan.per_structure,
            pattern_overrides=removal_plan.pattern_overrides,
        )
    return removal_plan


@dataclass
class GenerateAimsDFTData:
    """
    Create a flow for FHI-aims calculations, output collection, and optional
    DeepH input conversion.

    Two modes are supported:

    - Run-and-collect: provide ``aims_maker`` and ``structures_path``. The flow
      resolves structures under ``structures_path``, launches AIMS jobs, and then
      moves the produced run directories into ``collected_runs_root``.
    - Collect-only: provide ``source_run_dirs`` and set ``aims_maker`` to ``None``.
      In this mode the flow skips structure discovery and only moves the existing
      run directories into ``collected_runs_root``.

    If ``aims_to_deeph_config`` is provided, the collected runs are passed to the
    DeepH conversion job and the resulting output root is included in the flow
    outputs.

    ``aims_kwargs`` are applied only when a ``StaticMaker`` is used. If an explicit
    ``StaticMaker`` is provided and ``aims_kwargs`` is empty, the maker is preserved
    as-is.

    The collected directory names are derived from the original structure names so
    downstream outputs remain easy to map back to their source structures.
    """

    structures_path: str | Path | None = None
    structure_pattern: str = "*"
    structure_file_format: FileFormats = "poscar"
    name: str = "generate_aims_dft_data"
    aims_kwargs: dict[str, Any] = field(default_factory=dict)
    aims_maker: Maker | None = field(
        default_factory=lambda: StaticMaker(input_set_generator=StaticSetGenerator())
    )
    collected_runs_root: str | Path = "./aims_calculations"
    source_run_dirs: list[str | Path] | None = None
    aims_to_deeph_config: ConvertAimsToDeephConfig | None = None

    def __post_init__(self):
        merged_aims_kwargs = {**DEFAULT_AIMS_KWARGS, **self.aims_kwargs}

        if self.structure_file_format and self.structure_file_format not in get_args(FileFormats):
            raise ValueError(
                f"Unsupported structure_file_format: {self.structure_file_format}. "
                f"Supported formats are: {[fmt for fmt in get_args(FileFormats)]}"
            )

        if self.aims_maker is not None and self.structures_path is None:
            raise ValueError(
                "structures_path is required when aims_maker is provided to run new "
                "AIMS jobs."
            )

        if self.aims_maker is None and not self.source_run_dirs:
            raise ValueError(
                "Provide aims_maker to run AIMS jobs or source_run_dirs to collect "
                "existing runs."
            )

        if self.aims_maker is not None and isinstance(self.aims_maker, StaticMaker):
            if self.aims_kwargs:
                static_maker = cast(StaticMaker, self.aims_maker)
                static_maker.input_set_generator = StaticSetGenerator(
                    user_params=dict(**merged_aims_kwargs)
                )
        elif self.aims_maker is not None and self.aims_kwargs:
            raise ValueError(
                "aims_kwargs are only supported with StaticMaker. Provide a configured "
                "custom maker or use the default StaticMaker path."
            )

        self.aims_kwargs = merged_aims_kwargs

    def make(self) -> Flow:
        jobs: list[Flow | Job] = []
        convert_job: Job | None = None
        if self.aims_maker is not None:
            if self.structures_path is None:
                raise ValueError(
                    "structures_path is required when aims_maker is provided to run new "
                    "AIMS jobs."
                )
            structures_filenames = resolve_structure_path(
                self.structures_path, self.structure_pattern, self.structure_file_format
            )
            aims_jobs = build_aims_dft_jobs(structures_filenames, self.aims_maker)
            jobs.extend(aims_jobs)
            source_run_dirs = [job.output.dir_name for job in aims_jobs]
            structure_names = [path.parent.name for path in structures_filenames]
        else:
            source_run_dirs = [str(Path(path)) for path in self.source_run_dirs or []]
            structure_names = [Path(path).name for path in source_run_dirs]

        collect_job = collect_aims_outputs(
            source_run_dirs=source_run_dirs,
            structure_names=structure_names,
            collected_runs_root=self.collected_runs_root,
        )
        jobs.append(collect_job)

        if self.aims_to_deeph_config is not None:
            convert_job = convert_aims_to_deeph(
                input_root=collect_job.output["collected_runs_root"],
                output_dir=self.aims_to_deeph_config.output_dir,
                jobs_num=self.aims_to_deeph_config.jobs_num,
                tier_num=self.aims_to_deeph_config.tier_num,
            )
            jobs.append(convert_job)

        outputs = {
            "aims_kwargs": self.aims_kwargs,
            "collection": collect_job.output,
            "deeph_inputs": convert_job.output if convert_job is not None else None,
        }
        return Flow(jobs=jobs, name=self.name, output=outputs)


@dataclass
class GenerateProjectedDeephInputs:
    """
    Create a projection-only flow over an existing DeepH input root.

    One projection job is created per matched child directory.
    """

    deeph_inputs_root: str | Path
    projection_config: ProjectDeephInputsConfig
    name: str = "generate_projected_deeph_inputs"

    def make(self) -> Flow:
        deeph_inputs_root = Path(self.deeph_inputs_root)
        if not deeph_inputs_root.is_dir():
            raise ValueError(f"DeepH input root does not exist: {deeph_inputs_root}")

        structure_names = sorted(
            path.name
            for path in deeph_inputs_root.glob(self.projection_config.structure_pattern)
            if path.is_dir()
        )
        if not structure_names:
            raise ValueError(
                "No DeepH structure directories found under "
                f"{deeph_inputs_root} matching pattern {self.projection_config.structure_pattern}"
            )

        projection_jobs: list[Job] = []
        for structure_name in structure_names:
            removal_plan = resolve_projection_removal_plan(
                structure_name=structure_name,
                removal_plan=self.projection_config.removal_plan,
            )
            projection_job = run_projection_for_structure(
                structure_name=structure_name,
                deeph_inputs_root=deeph_inputs_root,
                projected_root=self.projection_config.output_root,
                removal_plan=removal_plan,
                kgrid=self.projection_config.kgrid,
                reduction_mode=self.projection_config.reduction_mode,
            )
            projection_jobs.append(projection_job)

        outputs = {
            "deeph_inputs_root": str(deeph_inputs_root.resolve()),
            "projected_root": str(Path(self.projection_config.output_root).resolve()),
            "structure_names": structure_names,
            "projection_results": [job.output for job in projection_jobs],
        }
        return Flow(jobs=projection_jobs, name=self.name, output=outputs)

