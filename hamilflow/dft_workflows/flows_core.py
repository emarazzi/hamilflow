from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, cast

from pymatgen.core.structure import FileFormats

__all__ = [
    "GenerateAimsToProjectedDeephData",
    "GenerateOvlOnlyAimsToProjectedDeephData",
    "GenerateTwoStepProjectedDeephInputs",
]

from jobflow.core.flow import Flow
from jobflow.core.job import Job

from .aims_makers import OvlOnlyAimsMaker
from .flows_base import (
    ConvertAimsToDeephConfig,
    GenerateAimsDFTData,
    ProjectDeephInputsConfig,
    resolve_projection_removal_plan,
)
from .jobs import run_projection_for_structure
from .utils import resolve_structure_path, get_structure_names_from_path
from pymatgen.io.aims.sets.core import StaticSetGenerator

@dataclass
class GenerateAimsToProjectedDeephData:
    """
    Convenience wrapper flow chaining:

    AIMS run/collect -> optional AIMS-to-DeepH conversion (required here) -> projection
    -> optional second projection stage.
    """

    dft_data_flow: GenerateAimsDFTData
    projection_config: ProjectDeephInputsConfig
    second_projection_config: ProjectDeephInputsConfig | None = None
    name: str = "generate_aims_to_projected_deeph_data"

    def _resolve_structure_names(self) -> list[str]:
        if self.dft_data_flow.aims_maker is not None:
            if self.dft_data_flow.structures_path is None:
                raise ValueError(
                    "structures_path is required when aims_maker is provided to run new "
                    "AIMS jobs."
                )
            structures_filenames = resolve_structure_path(
                self.dft_data_flow.structures_path,
                self.dft_data_flow.structure_pattern,
                self.dft_data_flow.structure_file_format,
            )
            structure_names = get_structure_names_from_path(
                self.dft_data_flow.structures_path, structures_filenames
            )
        else:
            source_run_dirs = [str(Path(path)) for path in self.dft_data_flow.source_run_dirs or []]
            structure_names = [Path(path).name for path in source_run_dirs]

        return [
            name
            for name in structure_names
            if fnmatch(name, self.projection_config.structure_pattern)
        ]

    def make(self) -> Flow:
        if self.dft_data_flow.aims_to_deeph_config is None:
            raise ValueError(
                "GenerateAimsToProjectedDeephData requires dft_data_flow.aims_to_deeph_config "
                "to be provided."
            )

        upstream_flow = self.dft_data_flow.make()
        structure_names = self._resolve_structure_names()

        projection_jobs: list[Job] = []
        first_projection_outputs: dict[str, dict[str, object]] = {}
        for structure_name in structure_names:
            removal_plan = resolve_projection_removal_plan(
                structure_name=structure_name,
                removal_plan=self.projection_config.removal_plan,
            )
            projection_job = run_projection_for_structure(
                structure_name=structure_name,
                deeph_inputs_root=self.dft_data_flow.aims_to_deeph_config.output_dir,
                projected_root=self.projection_config.output_root,
                removal_plan=removal_plan,
                kgrid=self.projection_config.kgrid,
                user_kpoints_settings=self.projection_config.user_kpoints_settings,
                reduction_mode=self.projection_config.reduction_mode,
                deeph_conversion_output=upstream_flow.output["deeph_inputs"],
            )
            projection_jobs.append(projection_job)
            first_projection_outputs[structure_name] = cast(dict[str, object], projection_job.output)

        second_projection_jobs: list[Job] = []
        second_outputs: dict[str, object] | None = None
        if self.second_projection_config is not None:
            first_root = Path(self.projection_config.output_root)
            second_root = Path(self.second_projection_config.output_root)
            if first_root.resolve() == second_root.resolve():
                raise ValueError(
                    "projection_config.output_root and second_projection_config.output_root "
                    "must be different directories."
                )

            second_structure_names = [
                name
                for name in structure_names
                if fnmatch(name, self.second_projection_config.structure_pattern)
            ]
            if not second_structure_names:
                raise ValueError(
                    "No projected DeepH structure directories matched the second projection "
                    f"pattern {self.second_projection_config.structure_pattern!r}."
                )

            for structure_name in second_structure_names:
                removal_plan = resolve_projection_removal_plan(
                    structure_name=structure_name,
                    removal_plan=self.second_projection_config.removal_plan,
                )
                second_projection_job = run_projection_for_structure(
                    structure_name=structure_name,
                    deeph_inputs_root=self.projection_config.output_root,
                    projected_root=self.second_projection_config.output_root,
                    removal_plan=removal_plan,
                    kgrid=self.second_projection_config.kgrid,
                    user_kpoints_settings=self.second_projection_config.user_kpoints_settings,
                    reduction_mode=self.second_projection_config.reduction_mode,
                    upstream_projection_output=first_projection_outputs[structure_name],
                )
                second_projection_jobs.append(second_projection_job)

            second_outputs = {
                "projected_root": str(second_root.resolve()),
                "structure_names": second_structure_names,
                "projection_results": [job.output for job in second_projection_jobs],
            }

        outputs = {
            "upstream": upstream_flow.output,
            "projected_deeph_inputs": {
                "projected_root": str(Path(self.projection_config.output_root).resolve()),
                "structure_names": structure_names,
                "projection_results": [job.output for job in projection_jobs],
            },
            "second_projected_deeph_inputs": second_outputs,
        }
        return Flow(
            jobs=[upstream_flow, *projection_jobs, *second_projection_jobs],
            name=self.name,
            output=outputs,
        )


@dataclass
class GenerateOvlOnlyAimsToProjectedDeephData:
    """
    Convenience wrapper flow chaining:

    AIMS run/collect using ``OvlOnlyAimsMaker`` -> optional AIMS-to-DeepH
    conversion (required here) -> projection -> optional second projection stage.
    """

    projection_config: ProjectDeephInputsConfig
    second_projection_config: ProjectDeephInputsConfig | None = None
    structures_path: str | Path | None = None
    structure_pattern: str = "*"
    structure_file_format: FileFormats = "poscar"
    name: str = "generate_ovl_only_aims_to_projected_deeph_data"
    aims_kwargs: dict[str, Any] = field(default_factory=dict)
    kgrid: tuple[int, int, int] | None = None
    kpoints_updates: dict[str, Any] | None = None
    user_kpoints_settings: dict[str, Any] | Any | None = None
    force_gamma: bool = True
    symprec: float = 1e-5
    aims_maker: OvlOnlyAimsMaker = field(
        default_factory=lambda: OvlOnlyAimsMaker(input_set_generator=StaticSetGenerator())
        )
                                   
    collected_runs_root: str | Path = "./aims_calculations"
    source_run_dirs: list[str | Path] | None = None
    aims_to_deeph_config: ConvertAimsToDeephConfig | None = None

    def _build_dft_data_flow(self) -> GenerateAimsDFTData:
        return GenerateAimsDFTData(
            structures_path=self.structures_path,
            structure_pattern=self.structure_pattern,
            structure_file_format=self.structure_file_format,
            name=self.name,
            aims_kwargs=self.aims_kwargs,
            kgrid=self.kgrid,
            kpoints_updates=self.kpoints_updates,
            user_kpoints_settings=self.user_kpoints_settings,
            force_gamma=self.force_gamma,
            symprec=self.symprec,
            aims_maker=self.aims_maker,
            collected_runs_root=self.collected_runs_root,
            source_run_dirs=self.source_run_dirs,
            aims_to_deeph_config=self.aims_to_deeph_config,
        )

    def make(self) -> Flow:
        return GenerateAimsToProjectedDeephData(
            dft_data_flow=self._build_dft_data_flow(),
            projection_config=self.projection_config,
            second_projection_config=self.second_projection_config,
            name=self.name,
        ).make()


@dataclass
class GenerateTwoStepProjectedDeephInputs:
    """
    Run two chained projection stages over an existing DeepH input root.

    A common setup is stage 1 ``schur`` then stage 2 ``truncate``. Stage 2 jobs
    depend explicitly on stage 1 outputs to prevent race conditions.
    """

    deeph_inputs_root: str | Path
    first_projection_config: ProjectDeephInputsConfig
    second_projection_config: ProjectDeephInputsConfig
    name: str = "generate_two_step_projected_deeph_inputs"

    def make(self) -> Flow:
        deeph_inputs_root = Path(self.deeph_inputs_root)
        if not deeph_inputs_root.is_dir():
            raise ValueError(f"DeepH input root does not exist: {deeph_inputs_root}")

        first_root = Path(self.first_projection_config.output_root)
        second_root = Path(self.second_projection_config.output_root)
        if first_root.resolve() == second_root.resolve():
            raise ValueError(
                "first_projection_config.output_root and second_projection_config.output_root "
                "must be different directories."
            )

        structure_names = sorted(
            path.name
            for path in deeph_inputs_root.glob(self.first_projection_config.structure_pattern)
            if path.is_dir() and fnmatch(path.name, self.second_projection_config.structure_pattern)
        )
        if not structure_names:
            raise ValueError(
                "No DeepH structure directories found under "
                f"{deeph_inputs_root} matching both patterns "
                f"{self.first_projection_config.structure_pattern!r} and "
                f"{self.second_projection_config.structure_pattern!r}."
            )

        first_stage_jobs: list[Job] = []
        second_stage_jobs: list[Job] = []

        for structure_name in structure_names:
            first_removal_plan = resolve_projection_removal_plan(
                structure_name=structure_name,
                removal_plan=self.first_projection_config.removal_plan,
            )
            first_job = cast(
                Job,
                run_projection_for_structure(
                    structure_name=structure_name,
                    deeph_inputs_root=deeph_inputs_root,
                    projected_root=self.first_projection_config.output_root,
                    removal_plan=first_removal_plan,
                    kgrid=self.first_projection_config.kgrid,
                    user_kpoints_settings=self.first_projection_config.user_kpoints_settings,
                    reduction_mode=self.first_projection_config.reduction_mode,
                ),
            )
            first_stage_jobs.append(first_job)

            second_removal_plan = resolve_projection_removal_plan(
                structure_name=structure_name,
                removal_plan=self.second_projection_config.removal_plan,
            )
            second_job = cast(
                Job,
                run_projection_for_structure(
                    structure_name=structure_name,
                    deeph_inputs_root=self.first_projection_config.output_root,
                    projected_root=self.second_projection_config.output_root,
                    removal_plan=second_removal_plan,
                    kgrid=self.second_projection_config.kgrid,
                    user_kpoints_settings=self.second_projection_config.user_kpoints_settings,
                    reduction_mode=self.second_projection_config.reduction_mode,
                    upstream_projection_output=first_job.output,
                ),
            )
            second_stage_jobs.append(second_job)

        outputs = {
            "deeph_inputs_root": str(deeph_inputs_root.resolve()),
            "first_stage": {
                "projected_root": str(first_root.resolve()),
                "reduction_mode": self.first_projection_config.reduction_mode,
                "structure_names": structure_names,
                "projection_results": [job.output for job in first_stage_jobs],
            },
            "second_stage": {
                "projected_root": str(second_root.resolve()),
                "reduction_mode": self.second_projection_config.reduction_mode,
                "structure_names": structure_names,
                "projection_results": [job.output for job in second_stage_jobs],
            },
        }
        return Flow(
            jobs=[*first_stage_jobs, *second_stage_jobs],
            name=self.name,
            output=outputs,
        )

