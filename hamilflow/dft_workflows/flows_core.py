from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import cast

__all__ = ["GenerateAimsToProjectedDeephData", "GenerateTwoStepProjectedDeephInputs"]

from jobflow.core.flow import Flow
from jobflow.core.job import Job

from .flows_base import (
    GenerateAimsDFTData,
    ProjectDeephInputsConfig,
    resolve_projection_removal_plan,
)
from .jobs import run_projection_for_structure
from .utils import resolve_structure_path


@dataclass
class GenerateAimsToProjectedDeephData:
    """
    Convenience wrapper flow chaining:

    AIMS run/collect -> optional AIMS-to-DeepH conversion (required here) -> projection.
    """

    dft_data_flow: GenerateAimsDFTData
    projection_config: ProjectDeephInputsConfig
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
            structure_names = [path.parent.name for path in structures_filenames]
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
                reduction_mode=self.projection_config.reduction_mode,
                deeph_conversion_output=upstream_flow.output["deeph_inputs"],
            )
            projection_jobs.append(projection_job)

        outputs = {
            "upstream": upstream_flow.output,
            "projected_deeph_inputs": {
                "projected_root": str(Path(self.projection_config.output_root).resolve()),
                "structure_names": structure_names,
                "projection_results": [job.output for job in projection_jobs],
            },
        }
        return Flow(jobs=[upstream_flow, *projection_jobs], name=self.name, output=outputs)


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

