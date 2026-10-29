import logging
import os
import warnings
from dataclasses import dataclass, field, replace
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
from pymatgen.io.aims.sets.core import StaticSetGenerator

from .aims_makers import OvlOnlyAimsMaker
from .flows_base import (
    ConvertAimsToDeephConfig,
    GenerateAimsDFTData,
    ProjectDeephInputsConfig,
    resolve_projection_removal_plan,
)
from .jobs import run_projection_for_structure, set_projection_job_name

logger = logging.getLogger(__name__)

# Added on top of DEFAULT_AIMS_KWARGS for overlap-only runs: stop FHI-aims
# right after writing the real-space matrices.
DEFAULT_OVL_ONLY_AIMS_KWARGS: dict[str, Any] = {
    "output_rs_matrices_stop": "",
}

@dataclass
class GenerateAimsToProjectedDeephData:
    """
    Convenience wrapper flow chaining, independently for every structure:

    AIMS run -> collect -> AIMS-to-DeepH conversion (required here) -> projection
    -> optional second projection stage.

    The flow contains one sub-flow per structure (named after it). Each job only
    depends on the previous job of the same structure, so a structure is
    projected as soon as its own conversion finishes, regardless of the others.
    Structures not matching ``projection_config.structure_pattern`` still run the
    DFT/conversion stages but are not projected.
    """

    dft_data_flow: GenerateAimsDFTData
    projection_config: ProjectDeephInputsConfig
    second_projection_config: ProjectDeephInputsConfig | None = None
    name: str = "generate_aims_to_projected_deeph_data"

    def make(self) -> Flow:
        aims_to_deeph_config = self.dft_data_flow.aims_to_deeph_config
        if aims_to_deeph_config is None:
            raise ValueError(
                "GenerateAimsToProjectedDeephData requires dft_data_flow.aims_to_deeph_config "
                "to be provided."
            )

        entries = self.dft_data_flow.structure_entries()
        structure_names = [
            name
            for name, _ in entries
            if fnmatch(name, self.projection_config.structure_pattern)
        ]
        if entries and not structure_names:
            warnings.warn(
                f"No structure matches projection_config.structure_pattern "
                f"{self.projection_config.structure_pattern!r}: the flow runs DFT only, no projection",
                # attributed to the first caller outside hamilflow
                skip_file_prefixes=(os.path.dirname(os.path.dirname(__file__)) + os.sep,),
            )

        second_structure_names: list[str] = []
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

        structure_flows: list[Flow] = []
        dft_by_structure: dict[str, dict[str, Any]] = {}
        first_results: dict[str, Any] = {}
        second_results: dict[str, Any] = {}

        for structure_name, source in entries:
            jobs, dft_outputs = self.dft_data_flow.make_structure_jobs(structure_name, source)
            dft_by_structure[structure_name] = dft_outputs
            structure_output: dict[str, Any] = {"dft": dft_outputs}

            if structure_name in structure_names:
                projection_job = run_projection_for_structure(
                    structure_name=structure_name,
                    deeph_inputs_root=aims_to_deeph_config.output_dir,
                    projected_root=self.projection_config.output_root,
                    removal_plan=resolve_projection_removal_plan(
                        structure_name=structure_name,
                        removal_plan=self.projection_config.removal_plan,
                    ),
                    kgrid=self.projection_config.kgrid,
                    user_kpoints_settings=self.projection_config.user_kpoints_settings,
                    force_2d=self.projection_config.force_2d,
                    force_1d=self.projection_config.force_1d,
                    reduction_mode=self.projection_config.reduction_mode,
                    overlap_only=self.projection_config.overlap_only,
                    write_dummy_hamiltonian=self.projection_config.write_dummy_hamiltonian,
                    n_workers=self.projection_config.n_workers,
                    deeph_conversion_output=dft_outputs["deeph"],
                )
                projection_job = set_projection_job_name(
                    projection_job,
                    structure_name=structure_name,
                    reduction_mode=self.projection_config.reduction_mode,
                )
                jobs.append(projection_job)
                first_results[structure_name] = projection_job.output
                structure_output["projection"] = projection_job.output

                if (
                    self.second_projection_config is not None
                    and structure_name in second_structure_names
                ):
                    second_projection_job = run_projection_for_structure(
                        structure_name=structure_name,
                        deeph_inputs_root=self.projection_config.output_root,
                        projected_root=self.second_projection_config.output_root,
                        removal_plan=resolve_projection_removal_plan(
                            structure_name=structure_name,
                            removal_plan=self.second_projection_config.removal_plan,
                        ),
                        kgrid=self.second_projection_config.kgrid,
                        user_kpoints_settings=self.second_projection_config.user_kpoints_settings,
                        force_2d=self.second_projection_config.force_2d,
                        force_1d=self.second_projection_config.force_1d,
                        reduction_mode=self.second_projection_config.reduction_mode,
                        overlap_only=self.second_projection_config.overlap_only,
                        write_dummy_hamiltonian=self.second_projection_config.write_dummy_hamiltonian,
                        n_workers=self.second_projection_config.n_workers,
                        upstream_projection_output=projection_job.output,
                    )
                    second_projection_job = set_projection_job_name(
                        second_projection_job,
                        structure_name=structure_name,
                        reduction_mode=self.second_projection_config.reduction_mode,
                    )
                    jobs.append(second_projection_job)
                    second_results[structure_name] = second_projection_job.output
                    structure_output["second_projection"] = second_projection_job.output

            structure_flows.append(
                Flow(jobs=jobs, name=structure_name, output=structure_output)
            )

        second_outputs: dict[str, object] | None = None
        if self.second_projection_config is not None:
            second_outputs = {
                "projected_root": str(Path(self.second_projection_config.output_root).resolve()),
                "structure_names": second_structure_names,
                "projection_results": second_results,
            }

        outputs = {
            "upstream": self.dft_data_flow.flow_output(dft_by_structure),
            "projected_deeph_inputs": {
                "projected_root": str(Path(self.projection_config.output_root).resolve()),
                "structure_names": structure_names,
                "projection_results": first_results,
            },
            "second_projected_deeph_inputs": second_outputs,
        }
        logger.info(
            f"{self.name}: {len(entries)} structures, {len(structure_names)} projected"
            + (f", {len(second_structure_names)} projected twice" if self.second_projection_config is not None else "")
        )
        return Flow(jobs=structure_flows, name=self.name, output=outputs)


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
    force_2d: bool = False
    force_1d: bool = False
    symprec: float = 1e-5
    aims_maker: OvlOnlyAimsMaker = field(
        default_factory=lambda: OvlOnlyAimsMaker(input_set_generator=StaticSetGenerator())
        )
                                   
    collected_runs_root: str | Path = "./aims_calculations"
    source_run_dirs: list[str | Path] | None = None
    aims_to_deeph_config: ConvertAimsToDeephConfig | None = None

    def _build_dft_data_flow(self) -> GenerateAimsDFTData:
        # The overlap-only Hamiltonian is a placeholder: store it as zeros.
        aims_to_deeph_config = (
            replace(self.aims_to_deeph_config, dummy_hamiltonian=True)
            if self.aims_to_deeph_config is not None
            else None
        )
        return GenerateAimsDFTData(
            structures_path=self.structures_path,
            structure_pattern=self.structure_pattern,
            structure_file_format=self.structure_file_format,
            name=self.name,
            aims_kwargs={**DEFAULT_OVL_ONLY_AIMS_KWARGS, **self.aims_kwargs},
            kgrid=self.kgrid,
            kpoints_updates=self.kpoints_updates,
            user_kpoints_settings=self.user_kpoints_settings,
            force_gamma=self.force_gamma,
            force_2d=self.force_2d,
            force_1d=self.force_1d,
            symprec=self.symprec,
            aims_maker=self.aims_maker,
            collected_runs_root=self.collected_runs_root,
            source_run_dirs=self.source_run_dirs,
            aims_to_deeph_config=aims_to_deeph_config,
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
                    force_2d=self.first_projection_config.force_2d,
                    force_1d=self.first_projection_config.force_1d,
                    reduction_mode=self.first_projection_config.reduction_mode,
                    overlap_only=self.first_projection_config.overlap_only,
                    write_dummy_hamiltonian=self.first_projection_config.write_dummy_hamiltonian,
                    n_workers=self.first_projection_config.n_workers,
                ),
            )
            first_job = set_projection_job_name(
                first_job,
                structure_name=structure_name,
                reduction_mode=self.first_projection_config.reduction_mode,
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
                    force_2d=self.second_projection_config.force_2d,
                    force_1d=self.second_projection_config.force_1d,
                    reduction_mode=self.second_projection_config.reduction_mode,
                    overlap_only=self.second_projection_config.overlap_only,
                    write_dummy_hamiltonian=self.second_projection_config.write_dummy_hamiltonian,
                    n_workers=self.second_projection_config.n_workers,
                    upstream_projection_output=first_job.output,
                ),
            )
            second_job = set_projection_job_name(
                second_job,
                structure_name=structure_name,
                reduction_mode=self.second_projection_config.reduction_mode,
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
        logger.info(f"{self.name}: {len(structure_names)} structures under {deeph_inputs_root}, projected twice")
        return Flow(
            jobs=[*first_stage_jobs, *second_stage_jobs],
            name=self.name,
            output=outputs,
        )

