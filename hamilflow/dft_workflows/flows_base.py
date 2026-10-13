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
from pymatgen.core import Structure
from pymatgen.core.structure import FileFormats
from pymatgen.io.aims.sets.core import StaticSetGenerator

from .aims_makers import OvlOnlyAimsMaker
from ..projection.models import ReductionMode, RemovalPlanLike

from .jobs import (
    build_aims_dft_job,
    collect_aims_output,
    convert_aims_to_deeph_structure,
    resolve_structure_removal_plan,
    run_projection_for_structure,
    set_projection_job_name,
)
from .utils import load_structures

DEFAULT_AIMS_KWARGS: dict[str, Any] = {
    "species_dir": "tight",
    "density_update_method": "density_matrix",
    "output_rs_matrices": "plain",
    "occupation_type": "gaussian 0.01",
    "sc_accuracy_etot": "1.0e-08 eV",
    "compute_forces": "true",
    "preconditioner": "kerker 2.0",
    "sc_accuracy_rho": 1.0e-6,
    "use_local_index": "false",
}


@dataclass(frozen=True, slots=True)
class ConvertAimsToDeephConfig:
    """
    Configuration for the optional AIMS-to-DeepH conversion step.

    ``dummy_hamiltonian=True`` zeroes the converted ``hamiltonian.h5`` entries
    without storing them on disk (shape and layout kept). Meant for overlap-only
    runs, where the Hamiltonian is a placeholder; set automatically by
    ``GenerateOvlOnlyAimsToProjectedDeephData``.
    """

    output_dir: str | Path
    export_rho: bool = False
    export_r: bool = False
    minus_H0: bool = False
    dummy_hamiltonian: bool = False


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
    user_kpoints_settings: dict[str, Any] | Any | None = None
    force_2d: bool = False
    force_1d: bool = False
    reduction_mode: ReductionMode = "schur"
    overlap_only: bool = False
    write_dummy_hamiltonian: bool = False
    n_workers: int | None = None


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

    Every structure is handled independently: the flow contains one sub-flow per
    structure, named after the structure, chaining

        aims_static_<name> -> collect_<name> -> convert_<name> (optional)

    Each job depends only on the previous job of the same structure, so a
    structure is collected and converted as soon as its own AIMS run finishes,
    and a failed structure does not block the others.

    Two modes are supported:

    - Run-and-collect: provide ``aims_maker`` and ``structures_path``. Structures
      are read and named with :func:`load_structures`, AIMS jobs are launched, and
      each produced run directory is moved into ``collected_runs_root / <name>``.
    - Collect-only: provide ``source_run_dirs`` and set ``aims_maker`` to ``None``.
      Each existing run directory is moved into ``collected_runs_root`` under its
      own directory name.

    If ``aims_to_deeph_config`` is provided, each collected run is converted to
    DeepH format with ``PeriodicAimsDataTranslator.transfer_one_aims_to_deeph``.

    ``aims_kwargs`` are applied only when a ``StaticMaker`` is used. K-point
    sampling can be provided either as a direct ``kgrid`` or as richer sampling
    settings through ``kpoints_updates`` or ``user_kpoints_settings``; those are
    resolved into an AIMS k-point payload before job construction. Provide either
    ``kgrid`` or a k-point sampling payload, not both.

    Wrapper flows can extend each structure's chain by calling
    :meth:`structure_entries` and :meth:`make_structure_jobs` directly.
    """

    structures_path: str | Path | None = None
    structure_pattern: str = "*"
    structure_file_format: FileFormats = "poscar"
    name: str = "generate_aims_dft_data"
    aims_kwargs: dict[str, Any] = field(default_factory=dict)
    kgrid: tuple[int, int, int] | None = None
    kpoints_updates: dict[str, Any] | None = None
    user_kpoints_settings: dict[str, Any] | Any | None = None
    force_gamma: bool = True
    force_2d: bool = False
    force_1d: bool = False
    symprec: float = 1e-5
    aims_maker: Maker | None = field(
        default_factory=lambda: StaticMaker(input_set_generator=StaticSetGenerator())
    )
    collected_runs_root: str | Path = "./aims_calculations"
    source_run_dirs: list[str | Path] | None = None
    aims_to_deeph_config: ConvertAimsToDeephConfig | None = None

    def __post_init__(self):
        merged_aims_kwargs = {**DEFAULT_AIMS_KWARGS, **self.aims_kwargs}

        if self.force_2d and self.force_1d:
            raise ValueError("force_2d and force_1d are mutually exclusive.")

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

        if self.kgrid is not None and (self.kpoints_updates or self.user_kpoints_settings not in (None, {})):
            raise ValueError("Provide either kgrid or k-point sampling settings, not both.")
        if self.aims_maker is not None and not isinstance(self.aims_maker, (StaticMaker, OvlOnlyAimsMaker)):
            if self.aims_kwargs or self.kgrid is not None or self.kpoints_updates or self.user_kpoints_settings not in (None, {}):
                raise ValueError(
                    "aims_kwargs and k-point sampling options are only supported with StaticMaker. "
                    "Provide a configured custom maker or use the default StaticMaker path."
                )

        self.aims_kwargs = merged_aims_kwargs

    def structure_entries(self) -> list[tuple[str, Structure | Path]]:
        """
        Return ``(structure_name, source)`` for every structure of the flow.

        ``source`` is a ``Structure`` in run-and-collect mode and an existing run
        directory in collect-only mode.
        """
        if self.aims_maker is not None:
            if self.structures_path is None:
                raise ValueError(
                    "structures_path is required when aims_maker is provided to run new "
                    "AIMS jobs."
                )
            return list(
                load_structures(
                    self.structures_path, self.structure_pattern, self.structure_file_format
                )
            )

        entries: list[tuple[str, Structure | Path]] = [
            (Path(path).name, Path(path)) for path in self.source_run_dirs or []
        ]
        names = [name for name, _ in entries]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate run directory names in source_run_dirs: {duplicates}")
        return entries

    def make_structure_jobs(
        self, structure_name: str, source: Structure | Path
    ) -> tuple[list[Job], dict[str, Any]]:
        """
        Build the AIMS -> collect -> convert chain of a single structure.

        Returns the jobs and a dict of their outputs with keys ``aims`` (``None`` in
        collect-only mode), ``collection`` and ``deeph`` (``None`` without
        ``aims_to_deeph_config``).
        """
        jobs: list[Job] = []
        aims_output = None

        if isinstance(source, Structure):
            if self.aims_maker is None:
                raise ValueError("aims_maker is required to run AIMS on a Structure.")
            aims_job = build_aims_dft_job(
                source,
                structure_name,
                self.aims_maker,
                aims_kwargs=self.aims_kwargs,
                kgrid=self.kgrid,
                kpoints_updates=self.kpoints_updates,
                user_kpoints_settings=self.user_kpoints_settings,
                force_gamma=self.force_gamma,
                force_2d=self.force_2d,
                force_1d=self.force_1d,
                symprec=self.symprec,
            )
            jobs.append(aims_job)
            aims_output = aims_job.output
            source_run_dir = cast(str, aims_job.output.dir_name)
        else:
            source_run_dir = str(source)

        collect_job = cast(
            Job,
            collect_aims_output(
                source_run_dir=source_run_dir,
                structure_name=structure_name,
                collected_runs_root=self.collected_runs_root,
            ),
        )
        collect_job.name = f"collect_{structure_name}"
        jobs.append(collect_job)

        convert_output = None
        if self.aims_to_deeph_config is not None:
            convert_job = cast(
                Job,
                convert_aims_to_deeph_structure(
                    structure_name=structure_name,
                    input_root=self.collected_runs_root,
                    output_dir=self.aims_to_deeph_config.output_dir,
                    export_rho=self.aims_to_deeph_config.export_rho,
                    export_r=self.aims_to_deeph_config.export_r,
                    minus_H0=self.aims_to_deeph_config.minus_H0,
                    dummy_hamiltonian=self.aims_to_deeph_config.dummy_hamiltonian,
                    collection_output=collect_job.output,
                ),
            )
            convert_job.name = f"convert_{structure_name}"
            jobs.append(convert_job)
            convert_output = convert_job.output

        outputs = {
            "aims": aims_output,
            "collection": collect_job.output,
            "deeph": convert_output,
        }
        return jobs, outputs

    def flow_output(self, by_structure: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Assemble the flow-level output from the per-structure outputs."""
        return {
            "aims_kwargs": self.aims_kwargs,
            "collected_runs_root": str(Path(self.collected_runs_root).resolve()),
            "deeph_inputs_root": (
                str(Path(self.aims_to_deeph_config.output_dir).resolve())
                if self.aims_to_deeph_config is not None
                else None
            ),
            "structure_names": list(by_structure),
            "by_structure": by_structure,
        }

    def make(self) -> Flow:
        structure_flows: list[Flow] = []
        by_structure: dict[str, dict[str, Any]] = {}
        for structure_name, source in self.structure_entries():
            jobs, outputs = self.make_structure_jobs(structure_name, source)
            structure_flows.append(Flow(jobs=jobs, name=structure_name, output=outputs))
            by_structure[structure_name] = outputs

        return Flow(
            jobs=structure_flows, name=self.name, output=self.flow_output(by_structure)
        )


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
                user_kpoints_settings=self.projection_config.user_kpoints_settings,
                force_2d=self.projection_config.force_2d,
                force_1d=self.projection_config.force_1d,
                reduction_mode=self.projection_config.reduction_mode,
                overlap_only=self.projection_config.overlap_only,
                write_dummy_hamiltonian=self.projection_config.write_dummy_hamiltonian,
                n_workers=self.projection_config.n_workers,
            )
            projection_job = set_projection_job_name(
                projection_job,
                structure_name=structure_name,
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
