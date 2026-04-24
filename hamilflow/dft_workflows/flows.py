from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from atomate2.aims.jobs.core import StaticMaker
from jobflow.core.flow import Flow
from jobflow.core.job import Job
from jobflow.core.maker import Maker
from pymatgen.io.aims.sets.core import StaticSetGenerator
from pymatgen.core.structure import FileFormats

from .jobs import build_aims_dft_jobs, collect_aims_outputs, convert_aims_to_deeph
from .utils import resolve_structure_path

DEFAULT_AIMS_KWARGS: dict[str, Any] = {"output_rs_matrices": "plain"}


@dataclass(frozen=True, slots=True)
class ConvertAimsToDeephConfig:
    """Configuration for the optional AIMS-to-DeepH conversion step."""

    output_dirs: str | Path
    jobs_num: int = 1
    tier_num: int = 0


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

        if self.structure_file_format and self.structure_file_format not in FileFormats:
            raise ValueError(
                f"Unsupported structure_file_format: {self.structure_file_format}. "
                f"Supported formats are: {[fmt for fmt in FileFormats]}"
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
                output_dirs=self.aims_to_deeph_config.output_dirs,
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
