from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .models import PipelineConfig, RemovalPlanLike
from .pipeline import run_pipeline

if TYPE_CHECKING:
    from jobflow import Job


def make_run_pipeline_job() -> Any:
    """
    Create a jobflow-decorated job function lazily.

    This keeps jobflow as an optional dependency. Importing this module does not
    require jobflow unless this factory is called.
    """
    try:
        from jobflow import job
    except ImportError as exc:
        raise ImportError(
            "jobflow is not installed. Install jobflow to use make_run_pipeline_job()."
        ) from exc

    @job
    def run_pipeline_job(config: PipelineConfig, removal_plan: RemovalPlanLike) -> dict[str, object]:
        result = run_pipeline(config=config, removal_plan=removal_plan)
        return result.to_dict()

    return run_pipeline_job
