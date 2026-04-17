"""Compatibility wrapper for the packaged pipeline API.

Prefer importing from deepx_postdock directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deepx_postdock import PipelineConfig, PipelineResult, RemovalPlan, RemovalRule, run_pipeline
from deepx_postdock.removal import coerce_removal_plan


def run_pipeline_legacy(
    input_dir: Path,
    output_dir: Path,
    nk: tuple[int, int, int],
    removal_plan_json: Path,
    reduction_mode: str = "schur",
) -> PipelineResult:
    """Backward-compatible wrapper over deepx_postdock.run_pipeline."""
    config = PipelineConfig(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        kgrid=tuple(nk),
        reduction_mode=reduction_mode,
    )
    return run_pipeline(config=config, removal_plan=Path(removal_plan_json))


__all__ = [
    "PipelineConfig",
    "PipelineResult",
    "RemovalPlan",
    "RemovalRule",
    "coerce_removal_plan",
    "run_pipeline",
    "run_pipeline_legacy",
]
