"""Compatibility wrapper for the packaged projection API.

Prefer importing from deepx_postdock directly.
"""

from __future__ import annotations

from pathlib import Path

from deepx_postdock import ProjectionConfig, ProjectionResult, RemovalPlan, RemovalRule, run_projection
from deepx_postdock.removal import coerce_removal_plan


def run_projection_legacy(
    input_dir: Path,
    output_dir: Path,
    nk: tuple[int, int, int],
    removal_plan_json: Path,
    reduction_mode: str = "schur",
) -> ProjectionResult:
    """Thin wrapper over deepx_postdock.run_projection."""
    config = ProjectionConfig(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        kgrid=tuple(nk),
        reduction_mode=reduction_mode,
    )
    return run_projection(config=config, removal_plan=Path(removal_plan_json))


__all__ = [
    "ProjectionConfig",
    "ProjectionResult",
    "RemovalPlan",
    "RemovalRule",
    "coerce_removal_plan",
    "run_projection",
    "run_projection_legacy",
]
