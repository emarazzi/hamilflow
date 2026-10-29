"""Compatibility layer for workflow makers.

Use ``flows_base`` for reusable building blocks and ``flows_core`` for
higher-level wrappers.
"""

__all__ = [
    "DEFAULT_AIMS_KWARGS",
    "ConvertAimsToDeephConfig",
    "GenerateAimsDFTData",
    "GenerateAimsToProjectedDeephData",
    "GenerateOvlOnlyAimsToProjectedDeephData",
    "GenerateProjectedDeephInputs",
    "GenerateTwoStepProjectedDeephInputs",
    "ProjectDeephInputsConfig",
    "ProjectionRemovalPlanConfig",
    "get_ksampling",
    "resolve_projection_removal_plan",
]

from .flows_base import (
    DEFAULT_AIMS_KWARGS,
    ConvertAimsToDeephConfig,
    GenerateAimsDFTData,
    GenerateProjectedDeephInputs,
    ProjectDeephInputsConfig,
    ProjectionRemovalPlanConfig,
    resolve_projection_removal_plan,
)
from .flows_core import (
    GenerateAimsToProjectedDeephData,
    GenerateOvlOnlyAimsToProjectedDeephData,
    GenerateTwoStepProjectedDeephInputs,
)
from .kpoints import get_ksampling
