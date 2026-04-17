from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias


ReductionMode: TypeAlias = Literal["schur", "truncate"]


@dataclass(frozen=True)
class RemovalRule:
    """Single rule used to select orbitals to remove."""

    target_elements: list[str] = field(default_factory=list)
    target_atom_indices: list[int] = field(default_factory=list)
    remove_orbitals: list[str] = field(default_factory=list)
    remove_shells: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RemovalPlan:
    """Normalized removal plan model used internally by the pipeline."""

    rules: list[RemovalRule] = field(default_factory=list)


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration for the k->R reduction pipeline."""

    input_dir: Path
    output_dir: Path
    kgrid: tuple[int, int, int] = (4, 4, 4)
    reduction_mode: ReductionMode = "schur"

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_dir", Path(self.input_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))


@dataclass(frozen=True)
class PipelineResult:
    """Serializable pipeline outputs for scripts and workflow engines."""

    output_dir: Path
    hamiltonian_path: Path
    overlap_path: Path
    info_path: Path
    meta_path: Path
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "hamiltonian_path": str(self.hamiltonian_path),
            "overlap_path": str(self.overlap_path),
            "info_path": str(self.info_path),
            "meta_path": str(self.meta_path),
            "metadata": self.metadata,
        }


RemovalPlanLike: TypeAlias = RemovalPlan | Path | str | dict[str, Any] | list[dict[str, Any]]
