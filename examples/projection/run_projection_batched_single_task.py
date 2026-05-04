from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Literal, cast

from hamilflow import ProjectionConfig
from hamilflow.projection.batched import run_projection_batched

DEFAULT_STRUCTURE_PATTERN = "structure_*"
ReductionMode = Literal["schur", "truncate"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one projection task for a single subdirectory using batched k-point accumulation."
    )
    parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Directory containing the structure_* folders.",
    )
    parser.add_argument(
        "--output-directory-parent",
        type=Path,
        required=True,
        help="Parent directory where per-structure projection folders will be created.",
    )
    parser.add_argument(
        "--removal-plan",
        type=Path,
        required=True,
        help="Path to the removal plan JSON file.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="1-based index of the structure_* directory to process. Defaults to SLURM_ARRAY_TASK_ID.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_STRUCTURE_PATTERN,
        help="Glob pattern used to select child directories under --path.",
    )
    parser.add_argument(
        "--reduction-mode",
        choices=("schur", "truncate"),
        default="schur",
        help="Projection mode to use for this task.",
    )
    parser.add_argument(
        "--k-batch-size",
        type=int,
        default=16,
        help="Number of k-points to process per batch.",
    )
    parser.add_argument(
        "--r-batch-size",
        type=int,
        default=32,
        help="Number of R blocks to accumulate per batch.",
    )
    return parser.parse_args()


def select_structure_directory(parent_path: Path, pattern: str, index: int) -> Path:
    structure_directories = sorted(path for path in parent_path.glob(pattern) if path.is_dir())
    if not structure_directories:
        raise FileNotFoundError(
            f"No structure directories matching {pattern} found under: {parent_path}"
        )

    zero_based_index = index - 1
    if zero_based_index < 0 or zero_based_index >= len(structure_directories):
        raise IndexError(
            f"Index {index} is out of range for {len(structure_directories)} structure directories"
        )

    return structure_directories[zero_based_index]


def run_structure_projection(
    structure_directory: Path,
    output_directory_parent: Path,
    removal_plan: Path,
    reduction_mode: ReductionMode,
    k_batch_size: int,
    r_batch_size: int,
) -> Path:
    config = ProjectionConfig(
        input_dir=structure_directory,
        output_dir=output_directory_parent / structure_directory.name,
        kgrid=(9, 9, 1),
        reduction_mode=reduction_mode,
    )
    result = run_projection_batched(
        config=config,
        removal_plan=str(removal_plan),
        k_batch_size=k_batch_size,
        r_batch_size=r_batch_size,
    )
    return result.output_dir


def main() -> None:
    args = parse_args()
    reduction_mode = cast(ReductionMode, args.reduction_mode)

    if not args.path.exists() or not args.path.is_dir():
        raise FileNotFoundError(f"Input path is not a directory: {args.path}")
    if not args.output_directory_parent.exists():
        args.output_directory_parent.mkdir(parents=True, exist_ok=True)
    if not args.removal_plan.exists() or not args.removal_plan.is_file():
        raise FileNotFoundError(f"Removal plan file not found: {args.removal_plan}")

    index = args.index
    if index is None:
        slurm_task = os.environ.get("SLURM_ARRAY_TASK_ID")
        if slurm_task is None:
            raise ValueError("Provide --index or run inside a Slurm array job.")
        try:
            index = int(slurm_task)
        except ValueError as exc:
            raise ValueError(f"Invalid SLURM_ARRAY_TASK_ID value: {slurm_task}") from exc

    structure_directory = select_structure_directory(args.path, args.pattern, index)
    output_dir = run_structure_projection(
        structure_directory=structure_directory,
        output_directory_parent=args.output_directory_parent,
        removal_plan=args.removal_plan,
        reduction_mode=reduction_mode,
        k_batch_size=args.k_batch_size,
        r_batch_size=args.r_batch_size,
    )
    print(f"Completed {structure_directory} -> {output_dir}")


if __name__ == "__main__":
    main()