import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from hamilflow import run_projection, ProjectionConfig

DEFAULT_PATH = Path('./dft_original')


def run_structure_projection(
    structure_directory: Path,
    output_directory_parent: Path,
    removal_plan: Path,
) -> str:
    print(f'Processing {structure_directory}')
    config = ProjectionConfig(
        input_dir=structure_directory,
        output_dir=output_directory_parent / f'{structure_directory.name}',
        kgrid=(9, 9, 1),
        reduction_mode='schur',
    )

    run_projection(config=config, removal_plan=str(removal_plan))
    return str(structure_directory)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run projections in parallel.')
    parser.add_argument(
        '--path',
        type=Path,
        required=True,
        help='Directory containing the structure_* folders to process.',
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help='Maximum number of worker processes to use. Defaults to ProcessPoolExecutor behavior.',
    )
    parser.add_argument(
        '--output-directory-parent',
        type=Path,
        default=DEFAULT_PATH,
        help='Parent directory where per-structure projection folders will be created.',
    )
    parser.add_argument(
        '--removal-plan',
        type=Path,
        required=True,
        help='Path to the removal plan JSON file.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.path.exists() or not args.path.is_dir():
        raise FileNotFoundError(f'Input path is not a directory: {args.path}')

    if not args.removal_plan.exists() or not args.removal_plan.is_file():
        raise FileNotFoundError(f'Removal plan file not found: {args.removal_plan}')

    structure_directories = sorted(
        path for path in args.path.glob('structure_*') if path.is_dir()
    )
    if not structure_directories:
        raise FileNotFoundError(
            f'No structure directories matching structure_* found under: {args.path}'
        )

    failures: list[tuple[Path, Exception]] = []

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                run_structure_projection,
                structure_directory,
                args.output_directory_parent,
                args.removal_plan,
            ): structure_directory
            for structure_directory in structure_directories
        }

        for future in as_completed(futures):
            structure_directory = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f'Failed {structure_directory}: {exc}')
                failures.append((structure_directory, exc))

    if failures:
        print(f'Completed with failures: {len(failures)} / {len(structure_directories)} cases failed.')
        raise SystemExit(1)

    print(f'Completed successfully: {len(structure_directories)} / {len(structure_directories)} cases.')


if __name__ == '__main__':
    main()
