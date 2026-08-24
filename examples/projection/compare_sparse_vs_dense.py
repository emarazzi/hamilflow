"""
Compare the new SparseHamiltonianObj projection pipeline against the
original dense deepx_dock.HamiltonianObj pipeline on a real DeepH dataset.

Checks the thing that matters: same numerical result, less peak memory,
comparable time. Each implementation is run in its own subprocess (via
`/usr/bin/time -v`) so peak RSS is measured independently -- reading
resource.getrusage() from a single long-lived process would contaminate the
second measurement with the first implementation's already-freed memory.

Usage:
    python compare_sparse_vs_dense.py \\
        --data-dir /path/to/deeph_structure_dir \\
        --kgrid 4 4 2 \\
        --removal-plan /path/to/removal_plan.json \\
        --reduction-mode schur

    # or, without a removal plan file, remove raw global orbital indices directly:
    python compare_sparse_vs_dense.py --data-dir ... --remove 0 1 2 3 4

Requires the `hamilflow` env's Python (deepx_dock installed) and GNU time
(`/usr/bin/time -v`) for the memory measurement.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_DATA_DIR = "/home/emarazzi/Desktop/work/DeepH/MoS2/aims/relaxed/bilayer/red"

MAXRSS_RE = re.compile(r"Maximum resident set size \(kbytes\): (\d+)")


def resolve_removal_indices(obj, removal_plan: Path | None, remove: list[int] | None) -> list[int]:
    """Resolve --removal-plan (a RemovalPlan-like JSON file) or a raw --remove index list into global orbital indices."""
    if removal_plan is not None:
        from hamilflow.projection.removal import coerce_removal_plan, resolve_indices_from_rules

        plan_model = coerce_removal_plan(removal_plan)
        rm, _ = resolve_indices_from_rules(
            elements=[str(el) for el in obj.elements],
            elements_orbital_map={k: [int(v) for v in vals] for k, vals in obj.elements_orbital_map.items()},
            plan=plan_model,
        )
        return sorted(set(rm))
    return sorted(set(remove or []))


def run_dense(
    data_dir: Path,
    kgrid: tuple[int, int, int],
    removal_plan: Path | None,
    remove: list[int] | None,
    out_path: Path,
) -> None:
    from deepx_dock.compute.eigen.hamiltonian import HamiltonianObj

    from hamilflow.projection.kspace import apply_custom_kspace_transform, build_uniform_kmesh, hk_and_sk_to_real

    t0 = time.perf_counter()
    obj = HamiltonianObj(data_dir)
    rm = resolve_removal_indices(obj, removal_plan, remove)
    ks = build_uniform_kmesh(kgrid)
    Sk, Hk = obj.Sk_and_Hk(ks)  # materializes the full (Nk, Nb, Nb) stack -- the OOM-prone step
    Hk_new, Sk_new = apply_custom_kspace_transform(Hk, Sk, rm)
    HR, SR = hk_and_sk_to_real(ks, Hk_new, Sk_new, obj.Rijk_list)
    elapsed = time.perf_counter() - t0
    np.savez(out_path, HR=HR, SR=SR)
    print(
        json.dumps(
            {
                "elapsed_s": elapsed,
                "nb": int(obj.orbits_quantity),
                "r_quantity": len(obj.Rijk_list),
                "n_removed": len(rm),
            }
        )
    )


def run_sparse(
    data_dir: Path,
    kgrid: tuple[int, int, int],
    removal_plan: Path | None,
    remove: list[int] | None,
    out_path: Path,
    n_workers: int,
) -> None:
    from hamilflow.projection.kspace import build_uniform_kmesh, stream_project_to_real_space
    from hamilflow.sparse_hamiltonian import SparseHamiltonianObj

    t0 = time.perf_counter()
    obj = SparseHamiltonianObj(data_dir)
    rm = resolve_removal_indices(obj, removal_plan, remove)
    ks = build_uniform_kmesh(kgrid)
    HR, SR = stream_project_to_real_space(
        obj,
        ks,
        remove_indices=rm,
        reduction_mode="schur",
        Rijk_list=obj.Rijk_list,
        n_workers=n_workers,
        overlap_only=False,
    )
    elapsed = time.perf_counter() - t0
    np.savez(out_path, HR=HR, SR=SR)
    print(
        json.dumps(
            {
                "elapsed_s": elapsed,
                "nb": int(obj.orbits_quantity),
                "r_quantity": len(obj.Rijk_list),
                "n_removed": len(rm),
            }
        )
    )


def launch_worker(impl: str, args: argparse.Namespace, out_path: Path) -> dict:
    cmd = [
        "/usr/bin/time",
        "-v",
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        impl,
        "--data-dir",
        str(args.data_dir),
        "--kgrid",
        *[str(x) for x in args.kgrid],
        "--n-workers",
        str(args.n_workers),
        "--out",
        str(out_path),
    ]
    if args.removal_plan is not None:
        cmd += ["--removal-plan", str(args.removal_plan)]
    else:
        cmd += ["--remove", *[str(x) for x in args.remove]]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{impl} worker failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

    stdout_line = next(line for line in proc.stdout.splitlines() if line.strip().startswith("{"))
    info = json.loads(stdout_line)

    m = MAXRSS_RE.search(proc.stderr)
    if m is None:
        raise RuntimeError(f"Could not parse peak RSS from `time -v` output:\n{proc.stderr}")
    info["peak_rss_mb"] = int(m.group(1)) / 1024.0
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path(DEFAULT_DATA_DIR))
    parser.add_argument("--kgrid", type=int, nargs=3, default=(4, 4, 2))
    removal_group = parser.add_mutually_exclusive_group()
    removal_group.add_argument(
        "--removal-plan",
        type=Path,
        default=None,
        help="Path to a removal plan JSON file (same format accepted by run_projection).",
    )
    removal_group.add_argument(
        "--remove",
        type=int,
        nargs="+",
        default=None,
        help="Raw global orbital indices to remove, used only if --removal-plan is not given.",
    )
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--reduction-mode", choices=("schur",), default="schur")
    # Internal flags used for the subprocess re-invocation; not for end users.
    parser.add_argument("--worker", choices=("dense", "sparse"), default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.removal_plan is None and args.remove is None:
        args.remove = [0, 1, 2, 3, 4]

    if args.worker is not None:
        # Re-invoked as a worker subprocess: do exactly one implementation, then exit.
        if args.worker == "dense":
            run_dense(args.data_dir, tuple(args.kgrid), args.removal_plan, args.remove, args.out)
        else:
            run_sparse(args.data_dir, tuple(args.kgrid), args.removal_plan, args.remove, args.out, args.n_workers)
        return

    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"--data-dir not found: {args.data_dir}")

    dense_out = Path("/tmp/_compare_dense_result.npz")
    sparse_out = Path("/tmp/_compare_sparse_result.npz")

    print(f"Dataset: {args.data_dir}")
    if args.removal_plan is not None:
        print(f"k-grid: {tuple(args.kgrid)}, removal plan: {args.removal_plan}, reduction_mode: schur\n")
    else:
        print(f"k-grid: {tuple(args.kgrid)}, removed orbitals: {args.remove}, reduction_mode: schur\n")

    print("Running dense (original deepx_dock.HamiltonianObj) pipeline...")
    dense_info = launch_worker("dense", args, dense_out)

    print("Running sparse (SparseHamiltonianObj, streaming) pipeline...")
    sparse_info = launch_worker("sparse", args, sparse_out)

    dense_result = np.load(dense_out)
    sparse_result = np.load(sparse_out)
    hr_match = np.allclose(dense_result["HR"], sparse_result["HR"], atol=1e-8)
    sr_match = np.allclose(dense_result["SR"], sparse_result["SR"], atol=1e-8)
    max_diff = max(
        np.max(np.abs(dense_result["HR"] - sparse_result["HR"])),
        np.max(np.abs(dense_result["SR"] - sparse_result["SR"])),
    )
    dense_out.unlink(missing_ok=True)
    sparse_out.unlink(missing_ok=True)

    print(f"\nNb={dense_info['nb']}, R_quantity={dense_info['r_quantity']}, n_removed={dense_info['n_removed']}")
    if dense_info["n_removed"] != sparse_info["n_removed"]:
        raise RuntimeError(
            f"Removal plan resolved to different orbital counts between implementations: "
            f"dense={dense_info['n_removed']} vs sparse={sparse_info['n_removed']}"
        )
    print(f"{'':16s}{'dense (old)':>16s}{'sparse (new)':>16s}{'change':>12s}")
    print(
        f"{'peak RSS (MB)':16s}"
        f"{dense_info['peak_rss_mb']:16.1f}"
        f"{sparse_info['peak_rss_mb']:16.1f}"
        f"{sparse_info['peak_rss_mb'] / dense_info['peak_rss_mb']:11.2f}x"
    )
    print(
        f"{'time (s)':16s}"
        f"{dense_info['elapsed_s']:16.3f}"
        f"{sparse_info['elapsed_s']:16.3f}"
        f"{sparse_info['elapsed_s'] / dense_info['elapsed_s']:11.2f}x"
    )
    print(f"\nHR match: {hr_match}, SR match: {sr_match}, max abs diff: {max_diff:.3e}")
    print("RESULT: " + ("PASS -- identical result, see memory/time above" if hr_match and sr_match else "FAIL"))


if __name__ == "__main__":
    main()
