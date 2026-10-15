"""
Check that the optimized Schur projection reproduces the original implementation.

Two levels:
  1. Kernel: at a few random k-points, compare the original T(k)^dagger M T(k)
     (build_elimination_tk + apply_tk_projection) with the blocked
     _reduce_one_k used by the streaming code, on the same H(k)/S(k).
  2. End-to-end: extract the reference package from git (default: 02dfdc1,
     the last commit before the streaming/sparse rewrite),
     run the full projection with both versions in separate processes, and
     compare the written hamiltonian.h5 / overlap.h5 / json files.

Usage:
    python compare_schur_versions.py INPUT_DIR REMOVAL_PLAN.json \
        --kgrid 9 9 1 [--ref 02dfdc1] [--n-workers 1] [--overlap-only] [--mode schur]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]

RUNNER = r"""
import sys, json, time, resource
from hamilflow.projection import ProjectionConfig, run_projection
args = json.loads(sys.argv[1])
cfg = ProjectionConfig(
    input_dir=args["input_dir"], output_dir=args["output_dir"], kgrid=tuple(args["kgrid"]),
    reduction_mode=args["mode"], overlap_only=args["overlap_only"],
    write_dummy_hamiltonian=args["overlap_only"], n_workers=args["n_workers"],
)
t = time.time()
run_projection(cfg, args["plan"])
import hamilflow
print(f"{hamilflow.__file__}: {time.time() - t:.1f}s, "
      f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")
"""


def run_version(pythonpath: Path, out_dir: Path, args: argparse.Namespace) -> None:
    payload = json.dumps({
        "input_dir": str(args.input_dir), "output_dir": str(out_dir), "kgrid": args.kgrid,
        "mode": args.mode, "overlap_only": args.overlap_only, "n_workers": args.n_workers,
        "plan": str(args.removal_plan),
    })
    # cwd=out_dir.parent so the checkout under test is not shadowed by the cwd.
    subprocess.run([sys.executable, "-c", RUNNER, payload], check=True,
                   env={**os.environ, "PYTHONPATH": str(pythonpath)}, cwd=out_dir.parent)


def kernel_check(args: argparse.Namespace, n_k: int = 5) -> bool:
    sys.path.insert(0, str(REPO))
    from hamilflow.projection.kspace import _reduce_one_k, apply_tk_projection, build_elimination_tk
    from hamilflow.projection.removal import coerce_removal_plan, resolve_indices_from_rules
    from hamilflow.sparse_hamiltonian import SparseHamiltonianObj

    obj = SparseHamiltonianObj(args.input_dir)
    rm, _ = resolve_indices_from_rules(
        elements=[str(el) for el in obj.elements],
        elements_orbital_map={k: [int(v) for v in vals] for k, vals in obj.elements_orbital_map.items()},
        plan=coerce_removal_plan(args.removal_plan),
    )
    rm = np.array(sorted(set(rm)), dtype=np.int64)
    keep = np.setdiff1d(np.arange(obj._mat_dim), rm)

    ok = True
    rng = np.random.default_rng(0)
    print(f"[kernel] Nb={obj._mat_dim}, removed={len(rm)}, kept={len(keep)}")
    for k in np.vstack([np.zeros(3), rng.random((n_k - 1, 3))]):
        Sk, Hk = obj.Sk_and_Hk(k)
        Tk, _, _ = build_elimination_tk(Sk[None], rm.tolist())
        H_ref, S_ref = (m[0] for m in apply_tk_projection(Hk[None], Sk[None], Tk))
        H_new, S_new = _reduce_one_k(Hk, Sk, keep, rm, args.mode)
        dH = np.abs(H_new - H_ref).max() / np.abs(H_ref).max()
        dS = np.abs(S_new - S_ref).max() / np.abs(S_ref).max()
        ok &= dH < 1e-10 and dS < 1e-10
        print(f"  k={np.round(k, 3)}  rel max|dH|={dH:.1e}  rel max|dS|={dS:.1e}")
    return ok


def compare_outputs(ref: Path, new: Path) -> bool:
    ok = True
    for name in ("hamiltonian.h5", "overlap.h5"):
        if not (ref / name).exists():
            continue
        with h5py.File(ref / name) as fr, h5py.File(new / name) as fn:
            for key in ("atom_pairs", "chunk_shapes", "chunk_boundaries"):
                same = np.array_equal(fr[key][:], fn[key][:])
                ok &= same
                if not same:
                    print(f"  {name}/{key}: LAYOUT DIFFERS")
            er, en = fr["entries"][:], fn["entries"][:]
        if er.shape != en.shape:
            print(f"  {name}: entries shape {er.shape} vs {en.shape}")
            ok = False
            continue
        scale = np.abs(er).max() if er.size else 0.0
        diff = np.abs(er - en).max() if er.size else 0.0
        rel = diff / scale if scale > 0 else diff
        ok &= rel < 1e-10
        print(f"  {name}: {er.size} entries, max|diff|={diff:.2e}, max|ref|={scale:.2e}, rel={rel:.1e}")

    for name in ("info.json", "reduced_basis_meta.json"):
        a, b = json.loads((ref / name).read_text()), json.loads((new / name).read_text())
        # Diagnostic only; now measured on written entries instead of the dense grid.
        a.pop("overlap_imag_max_before_real_cast", None)
        b.pop("overlap_imag_max_before_real_cast", None)
        ok &= a == b
        print(f"  {name}: {'identical' if a == b else 'DIFFERS'}")
    return ok


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_dir", type=Path)
    p.add_argument("removal_plan", type=Path)
    p.add_argument("--kgrid", type=int, nargs=3, default=[6, 6, 1])
    p.add_argument("--ref", default="02dfdc1", help="git revision of the reference implementation")
    p.add_argument("--mode", default="schur", choices=["schur", "truncate"])
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--overlap-only", action="store_true")
    args = p.parse_args()
    args.input_dir = args.input_dir.resolve()
    args.removal_plan = args.removal_plan.resolve()

    kernel_ok = kernel_check(args) if args.mode == "schur" else True

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ref_src = tmp / "ref_src"
        ref_src.mkdir()
        archive = subprocess.run(["git", "-C", str(REPO), "archive", args.ref, "hamilflow"],
                                 check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(ref_src)], input=archive, check=True)

        print(f"[end-to-end] reference = {args.ref}, kgrid = {args.kgrid}")
        run_version(ref_src, tmp / "out_ref", args)
        run_version(REPO, tmp / "out_new", args)
        e2e_ok = compare_outputs(tmp / "out_ref", tmp / "out_new")

    print("PASS" if kernel_ok and e2e_ok else "FAIL")
    sys.exit(0 if kernel_ok and e2e_ok else 1)


if __name__ == "__main__":
    main()
