"""Diagnose suspiciously large band-uncertainty values.

For a few structures (default: the highest weighted sigma, plus the lowest two
as controls) this re-diagonalizes every ensemble member, the stored averaged
Hamiltonian and a freshly averaged one, and reports:

  [files]     model dirs found, whether any two members' hamiltonian.h5 are
              identical, info.json occupation per member
  [avg]       max |stored avg - fresh mean of members| over matrix entries
              (detects a stale avg_ham/<structure>/hamiltonian.h5)
  [shifts]    per-member midgap-at-Gamma shift: recomputed vs stored in the JSON
  [sigma]     weighted max of sigma with three references:
                - stored average Hamiltonian (what the calculator reports)
                - fresh average Hamiltonian
                - plain std of the member eigenvalues (no Hamiltonian average)
  [count]     number of eigenvalues below the averaged Hamiltonian's midgap,
              per member and k-point, compared with the average's own
              count (a difference means a missing/spurious state)
  [offset]    per member, RMS deviation from the average over the edge window
              for band-index offsets -2..2, after removing the mean (rigid)
              difference (best offset != 0 -> index shift)
  [argmax]    member/average eigenvalues around the worst state
  [overlap]   smallest eigenvalue of S(k) at Gamma and at the worst k-point

Run from the directory that contains train_*/ avg_ham/ uncertainty_train.json:

  python diagnose_band_uncertainty.py
  python diagnose_band_uncertainty.py --structures structure_356 structure_4554
  python diagnose_band_uncertainty.py --dft-root /path/to/dft --structures structure_4234_300
  python diagnose_band_uncertainty.py --structures structure_4234_300 --ill-method orbital_removal
  python diagnose_band_uncertainty.py --all --n-procs 16    # every structure, one CPU each

With --n-procs N > 1, up to N structures are diagnosed at the same time in separate
processes, each limited to one CPU (BLAS threads included, --n-jobs is ignored); each
structure's report is printed in one piece when it finishes.

Alignment and ill-conditioning handling default to the settings stored in the JSON
(Gamma alignment and no handling for files written before those settings existed).
"""

import argparse
import concurrent.futures
import contextlib
import hashlib
import io
import json
import tempfile
import traceback
from glob import glob
from pathlib import Path

import numpy as np

from hamilflow.sparse_hamiltonian import SparseHamiltonianObj
from hamilflow.uncertainty import (
    BandUncertaintyCalculator,
    BandUncertaintyPostProcessor,
    average_predicted_hamiltonians,
    read_deeph_hamiltonian,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-glob", default="train_{i}/infer/outputs_train/2026-10-06_09-41-0?/dft/",
                   help="glob for the member output dirs, '{i}' replaced by 1..n-models")
    p.add_argument("--n-models", type=int, default=5)
    p.add_argument("--avg-dir", default="avg_ham")
    p.add_argument("--uncertainty-json", default="uncertainty_train.json")
    p.add_argument("--dft-root", default=None, help="optional DFT reference root, one dir per structure")
    p.add_argument("--structures", nargs="*", default=None)
    p.add_argument("--all", action="store_true", help="diagnose every structure in the uncertainty JSON")
    p.add_argument("--n-top", type=int, default=3, help="highest-sigma structures picked if --structures is not given")
    p.add_argument("--n-bottom", type=int, default=2, help="lowest-sigma structures added as controls")
    p.add_argument("--e-cut", type=float, default=2.0)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--species", default='{"Mo": 42, "S": 16}', help="JSON element -> atomic number map")
    p.add_argument("--align-mode", choices=["grid", "gamma"], default=None,
                   help="default: as stored in the JSON ('gamma' for files written before align_mode existed)")
    p.add_argument("--ill-method", choices=["none", "orbital_removal"], default=None,
                   help="default: as stored in the JSON (none for older files)")
    p.add_argument("--ill-threshold", type=float, default=None, help="default: as stored in the JSON, else 1e-3")
    p.add_argument("--n-jobs", type=int, default=-1, help="CPU budget per structure when --n-procs is 1")
    p.add_argument("--n-procs", type=int, default=1, help="structures diagnosed in parallel, one CPU each")
    p.add_argument("--no-hash", action="store_true", help="skip hashing hamiltonian.h5 files")
    p.add_argument("--out", default="diagnose_band_uncertainty.json")
    return p.parse_args()


def md5(path: Path, block: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def weighted_max(sigma, d, e_cut, tau):
    """sigma, d: (n_bands, n_k). Returns (value, band, k) of max(w(d) * sigma)."""
    w = BandUncertaintyPostProcessor.soft_window(d, e_cut, tau)
    ws = w * sigma
    n, k = np.unravel_index(np.argmax(ws), ws.shape)
    return float(ws[n, k]), int(n), int(k)


def edge_distance(eig, vbm):
    """eig: (n_bands, n_k) aligned eigenvalues of the reference."""
    vbm_e, cbm_e = eig[vbm].max(), eig[vbm + 1].min()
    valence = (np.arange(eig.shape[0]) <= vbm)[:, None]
    return np.where(valence, vbm_e - eig, eig - cbm_e)


def offset_rms(member, ref, window_bands, offsets=range(-2, 3)):
    """RMS of member[n + o] - ref[n] over window bands and all k, for each offset o, after
    removing the mean difference (so a rigid misalignment does not hide an index shift)."""
    out = {}
    nb = member.shape[0]
    for o in offsets:
        idx = [n for n in window_bands if 0 <= n + o < nb]
        if not idx:
            continue
        diff = member[np.array(idx) + o] - ref[np.array(idx)]
        out[o] = float(np.sqrt(np.mean((diff - diff.mean()) ** 2)))
    return out


def pick_structures(args, results):
    if args.structures:
        return args.structures
    if args.all:
        return list(results)
    pp = BandUncertaintyPostProcessor(results)
    ranked = sorted(results, key=lambda s: pp.weighted_stats(s, args.e_cut, args.tau)["weighted_max_eV"])
    picked = ranked[::-1][: args.n_top] + ranked[: args.n_bottom]
    return list(dict.fromkeys(picked))


def diagnose(structure, model_dirs, args, calc, results):
    rep = {"structure": structure}
    res = results.get(structure)
    print(f"\n{'=' * 100}\n{structure}\n{'=' * 100}")

    # ---------------------------------------------------------------- files
    model_paths = [d / structure / calc.hamiltonian_name for d in model_dirs]
    occs = []
    for d in model_dirs:
        with open(d / structure / "info.json") as f:
            occs.append(json.load(f).get("occupation"))
    print(f"[files] info.json occupation per member: {occs}")
    rep["occupation_per_member"] = occs
    if not args.no_hash:
        hashes = [md5(p) for p in model_paths]
        dup = [(i, j) for i in range(len(hashes)) for j in range(i + 1, len(hashes)) if hashes[i] == hashes[j]]
        print(f"[files] identical hamiltonian.h5 between members: {dup if dup else 'none'}")
        rep["identical_member_pairs"] = dup

    # ---------------------------------------------------------------- average
    avg_path = Path(args.avg_dir) / structure / calc.hamiltonian_name
    tmp = tempfile.TemporaryDirectory()
    fresh_path = Path(tmp.name) / structure / calc.hamiltonian_name
    average_predicted_hamiltonians(model_paths, fresh_path)
    if avg_path.exists():
        *_, stored = read_deeph_hamiltonian(avg_path)
        *_, fresh = read_deeph_hamiltonian(fresh_path)
        diff = float(np.max(np.abs(stored - fresh))) if stored.shape == fresh.shape else float("nan")
        print(f"[avg] stored avg: {avg_path}  max|stored - fresh| = {diff:.3e}"
              + ("  <-- STALE AVERAGE" if not diff < 1e-8 else ""))
        rep["avg_stored_vs_fresh_max_abs"] = diff
    else:
        print(f"[avg] {avg_path} missing, using fresh average only")
        avg_path = fresh_path

    # ---------------------------------------------------------------- spectra
    ref_obj = SparseHamiltonianObj(model_dirs[0] / structure)
    mesh = tuple(res["grid_mesh"]) if res else calc.grid_mesh
    ks, weights, a = calc.build_irreducible_kpoints(ref_obj, mesh, calc.symprec)
    vbm, _ = calc.homo_lumo_indices(ref_obj)
    print(f"[k] mesh {mesh}, {len(ks)} irreducible k-points, anchor index {a}, N_occ (bands) = {vbm + 1}")
    ill_handler = calc.make_ill_handler(ref_obj, ks, n_jobs=args.n_jobs)
    print(f"[settings] {calc.settings()}, kept orbitals: {calc.n_kept_orbitals(ill_handler)}")

    def raw_spectrum(data_dir, H_file_path=None):
        obj = SparseHamiltonianObj(data_dir, H_file_path=H_file_path)
        return obj.diag(ks, bands_only=True, n_jobs=args.n_jobs, ill_handler=ill_handler)

    spectra = [raw_spectrum(d / structure) for d in model_dirs]
    spectra += [raw_spectrum(model_dirs[0] / structure, avg_path), raw_spectrum(model_dirs[0] / structure, fresh_path)]
    if args.dft_root:
        spectra.append(raw_spectrum(Path(args.dft_root) / structure))
    tmp.cleanup()
    spectra = calc.common_bands(*spectra)
    dft_raw = spectra.pop() if args.dft_root else None
    *member_raws, avg_raw, fresh_raw = spectra

    members = [(raw, *calc.align_to_midgap(raw, ref_obj, a)) for raw in member_raws]
    avg_al, avg_shift = calc.align_to_midgap(avg_raw, ref_obj, a)
    fresh_al, _ = calc.align_to_midgap(fresh_raw, ref_obj, a)
    raws = np.stack([m[0] for m in members])
    aligned = np.stack([m[1] for m in members])
    shifts = [float(m[2]) for m in members]
    print(f"[gap] on the k-grid: members {np.round([calc.gap(r, ref_obj) for r in raws], 4).tolist()}, "
          f"avg H {calc.gap(avg_raw, ref_obj):.4f}"
          + (f", DFT {calc.gap(dft_raw, ref_obj):.4f}" if dft_raw is not None else ""))

    # ---------------------------------------------------------------- shifts
    stored_shifts = res.get("per_model_shift_eV") if res else None
    print(f"[shifts] recomputed per member : {np.round(shifts, 4).tolist()}   avg H: {avg_shift:.4f}")
    if stored_shifts is not None:
        print(f"[shifts] stored in JSON        : {np.round(stored_shifts, 4).tolist()}   "
              f"avg H: {res.get('avg_hamiltonian_shift_eV', float('nan')):.4f}")
    print(f"[shifts] member - median       : {np.round(np.array(shifts) - np.median(shifts), 4).tolist()}")
    rep["shifts"] = shifts
    rep["avg_shift"] = float(avg_shift)

    # ---------------------------------------------------------------- sigma variants
    d = edge_distance(avg_al, vbm)
    variants = {
        "stored_avg_H": np.sqrt(np.mean((aligned - avg_al[None]) ** 2, axis=0)),
        "fresh_avg_H": np.sqrt(np.mean((aligned - fresh_al[None]) ** 2, axis=0)),
        "member_std": np.std(aligned, axis=0),
    }
    rep["weighted_max"] = {}
    for name, sig in variants.items():
        val, n_, k_ = weighted_max(sig, d, args.e_cut, args.tau)
        rep["weighted_max"][name] = {"value": val, "band": n_, "k": k_}
        print(f"[sigma] {name:13s} weighted max = {val:.4f} eV at band {n_} (vbm {vbm}), k {k_}, d = {d[n_, k_]:.3f} eV")
    sig = variants["stored_avg_H"]
    _, n_star, k_star = weighted_max(sig, d, args.e_cut, args.tau)

    # ---------------------------------------------------------------- state counting
    midgap_raw = -avg_shift
    counts = (raws < midgap_raw).sum(axis=1)  # (n_models, n_k)
    avg_counts = (avg_raw < midgap_raw).sum(axis=0)
    print(f"[count] eigenvalues below avg-H midgap, per k (N_occ = {vbm + 1}):")
    print(f"        avg H    : {avg_counts.tolist()}")
    for i, c in enumerate(counts):
        flag = "" if np.array_equal(c, avg_counts) else "  <-- DIFFERS FROM AVG"
        print(f"        member {i + 1}: {c.tolist()}{flag}")
    rep["count_below_midgap"] = {"avg": avg_counts.tolist(), "members": counts.tolist()}

    # ---------------------------------------------------------------- index offset
    window = sorted(set(np.where(d <= args.e_cut)[0].tolist()))
    print(f"[offset] RMS(member[n+o] - avg[n] - mean) over {len(window)} edge-window bands, eV:")
    rep["offset_rms"] = []
    for i in range(len(members)):
        o = offset_rms(aligned[i], avg_al, window)
        best = min(o, key=o.get)
        rep["offset_rms"].append({str(k): v for k, v in o.items()})
        flag = f"  <-- best offset {best:+d}" if best != 0 else ""
        print(f"        member {i + 1}: " + "  ".join(f"{k:+d}:{v:.4f}" for k, v in o.items()) + flag)

    # ---------------------------------------------------------------- around argmax
    lo, hi = max(0, n_star - 3), min(avg_al.shape[0], n_star + 4)
    print(f"[argmax] aligned eigenvalues, bands {lo}..{hi - 1} at k {k_star} {np.round(ks[k_star], 3).tolist()} "
          f"(* = worst state, | = vbm/cbm boundary)")
    header = "".join(f"{('*' if n == n_star else '') + str(n) + ('|' if n == vbm else ''):>10s}" for n in range(lo, hi))
    print(f"        {'':10s}{header}")
    print(f"        {'avg H':10s}" + "".join(f"{avg_al[n, k_star]:10.4f}" for n in range(lo, hi)))
    for i in range(len(members)):
        print(f"        {'member ' + str(i + 1):10s}" + "".join(f"{aligned[i, n, k_star]:10.4f}" for n in range(lo, hi)))
    dev = aligned[:, n_star, k_star] - avg_al[n_star, k_star]
    share = dev ** 2 / np.sum(dev ** 2)
    print(f"        deviation at worst state per member: {np.round(dev, 4).tolist()}  "
          f"(share of sigma^2: {np.round(share, 2).tolist()})")
    rep["argmax_deviation_per_member"] = dev.tolist()

    # ---------------------------------------------------------------- overlap conditioning
    s_min = {}
    for label, kk in (("Gamma", a), ("argmax_k", k_star)):
        s_min[label] = float(np.linalg.eigvalsh(ref_obj.get_Sk(ks[kk]))[0])
    print(f"[overlap] min eigenvalue of S(k): " + ", ".join(f"{k} {v:.3e}" for k, v in s_min.items()))
    rep["overlap_min_eig"] = s_min

    # ---------------------------------------------------------------- DFT
    if args.dft_root:
        dft_al, dft_shift = calc.align_to_midgap(dft_raw, ref_obj, a)
        err = np.abs(avg_al - dft_al)
        val, n_e, k_e = weighted_max(err, d, args.e_cut, args.tau)
        o = offset_rms(avg_al, dft_al, window)
        print(f"[dft] weighted max |avg - DFT| = {val:.4f} eV at band {n_e}, k {k_e}; "
              f"shift avg {avg_shift:.4f} vs DFT {float(dft_shift):.4f}")
        print(f"[dft] RMS(avg[n+o] - DFT[n] - mean): " + "  ".join(f"{k:+d}:{v:.4f}" for k, v in o.items()))
        print(f"[dft] eigenvalues below own midgap per k: DFT {(dft_raw < -dft_shift).sum(axis=0).tolist()}"
              f"  avg H {avg_counts.tolist()}")
        rep["dft"] = {"weighted_max_err": val, "band": n_e, "k": k_e, "offset_rms": {str(k): v for k, v in o.items()}}

    return rep


_BLAS_LIMIT = None


def _init_worker():
    # One CPU per worker process: also caps BLAS threads outside SparseHamiltonianObj.diag
    # (averaging, overlap eigenvalues, orbital truncation).
    global _BLAS_LIMIT
    import threadpoolctl

    _BLAS_LIMIT = threadpoolctl.threadpool_limits(limits=1)


def _diagnose_captured(structure, model_dirs, args, calc, results):
    """Run `diagnose` in a worker, returning its printed report as one string so that the
    reports of structures running in parallel are not interleaved."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rep = diagnose(structure, model_dirs, args, calc, results)
    except Exception:
        rep = {"structure": structure, "error": traceback.format_exc()}
        buf.write(f"\n{'=' * 100}\n{structure}: FAILED\n{rep['error']}")
    return buf.getvalue(), rep


def main():
    args = parse_args()
    model_dirs = []
    for i in range(1, args.n_models + 1):
        model_dirs.extend(Path(s) for s in sorted(glob(args.model_glob.format(i=i))))
    print("model dirs (member numbers used below):")
    for i, d in enumerate(model_dirs, 1):
        print(f"  member {i}: {d}")
    if len(model_dirs) != args.n_models:
        print(f"WARNING: expected {args.n_models} model dirs, found {len(model_dirs)}")

    with open(args.uncertainty_json) as f:
        results = json.load(f)

    stored = next(iter(results.values()), {})
    calc = BandUncertaintyCalculator(
        species_number=json.loads(args.species),
        align_mode=args.align_mode or stored.get("align_mode") or "gamma",
        ill_method=args.ill_method or stored.get("ill_method") or "none",
        ill_threshold=args.ill_threshold or stored.get("ill_threshold") or 1e-3,
    )
    structures = pick_structures(args, results)
    print("structures:", structures)

    if args.n_procs <= 1:
        report = [diagnose(s, model_dirs, args, calc, results) for s in structures]
    else:
        args.n_jobs = 1
        n_procs = min(args.n_procs, len(structures))
        print(f"diagnosing {len(structures)} structures with {n_procs} processes, one CPU each", flush=True)
        reports = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_procs, initializer=_init_worker) as ex:
            futures = [
                ex.submit(_diagnose_captured, s, model_dirs, args, calc, {s: results.get(s)})
                for s in structures
            ]
            for fut in concurrent.futures.as_completed(futures):
                text, rep = fut.result()
                print(text, flush=True)
                reports[rep["structure"]] = rep
        report = [reports[s] for s in structures]
        failed = [r["structure"] for r in report if "error" in r]
        if failed:
            print(f"\nFAILED: {failed}")
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
