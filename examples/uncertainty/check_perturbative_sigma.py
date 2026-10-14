"""Test whether first-order perturbation theory reproduces the ensemble band uncertainty.

For every structure, on the same irreducible k-grid and with the same overlap handling and
rigid-shift alignment as `BandUncertaintyCalculator(align_mode="matrix")`:

- exact:  diagonalize every member, sigma_n(k) = RMS_m (eps_n^m - c_m - eps_avg_n)
          (M + 1 diagonalizations, = `sigma_eV_avg_ham` of the band uncertainty);
- perturbative: diagonalize only the averaged Hamiltonian, keep its eigenvectors psi (S-normalized)
  for the bands of a window around the gap, and with dH_m = H_m - c_m S - H_avg, X_m = psi^dag dH_m psi:
    `pt1`      eps_avg_n + X_m[n, n]                       (plain first order)
    `pt1_deg`  eigenvalues of diag(eps_avg) + X_m within groups closer than `--deg-tol`
               (degenerate / quasi-degenerate first order)
    `ritz`     eigenvalues of diag(eps_avg) + X_m over the whole window (Rayleigh-Ritz in the
               subspace of the window states; adds the couplings inside the window)
  The perturbed levels of every member are sorted before comparing with eps_avg, as the exact
  eigenvalues are.

Per structure it reports the band score (`weighted_max_eV`/`weighted_mean_eV` of
`BandUncertaintyPostProcessor.weighted_stats`, `--e-cut`, `--tau`) for each variant, the per-state
agreement for the states inside the window, and the time spent; at the end, the agreement over all
structures. The perturbative timing counts one diagonalization with eigenvectors plus building
dH_m(k) and the projections; the exact timing the M member diagonalizations.

Example (cluster):
    python check_perturbative_sigma.py \
        --model-dirs train_{1..5}/infer/outputs_train/2026-10-06_09-41-0?/dft --avg-dir avg_ham \
        --band-json uncertainty_unseen.json --structures structure_untwisted_8675_300 structure_705 \
        --out perturbative_check.json --n-procs 4
"""

import argparse
import concurrent.futures
import json
import os
import time
from pathlib import Path

import numpy as np
import threadpoolctl
from scipy.linalg import eigh
from scipy.stats import spearmanr

from hamilflow.sparse_hamiltonian import SparseHamiltonianObj
from hamilflow.uncertainty import BandUncertaintyCalculator
from hamilflow.uncertainty.alignment import fit_rigid_shifts
from hamilflow.uncertainty.band_postprocess import BandUncertaintyPostProcessor
from hamilflow.uncertainty.hamiltonian_io import average_predicted_hamiltonians

VARIANTS = ("exact", "pt1", "pt1_deg", "ritz")


def _eig(hk, sk, kept, vecs):
    if kept is None:
        return eigh(hk, sk) if vecs else (eigh(hk, sk, eigvals_only=True), None)
    from deepx_dock.compute.eigen.ill_conditioned import eig_with_orbital_mask
    vals, v = eig_with_orbital_mask(hk, sk, kept, return_vecs=vecs)
    return vals[:len(kept)], (v if vecs else None)


def _as_eigh(a: np.ndarray) -> np.ndarray:
    """The Hermitian matrix `eigh` actually diagonalizes: it reads only the lower triangle. A
    predicted H(k) need not be exactly Hermitian, so the projections use the same matrix."""
    low = np.tril(a, -1)
    return low + low.conj().T + np.diag(np.real(np.diag(a)))


def _groups(e: np.ndarray, tol: float) -> list[np.ndarray]:
    cut = np.where(np.diff(e) > tol)[0] + 1
    return np.split(np.arange(len(e)), cut)


def structure_check(name: str, structure_dirs: list[Path], avg_dir: Path, calc: BandUncertaintyCalculator,
                    e_cut: float, tau: float, margin: float, deg_tol: float, blas_threads: int) -> tuple[str, dict]:
    with threadpoolctl.threadpool_limits(blas_threads, user_api="blas"):
        member_paths = [d / calc.hamiltonian_name for d in structure_dirs]
        avg_path = avg_dir / name / calc.hamiltonian_name
        if not avg_path.exists():
            average_predicted_hamiltonians(member_paths, avg_path)
        s_path = structure_dirs[0] / calc.overlap_name
        c = [r["c_eV"] for r in fit_rigid_shifts(member_paths, avg_path, s_path, calc.batch_entries)]

        avg = SparseHamiltonianObj(structure_dirs[0], H_file_path=avg_path)
        members = [SparseHamiltonianObj(d) for d in structure_dirs]
        ks, kw, _ = calc.build_irreducible_kpoints(avg, calc.grid_mesh, calc.symprec)
        handler = calc.make_ill_handler(avg, ks, n_jobs=1, parallel_k=False)
        kept = None if handler is None else handler.kept_orbitals
        homo, lumo = calc.homo_lumo_indices(avg)

        # averaged Hamiltonian with eigenvectors (the only diagonalization the perturbative route needs)
        non_herm = 0.0
        t_build = t_avg = 0.0
        avg_vals, avg_vecs = [], []
        for k in ks:
            t0 = time.monotonic()
            sk, hk = avg.Sk_and_Hk(k)
            t_build += time.monotonic() - t0
            non_herm = max(non_herm, float(np.abs(hk - hk.conj().T).max()))
            t0 = time.monotonic()
            v, x = _eig(hk, sk, kept, vecs=True)
            t_avg += time.monotonic() - t0
            avg_vals.append(v)
            avg_vecs.append(x)
        avg_vals = np.stack(avg_vals)                        # (k, band)
        vbm, cbm = avg_vals[:, homo].max(), avg_vals[:, lumo].min()
        reach = e_cut + margin
        inside = (avg_vals >= vbm - reach) & (avg_vals <= cbm + reach)
        b0, b1 = int(np.where(inside.any(0))[0][0]), int(np.where(inside.any(0))[0][-1]) + 1
        win = slice(b0, b1)

        t_pt = 0.0
        t_exact = 0.0
        dev = {v: np.zeros((len(members), len(ks), b1 - b0)) for v in VARIANTS}
        for ik, k in enumerate(ks):
            e_w = avg_vals[ik, win]
            psi = avg_vecs[ik][:, win]
            t1 = time.monotonic()
            sk = avg.get_Sk(k)
            hk_avg = _as_eigh(avg._dense_k(k))
            t_build += time.monotonic() - t1
            for m, mem in enumerate(members):
                t1 = time.monotonic()
                hk_m = mem._dense_k(k)
                t_build += time.monotonic() - t1
                non_herm = max(non_herm, float(np.abs(hk_m - hk_m.conj().T).max()))
                hk_m = _as_eigh(hk_m)
                t1 = time.monotonic()
                x = psi.conj().T @ ((hk_m - c[m] * sk - hk_avg) @ psi)
                pt1 = e_w + np.real(np.diag(x))
                deg = e_w.copy()
                for g in _groups(e_w, deg_tol):
                    deg[g] = np.linalg.eigvalsh(np.diag(e_w[g]) + x[np.ix_(g, g)])
                ritz = np.linalg.eigvalsh(np.diag(e_w) + x)
                t_pt += time.monotonic() - t1
                dev["pt1"][m, ik] = np.sort(pt1) - e_w
                dev["pt1_deg"][m, ik] = np.sort(deg) - e_w
                dev["ritz"][m, ik] = ritz - e_w

                t1 = time.monotonic()
                vals, _ = _eig(hk_m, sk, kept, vecs=False)
                t_exact += time.monotonic() - t1
                dev["exact"][m, ik] = vals[win] - c[m] - e_w

    sigma = {v: np.sqrt(np.mean(d ** 2, axis=0)) for v, d in dev.items()}   # (k, band in window)
    midgap = (vbm + cbm) / 2
    res = {
        "vbm_index": homo - b0,
        "kpoints": {f"k{i}": {"weight": int(kw[i]),
                              "eigvals_avg_ham_eV": (avg_vals[i, win] - midgap).tolist(),
                              "dev_exact_eV": dev["exact"][:, i].tolist(), "dev_ritz_eV": dev["ritz"][:, i].tolist(),
                              **{f"sigma_{v}": sigma[v][i].tolist() for v in VARIANTS}}
                    for i in range(len(ks))},
    }
    out = {"n_orbitals": int(avg.orbits_quantity), "n_kept_orbitals": None if kept is None else len(kept),
           "n_kpoints": len(ks), "window_bands": [b0, b1], "rigid_shift_eV": c,
           "max_non_hermiticity_eV": non_herm, "time_build_matrices_s": t_build, "time_avg_diag_s": t_avg, "time_projection_s": t_pt, "time_member_diag_s": t_exact,
           "scores": {}, "per_state": {}}
    w = None
    for v in VARIANTS:
        pp = BandUncertaintyPostProcessor({name: res}, key=f"sigma_{v}")
        st = pp.weighted_stats(name, e_cut, tau)
        out["scores"][v] = {"weighted_max_eV": st["weighted_max_eV"], "weighted_mean_eV": st["weighted_mean_eV"]}
        if w is None:
            w = pp.soft_window(pp.edge_distance(name), e_cut, tau)
    sel = w > 0.5
    ex = sigma["exact"][sel]
    for v in VARIANTS[1:]:
        pv = sigma[v][sel]
        rel = np.abs(pv - ex) / np.maximum(ex, 1e-6)
        out["per_state"][v] = {"n_states": int(sel.sum()), "median_rel_err": float(np.median(rel)),
                               "p90_rel_err": float(np.percentile(rel, 90)), "max_abs_err_eV": float(np.abs(pv - ex).max()),
                               "spearman": float(spearmanr(ex, pv)[0]) if sel.sum() > 2 else None}
    out["sigma"] = res
    return name, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dirs", nargs="+", required=True,
                    help="one entry per model; several roots per model separated by commas, searched in order")
    ap.add_argument("--avg-dir", required=True, type=Path)
    ap.add_argument("--structures", nargs="*", default=None, help="default: all structures in --band-json")
    ap.add_argument("--band-json", type=Path, default=None,
                    help="band uncertainty output: structure list, grid mesh and overlap settings are taken from it")
    ap.add_argument("--mesh", type=int, nargs=3, default=None, help="k-mesh (default: from --band-json)")
    ap.add_argument("--ill-threshold", type=float, default=1e-3, help="orbital removal threshold; <= 0 disables it")
    ap.add_argument("--species", type=json.loads, default=None, help='e.g. \'{"Mo": 42, "S": 16}\'')
    ap.add_argument("--e-cut", type=float, default=2.0)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=1.0, help="window = band edges +- (e_cut + margin) eV")
    ap.add_argument("--deg-tol", type=float, default=5e-3, help="eV; levels closer than this are treated as degenerate")
    ap.add_argument("--out", type=Path, default=Path("perturbative_check.json"))
    ap.add_argument("--n-procs", type=int, default=1)
    args = ap.parse_args()

    band = json.loads(args.band_json.read_text()) if args.band_json else {}
    names = args.structures or list(band)
    mesh = args.mesh or (band[names[0]]["grid_mesh"] if names and names[0] in band else None)
    if mesh is None:
        ap.error("give --mesh (or --band-json containing the structures)")
    kwargs = dict(grid_mesh=tuple(mesh), ill_method="orbital_removal" if args.ill_threshold > 0 else None,
                  ill_threshold=args.ill_threshold)
    if args.species:
        kwargs["species_number"] = args.species
    calc = BandUncertaintyCalculator(**kwargs)

    model_roots = [[Path(r) for r in entry.split(",") if r] for entry in args.model_dirs]
    located = {}
    for n in names:
        dirs = [next((r / n for r in roots if (r / n / calc.hamiltonian_name).exists()), None) for roots in model_roots]
        if None not in dirs:
            located[n] = dirs
    if len(located) < len(names):
        print(f"skipping {len(names) - len(located)}/{len(names)} structures not found for every model")
    if not located:
        raise SystemExit("no structure found for all models")

    results = json.loads(args.out.read_text()) if args.out.exists() else {}
    todo = [n for n in located if n not in results]
    n_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    blas = max(1, n_cpus // max(1, args.n_procs))  # CPUs actually allocated (e.g. by SLURM), not the node's
    print(f"{len(located)} structures, {len(todo)} to compute, mesh {mesh}, ill_threshold {args.ill_threshold}, "
          f"{args.n_procs} processes x {blas} BLAS threads")
    with concurrent.futures.ProcessPoolExecutor(max(1, args.n_procs)) as ex:
        futs = [ex.submit(structure_check, n, located[n], args.avg_dir, calc, args.e_cut, args.tau,
                          args.margin, args.deg_tol, blas) for n in todo]
        for fut in concurrent.futures.as_completed(futs):
            name, res = fut.result()
            results[name] = res
            tmp = args.out.with_suffix(".tmp")
            tmp.write_text(json.dumps(results))
            os.replace(tmp, args.out)
            sc = {v: res["scores"][v]["weighted_max_eV"] * 1e3 for v in VARIANTS}
            print(f"{name}: weighted_max exact {sc['exact']:.2f} | pt1 {sc['pt1']:.2f} | pt1_deg {sc['pt1_deg']:.2f} "
                  f"| ritz {sc['ritz']:.2f} meV;  time avg+proj {res['time_avg_diag_s'] + res['time_projection_s']:.1f}s"
                  f" vs members {res['time_member_diag_s']:.1f}s (building H(k) {res['time_build_matrices_s']:.1f}s);  max|H-H^dag| {res['max_non_hermiticity_eV']:.1e} eV", flush=True)

    done = [n for n in located if n in results]
    print(f"\n{len(done)} structures (band score: weighted_max, e_cut={args.e_cut}, tau={args.tau})")
    print(f"{'variant':9s} {'rho':>6s} {'median ratio':>13s} {'min ratio':>10s} {'max ratio':>10s} "
          f"{'state med rel err':>18s} {'state p90 rel err':>18s}")
    ex = np.array([results[n]["scores"]["exact"]["weighted_max_eV"] for n in done])
    for v in VARIANTS[1:]:
        x = np.array([results[n]["scores"][v]["weighted_max_eV"] for n in done])
        r = x / ex
        med = np.median([results[n]["per_state"][v]["median_rel_err"] for n in done])
        p90 = np.median([results[n]["per_state"][v]["p90_rel_err"] for n in done])
        rho = spearmanr(ex, x)[0] if len(done) > 2 else float("nan")
        print(f"{v:9s} {rho:6.2f} {np.median(r):13.3f} {r.min():10.3f} {r.max():10.3f} {med:18.3f} {p90:18.3f}")
    t_pt = sum(results[n]["time_avg_diag_s"] + results[n]["time_projection_s"] for n in done)
    t_ex = sum(results[n]["time_avg_diag_s"] + results[n]["time_member_diag_s"] for n in done)
    print(f"time: perturbative {t_pt:.1f}s, exact {t_ex:.1f}s (x{t_ex / max(t_pt, 1e-9):.1f})")


if __name__ == "__main__":
    main()
