"""Orbital-weighted matrix-element uncertainty: a band-edge-aware score with no diagonalization
per structure.

To first order a band-edge state moves by psi^dag dH psi, so matrix errors matter in proportion to
how much the edge states live on the orbitals involved. This script takes that orbital character
from ONE reference structure (one diagonalization in total) and uses it as a prior for all others:

1. Reference (`--reference-dir` + `--reference-structure`, or `--reference-dir` pointing at the
   structure folder itself; ideally a DFT structure: `hamiltonian.h5`, `overlap.h5`,
   `info.json`, `POSCAR`). Diagonalize on a small k-mesh, take the Loewdin population of every
   state, weight the states by their distance d from the band edges with the soft window
   `F((d - e_cut)/tau)` (as in `BandUncertaintyPostProcessor`), and average per orbital channel
   (species, local orbital index). This gives p_c^VB and p_c^CB: edge population per orbital of
   channel c, normalized to max 1. Saved to `--weights-out` (reused with `--weights-in`).

2. Every structure of the ensemble: align each member to the average by its rigid shift c_m S,
   sigma_e = RMS_m |h_m - c_m s - h_avg| per stored element (as `MatrixUncertaintyCalculator`),
   and with W_e = p_{c(i)} p_{c(j)} (edge = mean of VB and CB weights):
   - `orb_rms`:      sqrt(sum W sigma^2 / sum W)
   - `orb_pair_max`: max over atom pairs of sqrt(sum_block W sigma^2 / n_block)
   - `coh_vb`, `coh_cb`, `coh_gap`: coherent on-site estimate. For member m,
     d_m^X = sum_i p_i^X dH^m_ii / sum_i p_i^X over on-site diagonal elements (signed), i.e. the
     edge shift if the edge state sat on isolated orbitals; reported as RMS over members of
     d^VB, d^CB and (d^CB - d^VB)/2 (the edge shift after midgap alignment).

3. With `--band-json`, the scores are compared to the band score (`weighted_max_eV` of
   `BandUncertaintyPostProcessor.weighted_summary`, `e_cut=2.0`, `vbm_fraction=None` by default):
   Spearman rho overall and per family, AUC for band sigma > `--tail-mev`, top-10 overlap.

Example (cluster):
    python orbital_weighted_uncertainty.py \
        --model-dirs train_{1..5}/infer/outputs_train/2026-10-06_09-41-0?/dft \
        --avg-dir avg_ham --reference-dir <dft_root> --reference-structure structure_4234_300 \
        --band-json uncertainty_unseen.json --out orbital_weighted.json --n-procs 16
"""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.stats import mannwhitneyu, spearmanr

from hamilflow.sparse_hamiltonian import SparseHamiltonianObj
from hamilflow.uncertainty.alignment import SparseStream, fit_rigid_shifts
from hamilflow.uncertainty.band_postprocess import BandUncertaintyPostProcessor
from hamilflow.uncertainty.hamiltonian_io import average_predicted_hamiltonians


def channel_labels(obj) -> list[tuple[str, int]]:
    """(species, local orbital index) of every orbital of the cell, in matrix order."""
    return [(el, i) for el in obj.elements
            for i in range(int(np.sum(2 * np.array(obj.elements_orbital_map[el]) + 1)))]


def reference_weights(ref_dir: Path, h_file: Path | None, mesh: list[int], e_cut: float, tau: float,
                      occupation: int | None) -> dict:
    obj = SparseHamiltonianObj(ref_dir, H_file_path=h_file)
    if obj.spinful:
        raise NotImplementedError("spinful reference: use a spinless reference structure")
    n_elec = occupation if occupation is not None else obj.occupation
    if n_elec is None:
        raise ValueError(f"{ref_dir}: no occupation in info.json, pass --occupation")
    n_occ = n_elec if obj.spinful else n_elec // 2
    labels = channel_labels(obj)
    n_orb = obj.orbits_quantity
    grid = np.stack(np.meshgrid(*[np.arange(m) / m for m in mesh], indexing="ij"), -1).reshape(-1, 3)

    vals, pops = [], []
    for k in grid:
        sk, hk = obj.Sk_and_Hk(k)
        e, v = eigh(hk, sk)
        w_s, u_s = np.linalg.eigh(sk)
        s_half = (u_s * np.sqrt(np.clip(w_s, 0, None))) @ u_s.conj().T
        pop = np.abs(s_half @ v) ** 2  # Loewdin: each column sums to 1
        vals.append(e)
        pops.append(pop)
    vals = np.stack(vals, 1)                   # (band, k)
    vbm, cbm = vals[n_occ - 1].max(), vals[n_occ].min()

    out = {"reference": str(ref_dir), "mesh": mesh, "e_cut": e_cut, "tau": tau,
           "gap_eV": float(cbm - vbm), "weights": {}}
    for edge in ("vb", "cb"):
        acc = np.zeros(n_orb)
        for ik, pop in enumerate(pops):
            band = np.arange(vals.shape[0])
            sel = band < n_occ if edge == "vb" else band >= n_occ
            d = (vbm - vals[sel, ik]) if edge == "vb" else (vals[sel, ik] - cbm)
            acc += pop[:, sel] @ BandUncertaintyPostProcessor.soft_window(np.clip(d, 0, None), e_cut, tau)
        per_channel = {}
        for (el, i), a in zip(labels, acc):
            per_channel.setdefault(el, {}).setdefault(i, []).append(a)
        mean = {el: [float(np.mean(ch[i])) for i in sorted(ch)] for el, ch in per_channel.items()}
        top = max(max(v) for v in mean.values())
        out["weights"][edge] = {el: [x / top for x in v] for el, v in mean.items()}
    return out


def _flat_weights(obj, weights: dict, spinful: bool) -> dict:
    """Per (species_i, species_j): flattened block of p_i p_j, with spin blocks if spinful."""
    cache = {}
    for a in set(obj.elements):
        for b in set(obj.elements):
            blk = np.outer(weights[a], weights[b])
            if spinful:
                blk = np.tile(blk, (2, 2))  # same weight in every spin block
            cache[(a, b)] = blk.reshape(-1)
    return cache


def structure_scores(name: str, structure_dirs: list[Path], avg_dir: Path, weights: dict,
                     hname: str, sname: str, batch_entries: int) -> tuple[str, dict]:
    member_paths = [d / hname for d in structure_dirs]
    avg_path = avg_dir / name / hname
    if not avg_path.exists():
        average_predicted_hamiltonians(member_paths, avg_path)
    s_path = structure_dirs[0] / sname
    obj = SparseHamiltonianObj(structure_dirs[0], load_hamiltonian=False)
    rigid = fit_rigid_shifts(member_paths, avg_path, s_path, batch_entries)
    c = [r["c_eV"] for r in rigid]
    w_vb, w_cb = weights["vb"], weights["cb"]
    w_edge = {el: [(x + y) / 2 for x, y in zip(w_vb[el], w_cb[el])] for el in w_vb}
    cache = _flat_weights(obj, w_edge, obj.spinful)
    diag_vb = {el: np.array(w_vb[el]) for el in w_vb}
    diag_cb = {el: np.array(w_cb[el]) for el in w_cb}

    n_m = len(member_paths)
    sw = swsig = 0.0
    pair_max = 0.0
    d_vb, d_cb = np.zeros(n_m), np.zeros(n_m)
    n_vb = n_cb = 0.0
    stream = SparseStream(member_paths + [avg_path], s_path, batch_entries)
    els = obj.elements
    for p0, p1, hs, s in stream:
        ref = hs[n_m]
        devs = np.stack([hs[m] - c[m] * s - ref for m in range(n_m)])
        sig2 = np.mean(np.abs(devs) ** 2, axis=0)
        pairs = stream.atom_pairs[p0:p1]
        w = np.concatenate([cache[(els[i], els[j])] for i, j in pairs[:, 3:5]])
        sw += float(w.sum())
        swsig += float(np.sum(w * sig2))
        offsets = stream.bounds[p0:p1] - stream.bounds[p0]
        sizes = np.diff(np.append(offsets, len(w)))
        pair_ms = np.add.reduceat(w * sig2, offsets) / sizes
        pair_max = max(pair_max, float(np.sqrt(pair_ms.max(initial=0.0))))
        onsite = np.where((pairs[:, :3] == 0).all(1) & (pairs[:, 3] == pairs[:, 4]))[0]
        for q in onsite:
            n_blk = int(stream.shapes[p0 + q][0])
            e0 = offsets[q]
            dd = np.real(devs[:, e0:e0 + n_blk * n_blk].reshape(n_m, n_blk, n_blk).diagonal(axis1=1, axis2=2))
            el = els[pairs[q, 3]]
            pv, pc = diag_vb[el], diag_cb[el]
            if obj.spinful:
                pv, pc = np.tile(pv, 2), np.tile(pc, 2)
            d_vb += dd @ pv
            d_cb += dd @ pc
            n_vb += pv.sum()
            n_cb += pc.sum()
    d_vb /= n_vb
    d_cb /= n_cb
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
    return name, {
        "orb_rms": float(np.sqrt(swsig / sw)),
        "orb_pair_max": pair_max,
        "coh_vb": rms(d_vb), "coh_cb": rms(d_cb), "coh_gap": rms((d_cb - d_vb) / 2),
        "per_model_rigid_shift_eV": c,
    }


def family(name: str) -> str:
    return "untwisted" if "untwisted" in name else ("300" if name.endswith("_300") else "100")


def compare(results: dict, band_json: Path, e_cut: float, tau: float, tail: float) -> None:
    pp = BandUncertaintyPostProcessor.from_json(band_json, key="sigma_eV_avg_ham")
    names = [s for s in results if s in pp.results]
    ws = {s: pp.weighted_stats(s, e_cut, tau, None, None) for s in names}
    band = np.array([ws[s]["weighted_max_eV"] * 1e3 for s in names])
    fam = np.array([family(s) for s in names])
    print(f"\n{len(names)} structures; band score = weighted_max (e_cut={e_cut}, tau={tau}), meV")
    print(f"{'score':14s} {'rho all':>8s} " + " ".join(f"{'rho ' + f:>13s}" for f in sorted(set(fam)))
          + f" {'AUC>' + str(tail):>9s} {'top10':>6s}")
    top_band = set(np.argsort(-band)[:10])
    for key in ("orb_rms", "orb_pair_max", "coh_vb", "coh_cb", "coh_gap"):
        x = np.array([results[s][key] for s in names])
        row = f"{key:14s} {spearmanr(band, x)[0]:8.2f} "
        row += " ".join(f"{spearmanr(band[fam == f], x[fam == f])[0]:13.2f}" for f in sorted(set(fam)))
        hi = band > tail
        auc = mannwhitneyu(x[hi], x[~hi]).statistic / (hi.sum() * (~hi).sum()) if 0 < hi.sum() < len(hi) else np.nan
        row += f" {auc:9.2f} {len(top_band & set(np.argsort(-x)[:10])):6d}"
        print(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dirs", nargs="+", required=True,
                    help="one entry per model; an entry may list several roots separated by commas "
                         "(e.g. predictions split over several inference runs), searched in order")
    ap.add_argument("--avg-dir", required=True, type=Path, help="averaged Hamiltonians (read, or written if missing)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--reference-dir", type=Path,
                   help="root of the reference structures (with --reference-structure) or one structure folder; "
                        "DFT recommended")
    g.add_argument("--weights-in", type=Path, help="channel weights written by an earlier run")
    ap.add_argument("--reference-structure", default=None, help="structure folder under --reference-dir")
    ap.add_argument("--reference-h", type=Path, default=None, help="H file for the reference (default: its hamiltonian.h5)")
    ap.add_argument("--reference-mesh", type=int, nargs=3, default=[3, 3, 1])
    ap.add_argument("--occupation", type=int, default=None, help="override info.json occupation of the reference")
    ap.add_argument("--weights-out", type=Path, default=Path("orbital_weights.json"))
    ap.add_argument("--e-cut", type=float, default=2.0)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--structures", nargs="*", default=None, help="default: those in --band-json, else all")
    ap.add_argument("--band-json", type=Path, default=None)
    ap.add_argument("--tail-mev", type=float, default=15.0)
    ap.add_argument("--out", type=Path, default=Path("orbital_weighted.json"))
    ap.add_argument("--n-procs", type=int, default=1)
    ap.add_argument("--hamiltonian-name", default="hamiltonian.h5")
    ap.add_argument("--overlap-name", default="overlap.h5")
    ap.add_argument("--batch-entries", type=int, default=4_000_000)
    args = ap.parse_args()

    model_roots = [[Path(r) for r in entry.split(",") if r] for entry in args.model_dirs]

    if args.weights_in:
        ref = json.loads(args.weights_in.read_text())
    else:
        ref_dir = args.reference_dir / args.reference_structure if args.reference_structure else args.reference_dir
        if not (ref_dir / "info.json").exists():
            ap.error(f"no info.json in {ref_dir}: --reference-dir must be one structure's folder, "
                     "or the root together with --reference-structure <name>")
        ref = reference_weights(ref_dir, args.reference_h, args.reference_mesh,
                                args.e_cut, args.tau, args.occupation)
        args.weights_out.write_text(json.dumps(ref, indent=2))
        print(f"reference gap {ref['gap_eV']:.3f} eV; channel weights -> {args.weights_out}")
    for edge, w in ref["weights"].items():
        for el, v in w.items():
            print(f"  {edge} {el}: " + " ".join(f"{x:.2f}" for x in v))

    if args.structures:
        names = args.structures
    elif args.band_json:
        names = list(json.loads(args.band_json.read_text()))
    else:
        names = sorted({p.name for r in model_roots[0] for p in r.iterdir() if p.is_dir()})
    located, missing = {}, {}
    for n in names:
        dirs = [next((r / n for r in roots if (r / n / args.hamiltonian_name).exists()), None) for roots in model_roots]
        if None in dirs:
            missing[n] = [str(roots[0] / n) for roots, d in zip(model_roots, dirs) if d is None]
        else:
            located[n] = dirs
    if missing:
        n, m = next(iter(missing.items()))
        print(f"skipping {len(missing)}/{len(names)} structures not found for every model, e.g. {m[0]}")
    names = [n for n in names if n in located]
    if not names:
        raise SystemExit("no structure found in all --model-dirs; check --model-dirs against --band-json")

    results = json.loads(args.out.read_text()) if args.out.exists() else {}
    todo = [n for n in names if n not in results]
    print(f"{len(names)} structures, {len(todo)} to compute")
    with concurrent.futures.ProcessPoolExecutor(max(1, args.n_procs)) as ex:
        futs = [ex.submit(structure_scores, n, located[n], args.avg_dir, ref["weights"],
                          args.hamiltonian_name, args.overlap_name, args.batch_entries) for n in todo]
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            name, res = fut.result()
            results[name] = res
            tmp = args.out.with_suffix(".tmp")
            tmp.write_text(json.dumps(results, indent=2))
            os.replace(tmp, args.out)
            print(f"[{i}/{len(todo)}] {name}: orb_rms {res['orb_rms'] * 1e3:.3f} meV, "
                  f"pair_max {res['orb_pair_max'] * 1e3:.2f} meV, coh_gap {res['coh_gap'] * 1e3:.2f} meV", flush=True)

    if args.band_json:
        compare({n: results[n] for n in names}, args.band_json, args.e_cut, args.tau, args.tail_mev)


if __name__ == "__main__":
    main()
