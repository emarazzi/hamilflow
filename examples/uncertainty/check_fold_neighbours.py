"""Does a K-fold member that stands out on a structure miss the training data closest to it?

In a K-fold ensemble member m is trained without fold m. If the training structures nearest to a
pool structure lie mostly in fold m, member m extrapolates there while the others do not, and the
ensemble spread on that structure is driven by member m alone. This script checks that with the
AMD descriptor used for the structure selection (`CandidateSelector`):

1. Fold composition by family (twisted 100 K / twisted 300 K / untwisted 300 K).
2. For every query structure: its `--n-nearest` nearest training structures (AMD distance), their
   folds, and the share of each fold among them compared with the fold's share of the training set.
3. Over all pool structures with a per-member indicator, for each member m: the indicator of
   member m when the nearest training structure is in fold m versus when it is not, and the
   Spearman rho between the share of fold m among the nearest neighbours and the indicator.
   Per-member indicators:
   - `--band-json` (band uncertainty output): |gap of member m - gap of the mean| (all structures);
   - `--perturbative-json` (`check_perturbative_sigma.py` output, may be repeated): RMS over the
     states within `--e-cut` of the band edges of the exact deviation of member m (only the
     structures in those files; preferred, since member 4's deviations leave the gap unchanged).

Structures come from an extended-XYZ trajectory with `atoms.info["index"]` labels (as for the
selection; `--label-prefix` is prepended to labels that do not start with it), or from POSCARs in
`<root>/<structure>/POSCAR` under `--poscar-roots`.

Example (cluster):
    python check_fold_neighbours.py --folds folds.json --trajectory dataset.extxyz \
        --band-json uncertainty_unseen.json --perturbative-json perturbative_check.json perturbative_check_2.json \
        --queries structure_2246_300 structure_2467_300 structure_untwisted_3698_300 structure_3419_300 \
                  structure_untwisted_8675_300 structure_705 structure_1128 --out fold_neighbours.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from hamilflow.uncertainty.band_postprocess import BandUncertaintyPostProcessor


def family(name: str) -> str:
    return ("untwisted" if "untwisted" in name else "twisted") + ("_300" if name.endswith("_300") else "_100")


def load_structures(names: set[str], trajectory: Path | None, poscar_roots: list[Path], prefix: str) -> dict:
    from ase.io import read
    found = {}
    if trajectory is not None:
        # ASE's extxyz reader turns a label like 2246_300 into the integer 2246300, so labels are
        # also matched with the underscores removed
        loose = {n.replace("_", ""): n for n in names}
        for atoms in read(str(trajectory), index=":"):
            label = str(atoms.info.get("index", ""))
            name = label if label.startswith(prefix) else prefix + label
            name = name if name in names else loose.get(name.replace("_", ""))
            if name is not None:
                found[name] = atoms
    for root in poscar_roots:
        for n in names - set(found):
            if (root / n / "POSCAR").exists():
                found[n] = read(str(root / n / "POSCAR"), format="vasp")
    return found


def amd_vectors(structures: dict, k: int) -> dict:
    import amd
    return {n: np.asarray(amd.AMD(amd.PeriodicSet(a.get_positions(), a.get_cell()[:]), k=k)) for n, a in structures.items()}


def indicators(args) -> dict:
    """{indicator name: {structure: per-member array}}"""
    out = {}
    if args.band_json:
        band = json.loads(args.band_json.read_text())
        out["|gap dev| (meV)"] = {s: np.abs(np.array(v["per_model_gap_eV"]) - v["avg_hamiltonian_gap_eV"]) * 1e3
                                  for s, v in band.items() if "per_model_gap_eV" in v}
    for path in args.perturbative_json or []:
        res = json.loads(path.read_text())
        key = f"edge dev RMS, e_cut {args.e_cut} (meV)"
        for s, v in res.items():
            sig = v["sigma"]
            per_k = list(sig["kpoints"].values())
            if "dev_exact_eV" not in per_k[0]:
                continue
            pp = BandUncertaintyPostProcessor({s: sig}, key="sigma_exact")
            w = pp.soft_window(pp.edge_distance(s), args.e_cut, 0.1) > 0.5          # (k, band)
            dev = np.array([k["dev_exact_eV"] for k in per_k])                       # (k, member, band)
            out.setdefault(key, {})[s] = np.sqrt(np.mean(dev.transpose(1, 0, 2)[:, w] ** 2, axis=1)) * 1e3
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=Path, required=True,
                    help='JSON {"1": [structures held out by member 1], ...}; key order = member order')
    ap.add_argument("--trajectory", type=Path, default=None)
    ap.add_argument("--poscar-roots", type=Path, nargs="*", default=[])
    ap.add_argument("--label-prefix", default="structure_")
    ap.add_argument("--queries", nargs="*", default=[])
    ap.add_argument("--band-json", type=Path, default=None)
    ap.add_argument("--perturbative-json", type=Path, nargs="*", default=None)
    ap.add_argument("--e-cut", type=float, default=1.0)
    ap.add_argument("--k-neighbors", type=int, default=100, help="AMD k (use the value of the selection)")
    ap.add_argument("--n-nearest", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("fold_neighbours.json"))
    args = ap.parse_args()

    folds = {str(k): list(v) for k, v in json.loads(args.folds.read_text()).items()}
    fold_ids = list(folds)
    fold_of = {s: f for f, ss in folds.items() for s in ss}
    dup = [s for s, c in Counter(s for ss in folds.values() for s in ss).items() if c > 1]
    if dup:
        raise SystemExit(f"structures in more than one fold: {dup[:5]}")
    train = sorted(fold_of)

    print("1. fold composition (structures held out by each member)")
    fams = sorted({family(s) for s in train})
    print(f"   {'fold':>4s} {'n':>4s} " + " ".join(f"{f:>14s}" for f in fams))
    for f in fold_ids:
        c = Counter(family(s) for s in folds[f])
        print(f"   {f:>4s} {len(folds[f]):4d} " + " ".join(f"{c[g]:14d}" for g in fams))
    tot = Counter(family(s) for s in train)
    print(f"   {'all':>4s} {len(train):4d} " + " ".join(f"{tot[g]:14d}" for g in fams))

    ind = indicators(args)
    pool = set(args.queries) | {s for d in ind.values() for s in d}
    pool -= set(train)
    structures = load_structures(set(train) | pool, args.trajectory, args.poscar_roots, args.label_prefix)
    missing_train = [s for s in train if s not in structures]
    if missing_train:
        print(f"\nWARNING: {len(missing_train)} training structures not found, e.g. {missing_train[:3]}")
    train = [s for s in train if s in structures]
    pool = sorted(s for s in pool if s in structures)
    print(f"\n{len(train)} training and {len(pool)} pool structures found; computing AMD (k={args.k_neighbors})")
    vec = amd_vectors({s: structures[s] for s in train + pool}, args.k_neighbors)
    T = np.stack([vec[s] for s in train])
    share = np.array([sum(fold_of[s] == f for s in train) / len(train) for f in fold_ids])

    out = {"folds": {f: len(folds[f]) for f in fold_ids}, "structures": {}}
    for s in pool:
        d = np.linalg.norm(T - vec[s], axis=1)
        nn = np.argsort(d)[:args.n_nearest]
        frac = np.array([np.mean([fold_of[train[i]] == f for i in nn]) for f in fold_ids])
        out["structures"][s] = {
            "nearest": [{"structure": train[i], "fold": fold_of[train[i]], "amd_distance": float(d[i])} for i in nn],
            "fold_share_nearest": frac.tolist(),
            **{name: vals[s].tolist() for name, vals in ind.items() if s in vals},
        }

    print(f"\n2. nearest {args.n_nearest} training structures of the queries (fold shares in the training set: "
          + ", ".join(f"{f}:{x:.2f}" for f, x in zip(fold_ids, share)) + ")")
    for s in [q for q in args.queries if q in out["structures"]]:
        r = out["structures"][s]
        print(f"   {s} ({family(s)})")
        for nb in r["nearest"]:
            print(f"      fold {nb['fold']}  {nb['structure']:32s} ({family(nb['structure'])})  d={nb['amd_distance']:.4f}")
        for name, vals in ind.items():
            if s in vals:
                print(f"      {name} per member: " + " ".join(f"{x:.1f}" for x in vals[s]))
    for q in args.queries:
        if q not in out["structures"]:
            print(f"   {q}: not found{' (it is a training structure)' if q in fold_of else ''}")

    print("\n3. per-member indicator vs nearest training data in that member's held-out fold")
    for name, vals in ind.items():
        ss = [s for s in pool if s in vals]
        if len(ss) < 3:
            continue
        X = np.array([vals[s] for s in ss])                                           # (structure, member)
        F = np.array([out["structures"][s]["fold_share_nearest"] for s in ss])
        first = np.array([fold_ids.index(out["structures"][s]["nearest"][0]["fold"]) for s in ss])
        print(f"   {name}, {len(ss)} structures")
        print(f"   {'member':>6s} {'mean if NN in its fold (n)':>28s} {'mean otherwise (n)':>20s} {'rho(share, ind)':>16s}")
        for m, f in enumerate(fold_ids):
            if m >= X.shape[1]:
                break
            inside = first == m
            a = f"{X[inside, m].mean():.2f} ({inside.sum()})" if inside.any() else "- (0)"
            b = f"{X[~inside, m].mean():.2f} ({(~inside).sum()})" if (~inside).any() else "- (0)"
            rho = spearmanr(F[:, m], X[:, m])[0] if np.ptp(F[:, m]) > 0 else float("nan")
            print(f"   {f:>6s} {a:>28s} {b:>20s} {rho:16.2f}")
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
