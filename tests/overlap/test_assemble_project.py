"""Assembly of a structure's overlap from a two-centre table, and the real-space projection (path B)
against the production k-space projection (path A), on analytic Gaussian orbitals."""

import json

import h5py
import numpy as np
import pytest

from hamilflow.overlap.assemble import assemble_overlap, neighbour_pairs
from hamilflow.overlap.io import prepare_structure, read_info, read_layout, read_poscar
from hamilflow.overlap.pipeline import build_projected_overlap, project_full_overlap
from hamilflow.overlap.tables import TwoCentreTable, bond_blocks_to_channels, pair_vectors

# shells: a tight s ("1s", removed by Schur), a diffuse s, a p (truncated as "1p")
SHELLS = (0, 0, 1)
EXPONENTS = (4.0, 1.0, 1.2)  # 1/Å^2
CUTOFF = 5.5  # Å; exp(-0.5 * 5.5^2) ~ 3e-7 for the most diffuse pair
SCHUR = {"rules": [{"target_elements": ["C"], "remove_shells": ["1s"]}]}
TRUNC = {"rules": [{"target_elements": ["C"], "remove_shells": ["1p"]}]}


def _norm(a, l):
    """Normalization of r^l-weighted Gaussians x_k^l exp(-a r^2) (l = 0, 1)."""
    s = (np.pi / (2 * a)) ** 1.5
    return 1 / np.sqrt(s if l == 0 else s / (4 * a))


def gaussian_overlap(d):
    """Normalized overlap block between all shells on A = 0 and on B = d (5 x 5, p as (y, z, x))."""
    out = np.zeros((5, 5))
    off = [0, 1, 2]
    for sa, (la, a) in enumerate(zip(SHELLS, EXPONENTS)):
        for sb, (lb, b) in enumerate(zip(SHELLS, EXPONENTS)):
            g = a + b
            K = (np.pi / g) ** 1.5 * np.exp(-a * b / g * (d @ d)) * _norm(a, la) * _norm(b, lb)
            P = b * d / g
            PA, PB = P, P - d
            perm = [1, 2, 0]
            if la == 0 and lb == 0:
                blk = np.array([[K]])
            elif la == 0:
                blk = (PB * K)[perm][None, :]
            elif lb == 0:
                blk = (PA * K)[perm][:, None]
            else:
                blk = ((np.outer(PA, PB) + np.eye(3) / (2 * g)) * K)[np.ix_(perm, perm)]
            ra, rb = off[sa], off[sb]
            out[ra:ra + 2 * la + 1, rb:rb + 2 * lb + 1] = blk
    return out


@pytest.fixture(scope="module")
def table():
    r = np.linspace(0.8, CUTOFF + 0.1, 600)
    blocks = np.array([gaussian_overlap(np.array([0, 0, ri])) for ri in r])
    vals, resid = bond_blocks_to_channels(blocks, SHELLS, SHELLS)
    assert resid.max() < 1e-14
    radii = np.full(3, CUTOFF / 2 + 0.05)
    return TwoCentreTable("C", "C", SHELLS, SHELLS, r, vals, radii, radii,
                          onsite={"C": gaussian_overlap(np.zeros(3))}, atom_pair_cutoff=CUTOFF)


@pytest.fixture(scope="module")
def structure():
    rng = np.random.default_rng(0)
    lat = np.array([[3.1, 0.2, 0.0], [0.3, 2.9, 0.1], [0.0, 0.4, 3.4]])
    pos = np.array([[0.1, 0.2, 0.3], [1.3, 1.6, 1.2], [2.2, 0.4, 2.5]]) + rng.normal(scale=0.05, size=(3, 3))
    return prepare_structure(lat, pos, ["C"] * 3)


def test_neighbour_pairs_match_brute_force(structure):
    pairs, vecs = neighbour_pairs(structure, CUTOFF)
    brute = set()
    rng = range(-4, 5)
    for R in [(a, b, c) for a in rng for b in rng for c in rng]:
        for i in range(3):
            for j in range(3):
                d = structure.positions[j] + np.array(R) @ structure.lattice - structure.positions[i]
                if np.linalg.norm(d) < CUTOFF:
                    brute.add((*R, i, j))
    assert set(map(tuple, pairs.tolist())) == brute
    np.testing.assert_allclose(vecs, pair_vectors(structure, pairs), atol=1e-12)


def test_assembled_overlap_is_the_analytic_one(table, structure, tmp_path):
    stats = assemble_overlap(structure, table, tmp_path / "full", chunk_pairs=7)
    ap, bounds, shapes = read_layout(tmp_path / "full" / "overlap.h5")
    assert stats["n_pairs"] == len(ap)
    vecs = pair_vectors(read_poscar(tmp_path / "full" / "POSCAR"), ap)
    with h5py.File(tmp_path / "full" / "overlap.h5") as f:
        ent = f["entries"][:]
    index = {tuple(p): k for k, p in enumerate(ap.tolist())}
    for k, (p, d) in enumerate(zip(ap, vecs)):
        blk = ent[bounds[k]:bounds[k + 1]].reshape(shapes[k])
        np.testing.assert_allclose(blk, gaussian_overlap(d), atol=1e-7)
        partner = index[(-p[0], -p[1], -p[2], p[4], p[3])]
        np.testing.assert_allclose(blk, ent[bounds[partner]:bounds[partner + 1]].reshape(5, 5).T, atol=1e-14)
    info = read_info(tmp_path / "full")
    assert info["elements_orbital_map"] == {"C": [0, 0, 1]} and info["orbits_quantity"] == 15
    with h5py.File(tmp_path / "full" / "hamiltonian.h5") as f:
        assert f["entries"].shape == (int(bounds[-1]),) and not f["entries"][:].any()


@pytest.mark.parametrize("n_terms, tol", [(1, 1e-2), (2, 1e-4), (3, 1e-6)])
def test_path_b_matches_path_a(table, structure, tmp_path, n_terms, tol):
    info = build_projected_overlap(structure, "s", table, tmp_path, SCHUR, TRUNC, path="B", n_terms=n_terms)
    a_dir = project_full_overlap(tmp_path / "deeph_full" / "s", tmp_path, "s", SCHUR, TRUNC, path="A", n_workers=1)
    b_dir = info["projected_dir"]
    apa, _ba, sa = read_layout(a_dir / "overlap.h5")
    apb, _bb, sb = read_layout(f"{b_dir}/overlap.h5")
    np.testing.assert_array_equal(apa, apb)
    np.testing.assert_array_equal(sa, sb)
    assert sa[0].tolist() == [1, 1]  # only the diffuse s remains
    with h5py.File(a_dir / "overlap.h5") as fa, h5py.File(f"{b_dir}/overlap.h5") as fb:
        ea, eb = fa["entries"][:], fb["entries"][:]
    err = np.abs(ea - eb).max()
    assert err < tol
    # the correction itself is not small: the Schur step matters
    with h5py.File(tmp_path / "deeph_full" / "s" / "overlap.h5") as f:
        full = f["entries"][:].reshape(-1, 5, 5)[:, 1, 1]
    assert np.abs(full - ea).max() > 1e-2
    ia, ib = (json.loads((d / "info.json").read_text()) for d in (a_dir, __import__("pathlib").Path(b_dir)))
    assert ia["elements_orbital_map"] == ib["elements_orbital_map"] == {"C": [0]}
    assert ia["occupation"] == ib["occupation"] and ia["orbits_quantity"] == ib["orbits_quantity"]


def test_reorder_like(table, structure, tmp_path):
    from hamilflow.overlap.io import reorder_like

    assemble_overlap(structure, table, tmp_path / "a")
    ap, bounds, _ = read_layout(tmp_path / "a" / "overlap.h5")
    perm = np.random.default_rng(1).permutation(len(ap))
    with h5py.File(tmp_path / "ref.h5", "w") as f:
        f.create_dataset("atom_pairs", data=ap[perm])
    reorder_like(tmp_path / "a" / "overlap.h5", tmp_path / "ref.h5", tmp_path / "b.h5", chunk_pairs=5)
    ap2, b2, _ = read_layout(tmp_path / "b.h5")
    np.testing.assert_array_equal(ap2, ap[perm])
    with h5py.File(tmp_path / "a" / "overlap.h5") as fa, h5py.File(tmp_path / "b.h5") as fb:
        ea, eb = fa["entries"][:], fb["entries"][:]
    for k_new, k_old in enumerate(perm):
        np.testing.assert_array_equal(eb[b2[k_new]:b2[k_new + 1]], ea[bounds[k_old]:bounds[k_old + 1]])


def _naive_correction(pairs, X, U):
    """``corr(R, A, B) = sum X_(Rp, A, C) U_(Rq, B, C)^T`` over Rp - Rq = R, by brute force."""
    index = {tuple(p): k for k, p in enumerate(pairs.tolist())}
    out = np.zeros((len(pairs), X.shape[1], U.shape[1]))
    for kp, p in enumerate(pairs.tolist()):
        for kq, q in enumerate(pairs.tolist()):
            if p[4] != q[4]:
                continue
            o = index.get((p[0] - q[0], p[1] - q[1], p[2] - q[2], p[3], q[3]))
            if o is not None:
                out[o] += X[kp] @ U[kq].T
    return out


@pytest.mark.parametrize("max_rows", [1, 37, 1 << 18])
def test_schur_correction_matches_brute_force(table, structure, tmp_path, max_rows):
    from hamilflow.overlap.schur_rs import _PairIndex, neumann_X, schur_correction

    assemble_overlap(structure, table, tmp_path / "full")
    pairs, bounds, _ = read_layout(tmp_path / "full" / "overlap.h5")
    with h5py.File(tmp_path / "full" / "overlap.h5") as f:
        blocks = f["entries"][:].reshape(-1, 5, 5)
    U = blocks[:, 1:2, 0:1].copy()  # kept: the diffuse s; removed: the tight s
    E = blocks[:, 0:1, 0:1].copy()
    E[(pairs[:, 3] == pairs[:, 4]) & ~pairs[:, :3].any(axis=1)] -= 1.0
    assert ((pairs[:, 3] == pairs[:, 4]) & pairs[:, :3].any(axis=1)).any()  # self-images present
    for n_terms in (1, 2, 3):
        X = neumann_X(pairs, U, E, _PairIndex(pairs, 3), n_terms, max_rows)
        ref = _naive_correction(pairs, X, U)
        got = schur_correction(pairs, U, E, 3, n_terms=n_terms, max_rows=max_rows)
        assert np.abs(ref).max() > 1e-2
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-14)
