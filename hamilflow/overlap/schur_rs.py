"""Real-space Schur elimination of core orbitals from a full-basis overlap, then truncation.

The production projection (`hamilflow.projection.run_projection`, k-space) maps S to
``S' = S_kk - S_km S_mm^-1 S_mk`` (m: removed orbitals, e.g. the 1s of every C; k: kept), then
drops the truncated orbitals (e.g. g) from S'. Only S enters. Here the same is done in real
space, on the stored pairs only, with a Neumann series for ``S_mm^-1``:

    S_mm = I + E,    S_mm^-1 ~ I - E + E^2 - ...   (``n_terms`` = 1, 2, 3 terms)

For tight C, E is small: on-site 1s-1s is 1 to ~1e-6 and the largest off-site 1s-1s overlap
is ~2e-4 (1.3 Å), so one term (E^0) is the leading correction, the E term is ~1e-4 relative to
it, and E^2 is negligible.

Notation: a stored pair ``p = (R, i, C)`` gives ``U_p = S[(i, 0), (C, R)][k_out, m]``, the
overlap of the kept-and-not-truncated orbitals of atom i (cell 0) with the removed orbitals of
atom C in cell R. With ``X = U (I - E + E^2 ...)`` (E applied across the sites (C, R)),

    S''_{(A, 0), (B, R)} = S_kk - sum_{C, Rc} X_{(Rc, A, C)} U_{(Rc - R, B, C)}^T,

accumulated only on the stored pairs (A, B, R), and Hermitized as the production does,
``0.5 (M(R) + M(-R)^T)``. Products ``U E`` that would land on (i, C, R) pairs beyond the stored
pair set are dropped: those involve the tails of two functions at > 11 Å apart and are ~0.

Evaluation: ``E`` is applied with one batched block product per (pair, E pair) row. The
correction is computed per output atom B: all outputs ``(R, A, B)`` share the list of pairs
``q = (Rq, B, C)``, so their terms are gathered into one matrix and reduced by a single GEMM
(see `correction_blocks`). ``max_rows`` bounds the rows of each step (memory), not the result.

Single element (all atoms with the same orbital list).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.sparse as sp
from deepx_dock.CONSTANT import DEEPX_OVERLAP_FILENAME, DEEPX_POSCAR_FILENAME

from hamilflow.projection.io import (
    build_reduced_elements_orbital_map,
    create_zero_dataset,
    reduce_occupation_value,
)
from hamilflow.projection.models import RemovalPlanLike
from hamilflow.projection.removal import coerce_removal_plan, resolve_indices_from_rules

from .io import pair_layout, read_info, read_layout, read_poscar

logger = logging.getLogger(__name__)


class _PairIndex:
    """Lookup of ``(R1, R2, R3, i, j)`` rows by encoded integer keys."""

    def __init__(self, atom_pairs: np.ndarray, n_atoms: int):
        self.n_atoms = n_atoms
        self.rmax = int(np.abs(atom_pairs[:, :3]).max()) * 2 + 2 if len(atom_pairs) else 2
        keys = self.encode(atom_pairs[:, :3], atom_pairs[:, 3], atom_pairs[:, 4])
        self.order = np.argsort(keys)
        self.sorted_keys = keys[self.order]

    def encode(self, R: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        w = 2 * self.rmax + 1
        R = np.asarray(R, dtype=np.int64) + self.rmax
        r = (R[..., 0] * w + R[..., 1]) * w + R[..., 2]
        return (r * self.n_atoms + np.asarray(i, dtype=np.int64)) * self.n_atoms + np.asarray(j, dtype=np.int64)

    def lookup(self, R: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        """Row index of each (R, i, j), -1 where absent (or where |R| is out of the encodable range)."""
        R = np.asarray(R, dtype=np.int64)
        inside = np.all(np.abs(R) <= self.rmax, axis=-1)
        keys = self.encode(np.where(inside[..., None], R, 0), i, j)
        pos = np.searchsorted(self.sorted_keys, keys)
        pos = np.minimum(pos, len(self.sorted_keys) - 1)
        found = inside & (self.sorted_keys[pos] == keys)
        return np.where(found, self.order[pos], -1)


def resolve_local_indices(
    shells: list[int], schur_plan: RemovalPlanLike, truncate_plan: RemovalPlanLike | None, element: str
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Per-atom orbital indices: removed by the Schur stage (``rm``), and kept after both stages
    (``keep_out``), plus the removed indices of each stage in its own (single atom) numbering.
    The truncation plan refers to the basis after the Schur stage, as in production."""
    eom = {element: list(shells)}
    rm, _ = resolve_indices_from_rules([element], eom, coerce_removal_plan(schur_plan))
    nb = sum(2 * l + 1 for l in shells)
    after_schur = [i for i in range(nb) if i not in set(rm)]
    rm_t: list[int] = []
    if truncate_plan is not None:
        reduced = build_reduced_elements_orbital_map([element], eom, rm)
        rm_t, _ = resolve_indices_from_rules([element], reduced, coerce_removal_plan(truncate_plan))
    keep_out = [after_schur[k] for k in range(len(after_schur)) if k not in set(rm_t)]
    return np.array(rm, dtype=np.int64), np.array(keep_out, dtype=np.int64), list(rm), list(rm_t)


def _read_columns(path: Path, bounds: np.ndarray, shapes: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                  chunk_pairs: int) -> np.ndarray:
    """``block[rows][:, cols]`` of every pair (n_pairs, len(rows), len(cols)), streamed."""
    n = len(bounds) - 1
    out = np.empty((n, len(rows), len(cols)))
    nb = int(shapes[0, 0])
    with h5py.File(path, "r") as f:
        ent = f["entries"]
        for p0 in range(0, n, chunk_pairs):
            p1 = min(p0 + chunk_pairs, n)
            blk = np.asarray(ent[bounds[p0]:bounds[p1]]).reshape(p1 - p0, nb, nb)
            out[p0:p1] = blk[:, rows][:, :, cols]
    return out


def _segment_sum(values: np.ndarray, seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(keys, sums)``: the distinct values of ``seg`` and the sums of the ``values`` rows with
    each of them (``seg`` in any order). A sparse 0/1 matrix product: several times faster than
    ``np.add.reduceat`` / ``np.add.at`` on (n, nk, nk) blocks."""
    if len(seg) == 0:
        return seg, values[:0]
    keys, inv = np.unique(seg, return_inverse=True)
    n = len(seg)
    W = sp.csr_matrix((np.ones(n), (inv, np.arange(n))), shape=(len(keys), n))
    return keys, np.asarray(W @ values.reshape(n, -1)).reshape((len(keys),) + values.shape[1:])


def _row_capped_ranges(rows_per_item: np.ndarray, max_rows: int) -> list[tuple[int, int]]:
    """Consecutive ``[i0, i1)`` ranges of items whose ``rows_per_item`` sum to at most
    ``max_rows`` (at least one item per range)."""
    cum = np.concatenate([[0], np.cumsum(rows_per_item, dtype=np.int64)])
    n, out, i0 = len(rows_per_item), [], 0
    while i0 < n:
        i1 = int(np.searchsorted(cum, cum[i0] + max_rows, side="right")) - 1
        i1 = min(max(i1, i0 + 1), n)
        out.append((i0, i1))
        i0 = i1
    return out


def _grouped(idx: np.ndarray, key: np.ndarray, n_keys: int) -> tuple[np.ndarray, np.ndarray]:
    """``idx`` stable-sorted by ``key`` and the start of each key's run (n_keys + 1)."""
    order = np.argsort(key, kind="stable")
    return idx[order], np.searchsorted(key[order], np.arange(n_keys + 1))


def _expand(starts: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For items with runs ``[starts, starts + counts)``: the owner item of every expanded row
    and the row's position in the run."""
    owner = np.repeat(np.arange(len(counts)), counts)
    offs = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
    return owner, np.repeat(starts, counts) + offs


def _apply_E(V: np.ndarray, pairs: np.ndarray, index: _PairIndex, E: np.ndarray, e_idx: np.ndarray,
             max_rows: int) -> np.ndarray:
    """``(V E)_p = sum_e V_q E_e`` over the E pairs ``e = (Re, C', C)`` ending at the atom C of
    ``p = (R, i, C)``, with ``q = (R - Re, i, C')`` (dropped when not stored). Pairs are taken
    in ranges of at most ``max_rows`` expanded (p, e) rows."""
    out = np.zeros_like(V)
    e_sorted, starts = _grouped(e_idx, pairs[e_idx, 4], index.n_atoms)
    counts = np.diff(starts)
    for p0, p1 in _row_capped_ranges(counts[pairs[:, 4]], max_rows):
        p = np.arange(p0, p1)
        c = pairs[p, 4]
        owner, pos = _expand(starts[c], counts[c])
        p_rep = p[owner]
        e = e_sorted[pos]
        q = index.lookup(pairs[p_rep, :3] - pairs[e, :3], pairs[p_rep, 3], pairs[e, 3])
        ok = q >= 0
        contrib = np.einsum("nkm,nml->nkl", V[q[ok]], E[e[ok]])
        keys, sums = _segment_sum(contrib, p_rep[ok])
        out[keys] += sums
    return out


def neumann_X(pairs: np.ndarray, U: np.ndarray, Emm: np.ndarray, index: _PairIndex, n_terms: int,
              max_rows: int = 1 << 18) -> np.ndarray:
    """``X = U (I - E + E^2 - ...)`` with ``n_terms`` terms, on the stored pairs."""
    e_idx = np.flatnonzero(np.abs(Emm).reshape(len(Emm), -1).max(axis=1) > 0)
    X = U.copy()
    term = U
    for t in range(1, n_terms):
        term = _apply_E(term, pairs, index, Emm, e_idx, max_rows)
        X = X + (-1) ** t * term
    return X


def correction_blocks(pairs: np.ndarray, X: np.ndarray, U: np.ndarray, index: _PairIndex, out_idx: np.ndarray,
                      max_rows: int = 1 << 18) -> np.ndarray:
    """``corr(o) = sum_q X_p U_q^T`` (len(out_idx), nk, nk) for the output pairs ``o = (R, A, B)``
    of ``out_idx``: q = (Rq, B, C) runs over the pairs of B with a nonzero U block, p = (Rq + R,
    A, C) (a zero block when not stored). This is the term set of the sum over C in the module
    docstring (zero blocks add exact zeros).

    Outputs are grouped by B, which share the q list: for a group, the X blocks are gathered as
    ``G[(q, m), (o, k)]`` and ``corr`` is one GEMM ``U_B^T G``. ``max_rows`` caps the gathered
    (o, q) rows per GEMM (``nm * nk * 8`` B each, plus ~60 B of index arrays)."""
    n_pairs, nk, nm = U.shape
    Xt = np.zeros((n_pairs + 1, nm, nk))  # row n_pairs: the zero block of absent pairs
    Xt[:n_pairs] = np.swapaxes(X, 1, 2)
    nz_u = np.flatnonzero(np.abs(U).reshape(n_pairs, -1).max(axis=1) > 0)
    u_sorted, ustart = _grouped(nz_u, pairs[nz_u, 3], index.n_atoms)
    out_idx = np.asarray(out_idx, dtype=np.int64)
    o_sorted, ostart = _grouped(np.arange(len(out_idx)), pairs[out_idx, 4], index.n_atoms)
    result = np.zeros((len(out_idx), nk, nk))
    for B in range(index.n_atoms):
        q = u_sorted[ustart[B]:ustart[B + 1]]
        loc_all = o_sorted[ostart[B]:ostart[B + 1]]
        if len(q) == 0 or len(loc_all) == 0:
            continue
        UB = np.swapaxes(U[q], 1, 2).reshape(len(q) * nm, nk)
        Rq, Cq = pairs[q, :3], pairs[q, 4]
        step = max(1, max_rows // len(q))
        for a0 in range(0, len(loc_all), step):
            loc = loc_all[a0:a0 + step]
            o = out_idx[loc]
            # p[t, o] for q_t, o
            p = index.lookup(Rq[:, None, :] + pairs[o, :3][None, :, :], pairs[o, 3][None, :], Cq[:, None])
            p[p < 0] = n_pairs
            G = Xt[p]  # (n_q, n_o, nm, nk)
            G = np.swapaxes(G, 1, 2).reshape(len(q) * nm, len(o) * nk)
            M = UB.T @ G  # M[l, (o, k)] = corr(o)[k, l]
            result[loc] = np.swapaxes(M.reshape(nk, len(o), nk), 0, 1).swapaxes(1, 2)
    return result


def schur_correction(pairs: np.ndarray, U: np.ndarray, Emm: np.ndarray, n_atoms: int, n_terms: int = 2,
                     max_rows: int = 1 << 18) -> np.ndarray:
    """``sum_C X U^T`` on the stored pairs (n_pairs, nk, nk); see the module docstring.

    ``U`` (n_pairs, nk, nm), ``Emm`` (n_pairs, nm, nm): ``S_mm`` blocks minus the identity on
    the on-site pairs. ``n_terms``: number of terms of the Neumann series (1: S_mm = I)."""
    index = _PairIndex(pairs, n_atoms)
    X = neumann_X(pairs, U, Emm, index, n_terms, max_rows)
    return correction_blocks(pairs, X, U, index, np.arange(len(pairs)), max_rows)


def _hermitize_in_place(out: np.ndarray, pairs: np.ndarray, n_atoms: int, chunk: int) -> None:
    """``M(R) <- 0.5 (M(R) + M(-R)^T)`` (and its partner the transpose), as the production
    projection writes its output; blocks without a partner are symmetrized alone."""
    partner = _PairIndex(pairs, n_atoms).lookup(-pairs[:, :3], pairs[:, 4], pairs[:, 3])
    alone = np.flatnonzero((partner < 0) | (partner == np.arange(len(pairs))))
    first = np.flatnonzero(partner > np.arange(len(pairs)))
    for a0 in range(0, len(alone), chunk):
        p = alone[a0:a0 + chunk]
        out[p] = 0.5 * (out[p] + np.swapaxes(out[p], 1, 2))
    for a0 in range(0, len(first), chunk):
        p = first[a0:a0 + chunk]
        q = partner[p]
        avg = 0.5 * (out[p] + np.swapaxes(out[q], 1, 2))
        out[p] = avg
        out[q] = np.swapaxes(avg, 1, 2)


def project_overlap_rs(
    input_dir: Path | str,
    output_dir: Path | str,
    schur_plan: RemovalPlanLike,
    truncate_plan: RemovalPlanLike | None = None,
    n_terms: int = 2,
    chunk_pairs: int = 4096,
    write_dummy_hamiltonian: bool = True,
    max_rows: int = 1 << 18,
) -> dict[str, Any]:
    """Real-space path B: full-basis DeepH directory -> Schur-reduced, then truncated, overlap
    (same pairs, ``info.json`` as the production stages write it, ``POSCAR`` copied, zero
    ``hamiltonian.h5``).

    Memory: one (n_pairs, n_kept, n_kept) float64 array (the output; ~1.6 GB for a 216-atom
    tight-C cell, 234k pairs, 29 kept orbitals), the removed-orbital columns U and X
    (n_pairs, n_kept, n_removed each) and per-step temporaries bounded by ``max_rows`` (~0.1 GB
    at the default 2^18). The full-basis S is streamed. ``max_rows`` does not change the result."""
    t0 = time.monotonic()
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    info = read_info(input_dir)
    eom = info["elements_orbital_map"]
    st = read_poscar(input_dir / DEEPX_POSCAR_FILENAME)
    if len(set(st.species)) != 1:
        raise NotImplementedError("project_overlap_rs supports a single element")
    el = st.species[0]
    shells = [int(v) for v in eom[el]]
    nb = sum(2 * l + 1 for l in shells)
    rm, keep_out, rm_schur, rm_trunc = resolve_local_indices(shells, schur_plan, truncate_plan, el)
    n_atoms = len(st.species)

    s_path = input_dir / DEEPX_OVERLAP_FILENAME
    pairs, bounds, shapes = read_layout(s_path)
    if np.any(shapes != nb):
        raise ValueError("unexpected block shapes")
    U = _read_columns(s_path, bounds, shapes, keep_out, rm, chunk_pairs)
    Emm = _read_columns(s_path, bounds, shapes, rm, rm, chunk_pairs)
    onsite = (pairs[:, 3] == pairs[:, 4]) & ~pairs[:, :3].any(axis=1)
    Emm[onsite] -= np.eye(len(rm))
    out = schur_correction(pairs, U, Emm, n_atoms, n_terms=n_terms, max_rows=max_rows)
    del U, Emm

    # out = S_kk - correction, in place, streaming S_kk
    out *= -1.0
    nb_ = int(shapes[0, 0])
    with h5py.File(s_path, "r") as f:
        ent = f["entries"]
        for p0 in range(0, len(pairs), chunk_pairs):
            p1 = min(p0 + chunk_pairs, len(pairs))
            blk = np.asarray(ent[bounds[p0]:bounds[p1]]).reshape(p1 - p0, nb_, nb_)
            out[p0:p1] += blk[:, keep_out][:, :, keep_out]
    _hermitize_in_place(out, pairs, n_atoms, chunk_pairs)
    output_dir.mkdir(parents=True, exist_ok=True)
    nko = len(keep_out)
    shapes_o, bounds_o = pair_layout(pairs, np.full(n_atoms, nko))
    with h5py.File(output_dir / DEEPX_OVERLAP_FILENAME, "w") as f:
        f.create_dataset("atom_pairs", data=pairs)
        f.create_dataset("chunk_shapes", data=shapes_o)
        f.create_dataset("chunk_boundaries", data=bounds_o)
        f.create_dataset("entries", data=out.reshape(-1))
    if write_dummy_hamiltonian:
        with h5py.File(output_dir / "hamiltonian.h5", "w") as f:
            f.create_dataset("atom_pairs", data=pairs)
            f.create_dataset("chunk_shapes", data=shapes_o)
            f.create_dataset("chunk_boundaries", data=bounds_o)
            create_zero_dataset(f, "entries", (int(bounds_o[-1]),), np.float64)

    # info.json exactly as the two production stages leave it
    elements = list(st.species)
    rm_glob = [a * nb + int(k) for a in range(n_atoms) for k in rm_schur]
    map1 = build_reduced_elements_orbital_map(elements, eom, rm_glob)
    occ = reduce_occupation_value(info.get("occupation"), rm_glob, "schur", info.get("orbits_quantity"))
    n1 = nb - len(rm_schur)
    rm_glob_t = [a * n1 + int(k) for a in range(n_atoms) for k in rm_trunc]
    map2 = build_reduced_elements_orbital_map(elements, map1, rm_glob_t) if rm_trunc else map1
    new_info = {**info, "occupation": occ, "elements_orbital_map": map2, "orbits_quantity": nko * n_atoms}
    with open(output_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(new_info, f, indent=2)
        f.write("\n")
    (output_dir / DEEPX_POSCAR_FILENAME).write_bytes((input_dir / DEEPX_POSCAR_FILENAME).read_bytes())
    stats = {"n_pairs": len(pairs), "n_terms": n_terms, "kept_per_atom": nko,
             "seconds": time.monotonic() - t0}
    logger.info(f"[overlap] path B {input_dir} -> {output_dir}: {stats}")
    return stats
