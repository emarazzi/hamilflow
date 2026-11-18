"""Full-basis overlap of a structure from two-centre tables, written as a converted DeepH directory.

Pairs are those FHI-aims would store (see `tables`): every ``(R, i, j)`` with
``|tau_j + R . lattice - tau_i| < pair_cutoff``, both orientations, plus the on-site blocks.
Blocks are evaluated chunk by chunk (spline -> bond-frame block -> rotation) and streamed into
``overlap.h5``; a zero ``hamiltonian.h5`` with the same layout, ``info.json`` and ``POSCAR`` make
the directory a drop-in for the production ``deeph_full`` overlap-only data (input of
`hamilflow.projection.run_projection`).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from ase.neighborlist import primitive_neighbor_list

from .io import DeephStructure, OverlapWriter, prepare_structure, write_info, write_poscar
from .tables import TwoCentreTable

logger = logging.getLogger(__name__)


def neighbour_pairs(structure: DeephStructure, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """``atom_pairs`` rows ``(R1, R2, R3, i, j)`` (n, 5) with bond length ``< cutoff`` plus the
    on-site rows, sorted by ``(R, i, j)``, and their bond vectors (n, 3) in Å."""
    n = len(structure.species)
    i, j, D, S = primitive_neighbor_list(
        "ijDS", [True, True, True], structure.lattice, structure.positions, cutoff + 1e-6,
        self_interaction=False, use_scaled_positions=False,
    )
    keep = np.linalg.norm(D, axis=1) < cutoff
    i, j, D, S = i[keep], j[keep], D[keep], S[keep]
    pairs = np.concatenate([np.column_stack([S, i, j]),
                            np.column_stack([np.zeros((n, 3), dtype=np.int64), np.arange(n), np.arange(n)])])
    vecs = np.concatenate([D, np.zeros((n, 3))])
    order = np.lexsort(pairs.T[::-1])
    return pairs[order].astype(np.int64), vecs[order]


def assemble_overlap(
    structure: DeephStructure,
    table: TwoCentreTable,
    out_dir: Path | str,
    chunk_pairs: int = 4096,
    xp: Any = np,
    near_cutoff_tol: float = 1e-4,
) -> dict[str, Any]:
    """Write ``overlap.h5``, a zero ``hamiltonian.h5``, ``info.json`` and ``POSCAR`` of
    ``structure`` (already in the conversion's form, see `io.prepare_structure`) to ``out_dir``.
    Single element only (the table's)."""
    t0 = time.monotonic()
    el = table.element_a
    if table.element_b != el or set(structure.species) != {el}:
        raise NotImplementedError(f"assemble_overlap supports one element ({el}) only, "
                                  f"got {sorted(set(structure.species))}")
    if el not in table.onsite:
        raise ValueError(f"table has no on-site block for {el}")
    out_dir = Path(out_dir)
    pairs, vecs = neighbour_pairs(structure, table.pair_cutoff + 2 * near_cutoff_tol)
    dist = np.linalg.norm(vecs, axis=1)
    near = np.abs(dist - table.pair_cutoff) < near_cutoff_tol
    if near.any():
        logger.warning(f"[overlap] {out_dir}: {int(near.sum())} pair(s) within {near_cutoff_tol} Å of the pair "
                       f"cutoff {table.pair_cutoff:.5f} Å; the pair set may differ from FHI-aims' there "
                       "(those blocks are ~0 either way)")
    keep = dist < table.pair_cutoff
    pairs, vecs, dist = pairs[keep], vecs[keep], dist[keep]
    onsite = dist == 0
    if np.any(onsite & (pairs[:, 3] != pairs[:, 4])):
        raise ValueError("two atoms at the same position")
    off_dist = dist[~onsite]
    if off_dist.size and off_dist.min() < table.r[0]:
        raise ValueError(f"shortest distance {off_dist.min():.4f} Å is below the table range "
                         f"({table.r[0]:.4f} Å)")
    n_orb = int(sum(2 * l + 1 for l in table.shells_a))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_poscar(out_dir / "POSCAR", structure)
    write_info(out_dir, structure, {el: list(table.shells_a)})
    onsite_block = xp.asarray(table.onsite[el])
    with OverlapWriter(out_dir, pairs, np.full(len(structure.species), n_orb)) as w:
        for p0 in range(0, len(pairs), chunk_pairs):
            p1 = min(p0 + chunk_pairs, len(pairs))
            blocks = xp.empty((p1 - p0, n_orb, n_orb), dtype=xp.float64)
            on = onsite[p0:p1]
            if on.any():
                blocks[xp.asarray(np.flatnonzero(on))] = onsite_block
            if (~on).any():
                idx = np.flatnonzero(~on)
                blocks[xp.asarray(idx)] = table.blocks(vecs[p0:p1][idx], xp=xp)
            w.write(p0, np.asarray(blocks.get() if hasattr(blocks, "get") else blocks))
    stats = {
        "n_atoms": len(structure.species),
        "n_pairs": len(pairs),
        "min_distance": float(off_dist.min()) if off_dist.size else None,
        "seconds": time.monotonic() - t0,
    }
    logger.info(f"[overlap] {out_dir}: {stats['n_pairs']} pairs, {stats['seconds']:.1f} s")
    return stats


def assemble_from_ase(atoms, table: TwoCentreTable, out_dir: Path | str, **kwargs: Any) -> dict[str, Any]:
    """`assemble_overlap` for an ASE ``Atoms`` (periodic), mapped as the conversion does."""
    st = prepare_structure(atoms.cell.array, atoms.get_positions(), atoms.get_chemical_symbols())
    return assemble_overlap(st, table, out_dir, **kwargs)
