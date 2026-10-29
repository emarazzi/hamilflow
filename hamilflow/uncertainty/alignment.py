"""Rigid-shift alignment of sparse real-space Hamiltonians.

For a fixed overlap S, H - c S has the eigenvalues of H shifted by exactly -c and the same
eigenvectors. Predicted Hamiltonians carry an absolute energy offset that differs from model to
model (and from DFT), so before comparing two Hamiltonians that share S, the offset is removed
with the shift that best explains their difference,

    c = Re<H - H_ref, S> / <S, S>      (sums over all stored real-space entries),

and the relative residual |H - H_ref - c S| / |H - H_ref| measures how much of the difference
is not a rigid shift (0: purely rigid).

The files are the DeepH sparse format (`atom_pairs`, `chunk_boundaries`, `chunk_shapes`,
`entries`). Entries are streamed in batches of whole atom-pair blocks, so memory stays bounded.
H and S share `atom_pairs`; for spinful H (complex, 2x2 spin blocks) S is expanded onto the two
diagonal spin blocks, as `SparseHamiltonianObj` does.
"""

import concurrent.futures
from collections.abc import Iterator, Sequence
from pathlib import Path

import h5py
import numpy as np


def read_layout(path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(atom_pairs, chunk_boundaries, chunk_shapes)` of a sparse matrix file, never `entries`."""
    with h5py.File(path, "r") as f:
        return (
            np.array(f["atom_pairs"][:], dtype=np.int64),
            np.array(f["chunk_boundaries"][:], dtype=np.int64),
            np.array(f["chunk_shapes"][:], dtype=np.int64),
        )


def pair_batches(bounds: np.ndarray, batch_entries: int) -> Iterator[tuple[int, int]]:
    """Ranges `[p0, p1)` of atom pairs holding about `batch_entries` entries each (at least one
    pair per range)."""
    n_pairs = len(bounds) - 1
    p0 = 0
    while p0 < n_pairs:
        p1 = int(np.searchsorted(bounds, bounds[p0] + batch_entries, side="right")) - 1
        p1 = min(max(p1, p0 + 1), n_pairs)
        yield p0, p1
        p0 = p1


def overlap_like(s_entries: np.ndarray, s_shapes: np.ndarray, h_shapes: np.ndarray) -> np.ndarray:
    """S entries arranged like H's blocks: unchanged when the block shapes match, expanded onto
    the two diagonal spin blocks when H's blocks are twice as large (spinful H)."""
    if np.array_equal(s_shapes, h_shapes):
        return s_entries
    out, k = [], 0
    for (n, m), (hn, hm) in zip(s_shapes, h_shapes):
        block = s_entries[k:k + n * m].reshape(n, m)
        k += n * m
        if (hn, hm) != (2 * n, 2 * m):
            raise ValueError(f"H block {hn}x{hm} is neither the S block {n}x{m} nor its spin expansion")
        big = np.zeros((hn, hm), dtype=block.dtype)
        big[:n, :m] = block
        big[n:, m:] = block
        out.append(big.reshape(-1))
    return np.concatenate(out) if out else np.zeros(0)


class SparseStream:
    """Stream several Hamiltonians with the same sparse layout, together with their overlap in
    the Hamiltonians' layout.

    Iterating yields `(p0, p1, hs, s)`: the atom-pair range, the list of entry batches (one per
    path, in order) and the matching overlap batch.
    """

    def __init__(self, h_paths: Sequence[Path | str], s_path: Path | str,
                 batch_entries: int = 4_000_000, n_threads: int = 1):
        self.h_paths = [Path(p) for p in h_paths]
        self.s_path = Path(s_path)
        self.batch_entries = batch_entries
        self.n_threads = max(1, n_threads)
        self.atom_pairs, self.bounds, self.shapes = read_layout(self.h_paths[0])
        for p in self.h_paths[1:]:
            ap, _, sh = read_layout(p)
            if not (np.array_equal(ap, self.atom_pairs) and np.array_equal(sh, self.shapes)):
                raise ValueError(f"sparse layout of {p} differs from {self.h_paths[0]}")
        s_ap, self.s_bounds, self.s_shapes = read_layout(self.s_path)
        if not np.array_equal(s_ap, self.atom_pairs):
            raise ValueError(f"atom_pairs of {self.s_path} differ from those of {self.h_paths[0]}")

    @property
    def n_entries(self) -> int:
        return int(self.bounds[-1])

    def __iter__(self):
        h_files = [h5py.File(p, "r") for p in self.h_paths]
        s_file = h5py.File(self.s_path, "r")
        ex = concurrent.futures.ThreadPoolExecutor(self.n_threads) if self.n_threads > 1 and len(h_files) > 1 else None
        try:
            for p0, p1 in pair_batches(self.bounds, self.batch_entries):
                e0, e1 = self.bounds[p0], self.bounds[p1]
                read = lambda f: np.asarray(f["entries"][e0:e1])  # noqa: B023 (used within the iteration)
                hs = list(ex.map(read, h_files)) if ex else [read(f) for f in h_files]
                s = overlap_like(np.asarray(s_file["entries"][self.s_bounds[p0]:self.s_bounds[p1]]),
                                 self.s_shapes[p0:p1], self.shapes[p0:p1])
                yield p0, p1, hs, s
        finally:
            if ex:
                ex.shutdown()
            for f in h_files + [s_file]:
                f.close()


def fit_rigid_shifts(
    h_paths: Sequence[Path | str],
    ref_path: Path | str | None,
    s_path: Path | str,
    batch_entries: int = 4_000_000,
    n_threads: int = 1,
) -> list[dict]:
    """Rigid shift of every Hamiltonian in `h_paths` relative to `ref_path` (or, if None, to the
    mean of `h_paths`), all sharing the overlap `s_path`.

    Returns one dict per path: `c_eV` (subtract `c_eV * S` to align), `residual` (relative norm of
    what the shift does not explain), `rms_before_eV` and `rms_after_eV` (per stored entry).
    """
    paths = list(h_paths) + ([ref_path] if ref_path is not None else [])
    n = len(h_paths)
    ds, dd = np.zeros(n), np.zeros(n)
    ss, n_entries = 0.0, 0
    for _, _, hs, s in SparseStream(paths, s_path, batch_entries, n_threads):
        ref = hs[n] if ref_path is not None else np.mean(hs, axis=0)
        ss += float(np.sum(s * s))
        n_entries += s.size
        for k in range(n):
            d = hs[k] - ref
            ds[k] += float(np.sum(np.real(d) * s))
            dd[k] += float(np.sum(np.abs(d) ** 2))
    out = []
    for k in range(n):
        c = ds[k] / ss if ss > 0 else 0.0
        dd_after = max(dd[k] - 2 * c * ds[k] + c * c * ss, 0.0)
        out.append({
            "c_eV": float(c),
            "residual": float(np.sqrt(dd_after / dd[k])) if dd[k] > 0 else 0.0,
            "rms_before_eV": float(np.sqrt(dd[k] / n_entries)) if n_entries else 0.0,
            "rms_after_eV": float(np.sqrt(dd_after / n_entries)) if n_entries else 0.0,
        })
    return out


def write_aligned_hamiltonian(
    h_path: Path | str,
    s_path: Path | str,
    c: float,
    out_path: Path | str,
    aligned_to: str = "",
    batch_entries: int = 4_000_000,
) -> Path:
    """Write `H - c S` to `out_path` in the same sparse layout. The shift and the reference are
    stored as attributes (`rigid_shift_eV`, `aligned_to`, `source`), so a stale copy can be
    recognized."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stream = SparseStream([h_path], s_path, batch_entries)
    with h5py.File(h_path, "r") as src, h5py.File(out_path, "w") as dst:
        for key in ("atom_pairs", "chunk_boundaries", "chunk_shapes"):
            dst.create_dataset(key, data=src[key][:])
        entries = dst.create_dataset("entries", shape=src["entries"].shape, dtype=src["entries"].dtype)
        for p0, p1, (h,), s in stream:
            entries[stream.bounds[p0]:stream.bounds[p1]] = h - c * s
        dst.attrs["rigid_shift_eV"] = float(c)
        dst.attrs["aligned_to"] = str(aligned_to)
        dst.attrs["source"] = str(Path(h_path).resolve())
    return out_path


def read_poscar(structure_dir: Path | str) -> tuple[np.ndarray, list[str], np.ndarray]:
    """`(lattice, species, cartesian_positions)` from a VASP-5 POSCAR (Angstrom)."""
    lines = [l.strip() for l in (Path(structure_dir) / "POSCAR").read_text().splitlines()]
    scale = float(lines[1].split()[0])
    lattice = scale * np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)])
    names, counts = lines[5].split(), [int(x) for x in lines[6].split()]
    species = [nm for nm, c in zip(names, counts) for _ in range(c)]
    i = 7
    if lines[i][0] in "sS":  # selective dynamics
        i += 1
    cartesian = lines[i][0] in "cCkK"
    pos = np.array([[float(x) for x in lines[i + 1 + a].split()[:3]] for a in range(len(species))])
    pos = scale * pos if cartesian else pos @ lattice
    return lattice, species, pos


def pair_distances(structure_dir: Path | str, atom_pairs: np.ndarray) -> np.ndarray:
    """Distance |R + x_j - x_i| (Angstrom) of every atom pair `(R1, R2, R3, i, j)`; 0 on-site."""
    lattice, _, pos = read_poscar(structure_dir)
    shift = atom_pairs[:, :3] @ lattice
    return np.linalg.norm(shift + pos[atom_pairs[:, 4]] - pos[atom_pairs[:, 3]], axis=1)


__all__ = [
    "SparseStream",
    "fit_rigid_shifts",
    "overlap_like",
    "pair_batches",
    "pair_distances",
    "read_layout",
    "read_poscar",
    "write_aligned_hamiltonian",
]
