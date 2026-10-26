from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from deepx_dock.compute.eigen.matrix_obj import AOMatrixObj


class SparseAOMatrixObj(AOMatrixObj):
    """
    Memory-efficient counterpart to ``AOMatrixObj``.

    ``AOMatrixObj`` eagerly densifies every R-block into a full
    ``(orbits_quantity, orbits_quantity)`` array, even though only a small
    fraction of atom-pair blocks are actually nonzero for a given R (only
    atoms within the interaction cutoff have entries). For large structures
    that dense allocation (``R_quantity * Nb^2``) is what OOMs.

    This subclass keeps the matrix in its on-disk sparse block form
    (``atom_pairs``/``chunk_boundaries``/``chunk_shapes``/``entries``, read
    via the inherited ``AOMatrixObj._read_h5``) and only ever builds a dense
    matrix for a *single* k-point at a time, on demand, via ``_dense_k``.
    Metadata parsing (``_parse_info``, ``_parse_poscar``, ``_parse_orbit_types``)
    is reused unchanged from ``AOMatrixObj`` since none of it touches ``mats``.
    """

    def __init__(
        self,
        info_dir_path,
        matrix_file_path=None,
        matrix_type="hamiltonian",
        load_matrix: bool = True,
        layout_from: SparseAOMatrixObj | None = None,
    ):
        """
        ``load_matrix=False`` skips reading and scattering ``entries`` (the
        actual matrix values -- the largest dataset in the file) entirely,
        keeping only ``atom_pairs``/``Rijk_list``/structural metadata. Use
        this when only the structure is needed (e.g. cross-validating an
        overlap-only run against ``hamiltonian.h5`` without ever needing the
        Hamiltonian's values). ``_dense_k`` is unavailable on such an
        instance and raises if called.

        ``layout_from``: an already loaded object of the same structure (e.g.
        another ensemble member). If its blocks have the same layout
        (``atom_pairs``, chunk boundaries and shapes, matrix type), its
        destination arrays are shared instead of rebuilt -- building them is
        most of the loading time -- and only this file's entries are
        permuted into place. Otherwise it is ignored.
        """
        self._get_necessary_data_path(info_dir_path, matrix_file_path, matrix_type)
        self.mats = None
        self.Rijk_list = None
        self._parse_info()
        self._parse_poscar()
        self._parse_orbit_types()
        self._load_sparse(matrix_type, load_matrix, layout_from)

    def _load_sparse(
        self, matrix_type: str, load_matrix: bool = True, layout_from: SparseAOMatrixObj | None = None
    ) -> None:
        is_overlap = matrix_type == "overlap"
        self._is_overlap = is_overlap
        dtype = np.complex128 if (not is_overlap and self.spinful) else np.float64

        if load_matrix:
            atom_pairs, bounds, shapes, entries = self._read_h5(self.matrix_path, dtype=dtype)
        else:
            atom_pairs, bounds, shapes = self._read_h5_structure_only(self.matrix_path)
            entries = None
        self.atom_pairs = atom_pairs
        self.bounds = bounds
        self.shapes = shapes

        if load_matrix and layout_from is not None and self._same_layout(layout_from):
            self._share_layout(layout_from, entries)
            return

        r_to_idx: dict[tuple[int, int, int], int] = {}
        for ap in atom_pairs:
            r = (int(ap[0]), int(ap[1]), int(ap[2]))
            if r not in r_to_idx:
                r_to_idx[r] = len(r_to_idx)
        rijk_list = np.array(list(r_to_idx.keys()), dtype=int)
        if len(rijk_list) > 0:
            tx, ty, tz = rijk_list[:, 0], rijk_list[:, 1], rijk_list[:, 2]
            rijk_list = rijk_list[np.lexsort((tx, ty, tz))]
        self.Rijk_list = rijk_list

        if not load_matrix:
            self._entry_row = None
            self._entry_col = None
            self._entry_r_index = None
            self._entries_expanded = None
            self._dense_lin = None
            self._dense_starts = None
            self._csr_indices = None
            self._csr_indptr = None
            self._entry_order = None
            self._mat_dim = int(self.orbits_quantity * (2 if self.spinful else 1))
            return

        r_to_idx_sorted = {tuple(int(v) for v in r): i for i, r in enumerate(rijk_list)}
        self._build_scatter_arrays(atom_pairs, shapes, entries, matrix_type, r_to_idx_sorted)

    @staticmethod
    def _read_h5_structure_only(h5_path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Like ``AOMatrixObj._read_h5`` but never reads the (large) ``entries`` dataset."""
        h5_path_obj = Path(h5_path)
        if not h5_path_obj.exists():
            raise FileNotFoundError(f"File not found: {h5_path}")

        with h5py.File(h5_path, "r") as f:
            atom_pairs = np.array(f["atom_pairs"][:], dtype=np.int64)
            boundaries = np.array(f["chunk_boundaries"][:], dtype=np.int64)
            shapes = np.array(f["chunk_shapes"][:], dtype=np.int64)

        return atom_pairs, boundaries, shapes

    def _build_scatter_arrays(
        self,
        atom_pairs: np.ndarray,
        shapes: np.ndarray,
        entries: np.ndarray,
        matrix_type: str,
        r_to_idx: dict[tuple[int, int, int], int],
    ) -> None:
        """
        Precompute, once, the (row, col, R-index) destination of every scalar
        in ``entries`` within the eventual dense (mat_dim, mat_dim) matrix at
        a given k-point. This mirrors exactly where
        ``AOMatrixObj._assemble_matrix_from_deeph_data`` would scatter each
        chunk when densifying (same up/up, up/dn, dn/up, dn/dn spin-block
        layout, same overlap block_diag(S, S) expansion when spinful), but
        records positions instead of writing into a dense buffer, and does
        this once for the whole R-stack rather than once per k-point.
        """
        is_overlap = matrix_type == "overlap"
        spinful = self.spinful
        cumsum = self.atom_num_orbits_cumsum
        orbits_quantity = self.orbits_quantity
        self._mat_dim = int(orbits_quantity * (2 if spinful else 1))

        row_chunks = []
        col_chunks = []
        for i_ap in range(len(atom_pairs)):
            i_atom = int(atom_pairs[i_ap, 3])
            j_atom = int(atom_pairs[i_ap, 4])
            n_i_shape, n_j_shape = int(shapes[i_ap, 0]), int(shapes[i_ap, 1])
            r_local = np.arange(n_i_shape)
            c_local = np.arange(n_j_shape)

            if is_overlap or not spinful:
                row_abs = cumsum[i_atom] + r_local
                col_abs = cumsum[j_atom] + c_local
            else:
                n_i = n_i_shape // 2
                n_j = n_j_shape // 2
                row_abs = np.where(
                    r_local < n_i,
                    cumsum[i_atom] + r_local,
                    cumsum[i_atom] + (r_local - n_i) + orbits_quantity,
                )
                col_abs = np.where(
                    c_local < n_j,
                    cumsum[j_atom] + c_local,
                    cumsum[j_atom] + (c_local - n_j) + orbits_quantity,
                )

            rr, cc = np.meshgrid(row_abs, col_abs, indexing="ij")
            row_chunks.append(rr.reshape(-1))
            col_chunks.append(cc.reshape(-1))

        entry_row = np.concatenate(row_chunks) if row_chunks else np.zeros(0, dtype=int)
        entry_col = np.concatenate(col_chunks) if col_chunks else np.zeros(0, dtype=int)

        pair_r_index = np.fromiter(
            (r_to_idx[(int(a), int(b), int(c))] for a, b, c in atom_pairs[:, :3]),
            dtype=np.int64,
            count=len(atom_pairs),
        )
        chunk_sizes = (shapes[:, 0] * shapes[:, 1]).astype(np.int64)
        entry_r_index = np.repeat(pair_r_index, chunk_sizes)
        entries_expanded = entries

        if is_overlap and spinful:
            # Mirrors AOMatrixObj's np.block([[S, 0], [0, S]]) expansion: every
            # overlap entry lands twice, once in each diagonal spin block.
            entry_row = np.concatenate([entry_row, entry_row + orbits_quantity])
            entry_col = np.concatenate([entry_col, entry_col + orbits_quantity])
            entry_r_index = np.concatenate([entry_r_index, entry_r_index])
            entries_expanded = np.concatenate([entries_expanded, entries_expanded])

        # Sort entries by destination (row, col) once, so every k-point can
        # sum duplicates (same (row, col), different R) with a single
        # np.add.reduceat over contiguous segments instead of going through a
        # scipy COO matrix (whose constructor re-checks all nnz indices on
        # every call). Row/col fit in int32, halving their footprint.
        dim = np.int64(self._mat_dim)
        lin = entry_row.astype(np.int64) * dim + entry_col
        order = np.argsort(lin, kind="stable")
        lin = lin[order]
        starts = np.flatnonzero(np.r_[True, lin[1:] != lin[:-1]]) if len(lin) else np.zeros(0, dtype=np.int64)

        self._entry_row = entry_row[order].astype(np.int32)
        self._entry_col = entry_col[order].astype(np.int32)
        self._entry_r_index = entry_r_index[order]
        self._entries_expanded = entries_expanded[order]
        # kept to place another file's entries of the same layout (see `_share_layout`)
        self._entry_order = order.astype(np.int32) if len(order) < np.iinfo(np.int32).max else order
        self._dense_lin = lin[starts]
        self._dense_starts = starts
        # The summed segments are in row-major (row, col) order, i.e. already a
        # canonical CSR layout: keep its column indices and row pointers so
        # `_sparse_k` skips the COO -> CSR conversion at every k-point.
        idx_dtype = np.int32 if len(starts) < np.iinfo(np.int32).max else np.int64
        self._csr_indices = (self._dense_lin % dim).astype(idx_dtype)
        self._csr_indptr = np.searchsorted(self._dense_lin // dim, np.arange(dim + 1)).astype(idx_dtype)

    _SHARED_LAYOUT = (
        "Rijk_list", "_mat_dim", "_entry_row", "_entry_col", "_entry_r_index", "_entry_order",
        "_dense_lin", "_dense_starts", "_csr_indices", "_csr_indptr",
    )

    def _same_layout(self, other: SparseAOMatrixObj) -> bool:
        """Whether `other`'s destination arrays are valid for this object's entries."""
        return (
            getattr(other, "_entries_expanded", None) is not None
            and other._is_overlap == self._is_overlap
            and other.spinful == self.spinful
            and np.array_equal(other.atom_num_orbits_cumsum, self.atom_num_orbits_cumsum)
            and np.array_equal(other.atom_pairs, self.atom_pairs)
            and np.array_equal(other.bounds, self.bounds)
            and np.array_equal(other.shapes, self.shapes)
        )

    def _share_layout(self, other: SparseAOMatrixObj, entries: np.ndarray) -> None:
        """Take `other`'s (read-only) destination arrays and sort this file's entries like it."""
        for name in self._SHARED_LAYOUT:
            setattr(self, name, getattr(other, name))
        if self._is_overlap and self.spinful:
            # same block_diag(S, S) expansion as `_build_scatter_arrays`
            entries = np.concatenate([entries, entries])
        self._entries_expanded = entries[self._entry_order]

    @property
    def fill_fraction(self) -> float:
        """Fraction of the (mat_dim, mat_dim) entries of M(k) that are structurally nonzero."""
        return len(self._dense_lin) / self._mat_dim ** 2

    def _weighted_k(self, k: np.ndarray) -> np.ndarray:
        """Phase-weighted entries at one fractional k-point, in stored (destination-sorted) order."""
        if self._entries_expanded is None:
            raise AttributeError(
                "This SparseAOMatrixObj was constructed with load_matrix=False (structure/metadata "
                "only, e.g. for overlap-only cross-validation) -- its matrix values were never read."
            )
        phase_per_r = np.exp(2j * np.pi * (self.Rijk_list.astype(np.float64) @ np.asarray(k, dtype=np.float64)))
        weighted = phase_per_r[self._entry_r_index]
        weighted *= self._entries_expanded
        return weighted

    def _coo_k(self, k: np.ndarray) -> coo_matrix:
        """Phase-weighted (mat_dim, mat_dim) COO matrix at one fractional k-point (duplicates not yet summed)."""
        weighted = self._weighted_k(k)
        return coo_matrix((weighted, (self._entry_row, self._entry_col)), shape=(self._mat_dim, self._mat_dim))

    def _dense_k(self, k: np.ndarray) -> np.ndarray:
        """Build the dense (mat_dim, mat_dim) matrix at one fractional k-point."""
        weighted = self._weighted_k(k)
        out = np.zeros(self._mat_dim * self._mat_dim, dtype=np.complex128)
        if len(weighted):
            out[self._dense_lin] = np.add.reduceat(weighted, self._dense_starts)
        return out.reshape(self._mat_dim, self._mat_dim)

    def _sparse_k(self, k: np.ndarray) -> csr_matrix:
        """
        Build the (mat_dim, mat_dim) matrix at one fractional k-point as a
        scipy CSR matrix, never allocating the dense array. Contributions
        from different R vectors landing on the same (row, col) are summed,
        exactly as in ``_dense_k``.
        """
        weighted = self._weighted_k(k)
        data = np.add.reduceat(weighted, self._dense_starts) if len(weighted) else weighted
        return csr_matrix((data, self._csr_indices, self._csr_indptr), shape=(self._mat_dim, self._mat_dim))
