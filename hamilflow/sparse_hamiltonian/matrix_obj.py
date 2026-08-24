from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix

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

    def __init__(self, info_dir_path, matrix_file_path=None, matrix_type="hamiltonian"):
        self._get_necessary_data_path(info_dir_path, matrix_file_path, matrix_type)
        self.mats = None
        self.Rijk_list = None
        self._parse_info()
        self._parse_poscar()
        self._parse_orbit_types()
        self._load_sparse(matrix_type)

    def _load_sparse(self, matrix_type: str) -> None:
        is_overlap = matrix_type == "overlap"
        dtype = np.complex128 if (not is_overlap and self.spinful) else np.float64
        atom_pairs, bounds, shapes, entries = self._read_h5(self.matrix_path, dtype=dtype)
        self.atom_pairs = atom_pairs
        self.bounds = bounds
        self.shapes = shapes

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
        r_to_idx_sorted = {tuple(int(v) for v in r): i for i, r in enumerate(rijk_list)}

        self._build_scatter_arrays(atom_pairs, shapes, entries, matrix_type, r_to_idx_sorted)

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

        self._entry_row = entry_row
        self._entry_col = entry_col
        self._entry_r_index = entry_r_index
        self._entries_expanded = entries_expanded

    def _dense_k(self, k: np.ndarray) -> np.ndarray:
        """Build the dense (mat_dim, mat_dim) matrix at one fractional k-point."""
        phase_per_r = np.exp(2j * np.pi * (self.Rijk_list.astype(np.float64) @ np.asarray(k, dtype=np.float64)))
        weighted = phase_per_r[self._entry_r_index] * self._entries_expanded
        return coo_matrix(
            (weighted, (self._entry_row, self._entry_col)), shape=(self._mat_dim, self._mat_dim)
        ).toarray()
