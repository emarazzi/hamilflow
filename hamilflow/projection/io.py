from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from deepx_dock.CONSTANT import DEEPX_INFO_FILENAME


def build_reduced_elements_orbital_map(
    elements: list[str],
    elements_orbital_map: dict[str, list[int]],
    removed_indices: list[int],
) -> dict[str, list[int]]:
    """
    Build updated elements_orbital_map after shell/channel removal.

    This requires that all atoms of the same element keep the same shell list,
    otherwise info.json cannot represent the reduced basis consistently.
    """
    rm_set = set(int(i) for i in removed_indices)
    per_element_kept_shells: dict[str, list[int]] = {}
    g0 = 0

    for el in elements:
        shell_ls = [int(v) for v in elements_orbital_map[el]]
        kept_shells_for_atom: list[int] = []
        for l in shell_ls:
            dim = 2 * l + 1
            rng = set(range(g0, g0 + dim))
            kept_count = len(rng - rm_set)
            if kept_count not in (0, dim):
                raise ValueError(
                    f"Partial removal inside shell is not supported for info.json rewrite: "
                    f"element={el}, l={l}, global_range=[{g0}, {g0 + dim})"
                )
            if kept_count == dim:
                kept_shells_for_atom.append(l)
            g0 += dim

        if el in per_element_kept_shells:
            if per_element_kept_shells[el] != kept_shells_for_atom:
                raise ValueError(
                    f"Inconsistent reduced shells among atoms of element {el}. "
                    "Use element-wide consistent selectors for info.json compatibility."
                )
        else:
            per_element_kept_shells[el] = kept_shells_for_atom

    return per_element_kept_shells


def reduce_occupation_value(
    occupation: object,
    removed_indices: list[int],
    reduction_mode: str,
    original_orbits_quantity: int | None,
) -> object:
    """
    Reduce the occupation metadata after orbitals are removed.

    Occupation is treated as a scalar total. Schur reduction removes occupied
    valence states, so subtract two electrons per removed orbital. Truncation
    keeps the occupation unchanged.
    """
    if occupation is None:
        return None

    if not isinstance(occupation, (int, float, np.integer, np.floating)):
        raise ValueError("Unsupported occupation format in info.json. Expected a scalar total.")

    if reduction_mode == "truncate":
        return occupation
    if reduction_mode != "schur":
        raise ValueError(f"Unsupported reduction_mode '{reduction_mode}'. Expected 'schur' or 'truncate'.")

    removed_count = len(set(int(i) for i in removed_indices))
    reduced = float(occupation) - 2 * removed_count
    if reduced.is_integer():
        return int(reduced)
    return reduced


def write_reduced_info_json(
    input_dir: Path,
    output_dir: Path,
    elements: list[str],
    removed_indices: list[int],
    reduction_mode: str,
) -> Path:
    """Write updated info.json matching the reduced basis and return the output path."""
    with open(input_dir / DEEPX_INFO_FILENAME, "r", encoding="utf-8") as fr:
        raw_info = json.load(fr)

    raw_map = raw_info["elements_orbital_map"]
    new_map = build_reduced_elements_orbital_map(elements, raw_map, removed_indices)

    raw_info["occupation"] = reduce_occupation_value(
        raw_info.get("occupation"),
        removed_indices,
        reduction_mode,
        raw_info.get("orbits_quantity"),
    )
    new_orbits = int(sum(np.sum(2 * np.array(new_map[el], dtype=int) + 1) for el in elements))
    raw_info["elements_orbital_map"] = new_map
    raw_info["orbits_quantity"] = new_orbits

    out_path = output_dir / DEEPX_INFO_FILENAME
    with open(out_path, "w", encoding="utf-8") as fw:
        json.dump(raw_info, fw, indent=2)
        fw.write("\n")
    return out_path


def dump_reduced_matrix_h5(
    out_path: Path,
    mats_R: np.ndarray,
    Rijk_list: np.ndarray | None,
    atom_pairs: np.ndarray,
    atom_num_orbits_cumsum: np.ndarray,
    keep_global: list[int],
    zero_entries: bool = False,
) -> Path:
    """
    Dump reduced-basis real-space matrices to DeepH-like h5.

    Unlike the default serializer, this writer recomputes chunk shapes using
    the kept orbital indices per atom, so reduced basis dimensions are encoded
    correctly for each atom pair block.

    With ``zero_entries=True`` the values of ``mats_R`` are ignored (only its
    block layout is used) and ``entries`` is written as an all-zero dataset that
    takes no disk space (see :func:`create_zero_dataset`). Used for dummy
    Hamiltonians.
    """
    if Rijk_list is None:
        raise ValueError("Rijk_list is None")

    keep_arr = np.array(sorted(set(int(i) for i in keep_global)), dtype=int)

    r_to_idx = {tuple(int(v) for v in r): i for i, r in enumerate(Rijk_list)}
    atom_pairs = np.asarray(atom_pairs, dtype=np.int64)

    per_atom_reduced_indices = []
    n_atoms = len(atom_num_orbits_cumsum) - 1
    for ia in range(n_atoms):
        a0 = int(atom_num_orbits_cumsum[ia])
        a1 = int(atom_num_orbits_cumsum[ia + 1])
        idx = np.where((keep_arr >= a0) & (keep_arr < a1))[0]
        per_atom_reduced_indices.append(idx)

    entries_chunks = []
    chunk_shapes = np.zeros((len(atom_pairs), 2), dtype=np.int64)
    chunk_boundaries = np.zeros(len(atom_pairs) + 1, dtype=np.int64)

    for i_ap, ap in enumerate(atom_pairs):
        Rijk = (int(ap[0]), int(ap[1]), int(ap[2]))
        ia = int(ap[3])
        ja = int(ap[4])

        ii = per_atom_reduced_indices[ia]
        jj = per_atom_reduced_indices[ja]
        chunk_shapes[i_ap] = (len(ii), len(jj))
        chunk_boundaries[i_ap + 1] = chunk_boundaries[i_ap] + len(ii) * len(jj)
        if not zero_entries:
            mat_R = mats_R[r_to_idx[Rijk]]
            entries_chunks.append(mat_R[np.ix_(ii, jj)].reshape(-1))

    with h5py.File(out_path, "w") as f:
        f.create_dataset("atom_pairs", data=atom_pairs)
        f.create_dataset("chunk_shapes", data=chunk_shapes)
        f.create_dataset("chunk_boundaries", data=chunk_boundaries)
        if zero_entries:
            create_zero_dataset(f, "entries", (int(chunk_boundaries[-1]),), np.float64)
        else:
            if entries_chunks:
                entries = np.concatenate(entries_chunks)
            else:
                entries = np.array([], dtype=mats_R.dtype)
            f.create_dataset("entries", data=np.real(entries).astype(np.float64))

    return out_path


@dataclass(frozen=True)
class ReducedOutputLayout:
    """
    Sparse layout of the reduced-basis entries written to a DeepH-like h5.

    Instead of a dense (N_R, Nb_kept, Nb_kept) real-space array, the
    projection only accumulates the scalars that end up in ``entries``
    (plus their Hermitian partners). Each such scalar is identified by a key
    ``r * nkp**2 + a * nkp + b`` (``r`` indexes ``Rijk_list``, ``a``/``b``
    are reduced-basis orbital indices).

    The targets are the sorted union of written entries and their partners,
    so they are grouped by R: entries for R index ``r`` are
    ``target_flat[r_bounds[r]:r_bounds[r + 1]]`` (flat ``a * nkp + b`` index
    inside the reduced R-block, ascending). ``out_pos``/``partner_pos`` map
    every written entry (in file order) and its Hermitian partner into the
    targets. Index arrays are int32 whenever the values fit, since for large
    structures they are as long as the output files.
    """

    atom_pairs: np.ndarray
    chunk_shapes: np.ndarray
    chunk_boundaries: np.ndarray
    nkp: int
    target_flat: np.ndarray
    r_bounds: np.ndarray
    out_pos: np.ndarray
    partner_pos: np.ndarray

    def hermitize_real(self, acc: np.ndarray) -> np.ndarray:
        """Written entries of Re[0.5 * (M(R) + M(-R)^dagger)] from real accumulated targets."""
        return 0.5 * (acc[self.out_pos] + acc[self.partner_pos])

    def hermitize_imag(self, acc: np.ndarray) -> np.ndarray:
        """Written entries of Im[0.5 * (M(R) + M(-R)^dagger)] from imaginary accumulated targets."""
        return 0.5 * (acc[self.out_pos] - acc[self.partner_pos])


def _index_dtype(max_value: int) -> type:
    return np.int32 if max_value < np.iinfo(np.int32).max else np.int64


def build_reduced_output_layout(
    Rijk_list: np.ndarray,
    atom_pairs: np.ndarray,
    atom_num_orbits_cumsum: np.ndarray,
    keep_global: list[int],
) -> ReducedOutputLayout:
    """
    Sparse equivalent of the block selection done by :func:`dump_reduced_matrix_h5`
    combined with :func:`hermitize_real_space_blocks`.

    The written entries are exactly those ``dump_reduced_matrix_h5`` would
    extract from a dense ``mats_R``. Hermitization maps entry (R, a, b) to
    0.5 * (M(R)[a, b] + conj(M(-R)[b, a])), or to the same formula with -R
    replaced by R when -R is absent from ``Rijk_list``; the partner
    (-R, b, a) is added to the targets even when it is not itself written.
    """
    if Rijk_list is None:
        raise ValueError("Rijk_list is None")

    keep_arr = np.array(sorted(set(int(i) for i in keep_global)), dtype=np.int64)
    nkp = len(keep_arr)
    nkp2 = np.int64(nkp) * np.int64(nkp)
    n_r = len(Rijk_list)

    r_to_idx = {tuple(int(v) for v in r): i for i, r in enumerate(Rijk_list)}
    atom_pairs = np.asarray(atom_pairs, dtype=np.int64)

    per_atom_reduced_indices = []
    n_atoms = len(atom_num_orbits_cumsum) - 1
    for ia in range(n_atoms):
        a0 = int(atom_num_orbits_cumsum[ia])
        a1 = int(atom_num_orbits_cumsum[ia + 1])
        per_atom_reduced_indices.append(np.where((keep_arr >= a0) & (keep_arr < a1))[0].astype(np.int64))

    chunk_shapes = np.zeros((len(atom_pairs), 2), dtype=np.int64)
    chunk_boundaries = np.zeros(len(atom_pairs) + 1, dtype=np.int64)
    key_chunks = []
    for i_ap, ap in enumerate(atom_pairs):
        i_r = r_to_idx[(int(ap[0]), int(ap[1]), int(ap[2]))]
        ii = per_atom_reduced_indices[int(ap[3])]
        jj = per_atom_reduced_indices[int(ap[4])]
        chunk_shapes[i_ap] = (len(ii), len(jj))
        chunk_boundaries[i_ap + 1] = chunk_boundaries[i_ap] + len(ii) * len(jj)
        key_chunks.append((i_r * nkp2 + ii[:, None] * nkp + jj[None, :]).reshape(-1))

    out_keys = np.concatenate(key_chunks) if key_chunks else np.zeros(0, dtype=np.int64)

    neg_idx = np.arange(n_r, dtype=np.int64)
    for r, i_r in r_to_idx.items():
        i_neg = r_to_idx.get((-r[0], -r[1], -r[2]))
        if i_neg is not None:
            neg_idx[i_r] = i_neg

    # partner of (r, a, b) is (neg_idx[r], b, a); built in place to limit temporaries.
    r_of = out_keys // nkp2
    rem = out_keys - r_of * nkp2
    partner_keys = neg_idx[r_of]
    del r_of
    partner_keys *= nkp2
    a_of = rem // nkp
    rem -= a_of * nkp
    rem *= nkp
    partner_keys += rem
    del rem
    partner_keys += a_of
    del a_of

    # Sorted union of written entries and partners (np.union1d is several
    # times slower here: numpy's unique is hash-based for large int arrays).
    target_keys = np.concatenate([out_keys, partner_keys])
    target_keys.sort()
    if len(target_keys):
        target_keys = target_keys[np.r_[True, target_keys[1:] != target_keys[:-1]]]
    pos_dtype = _index_dtype(len(target_keys))
    out_pos = np.searchsorted(target_keys, out_keys).astype(pos_dtype)
    del out_keys
    partner_pos = np.searchsorted(target_keys, partner_keys).astype(pos_dtype)
    del partner_keys
    r_bounds = np.searchsorted(target_keys, np.arange(n_r + 1, dtype=np.int64) * nkp2)
    target_keys %= nkp2
    target_flat = target_keys.astype(_index_dtype(int(nkp2)))
    del target_keys

    return ReducedOutputLayout(
        atom_pairs=atom_pairs,
        chunk_shapes=chunk_shapes,
        chunk_boundaries=chunk_boundaries,
        nkp=nkp,
        target_flat=target_flat,
        r_bounds=r_bounds,
        out_pos=out_pos,
        partner_pos=partner_pos,
    )


def dump_reduced_entries_h5(
    out_path: Path,
    layout: ReducedOutputLayout,
    entries: np.ndarray | None,
) -> Path:
    """
    Write already-extracted reduced entries (file order, see
    :class:`ReducedOutputLayout`) to a DeepH-like h5. ``entries=None`` writes
    an all-zero ``entries`` dataset that takes no disk space (dummy Hamiltonians).
    """
    with h5py.File(out_path, "w") as f:
        f.create_dataset("atom_pairs", data=layout.atom_pairs)
        f.create_dataset("chunk_shapes", data=layout.chunk_shapes)
        f.create_dataset("chunk_boundaries", data=layout.chunk_boundaries)
        if entries is None:
            create_zero_dataset(f, "entries", (int(layout.chunk_boundaries[-1]),), np.float64)
        else:
            f.create_dataset("entries", data=np.asarray(entries, dtype=np.float64))
    return out_path


def create_zero_dataset(
    group: h5py.Group, name: str, shape: tuple[int, ...], dtype: np.dtype | type | str
) -> h5py.Dataset:
    """
    Create an all-zero dataset that occupies (almost) no disk space.

    The dataset is chunked with a zero fill value and no data is written, so
    HDF5 never allocates its chunks: readers get zeros with the right shape and
    dtype, but the file only stores the metadata.
    """
    return group.create_dataset(
        name, shape=shape, dtype=dtype, fillvalue=np.zeros((), dtype=dtype), chunks=True
    )


def zero_h5_entries(path: Path, dataset: str = "entries") -> Path:
    """
    Rewrite ``path`` with ``dataset`` replaced by a zero dataset of the same
    shape and dtype (see :func:`create_zero_dataset`).

    The file is rebuilt (written next to the original, then moved over it)
    rather than edited in place, because deleting a dataset in HDF5 does not
    shrink the file. All other datasets, groups and attributes are copied.
    """
    path = Path(path)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with h5py.File(path, "r") as src, h5py.File(tmp_path, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        for name in src:
            if name == dataset:
                zero = create_zero_dataset(dst, name, src[name].shape, src[name].dtype)
                for key, value in src[name].attrs.items():
                    zero.attrs[key] = value
            else:
                src.copy(src[name], dst, name=name)
    tmp_path.replace(path)
    return path


def hermitize_real_space_blocks(mats_R: np.ndarray, Rijk_list: np.ndarray) -> np.ndarray:
    """
    Enforce Hermiticity in real space by symmetrizing R and -R pairs.

    For each displacement R, this applies:
        M(R) <- 0.5 * (M(R) + M(-R)^dagger)
        M(-R) <- M(R)^dagger
    """
    mats_out = np.array(mats_R, copy=True)
    r_to_idx = {tuple(int(v) for v in r): i for i, r in enumerate(Rijk_list)}
    visited: set[tuple[int, ...]] = set()

    for r, i_r in r_to_idx.items():
        if r in visited:
            continue

        r_neg = (-r[0], -r[1], -r[2])
        i_neg = r_to_idx.get(r_neg)

        A = mats_out[i_r]
        if i_neg is None or i_neg == i_r:
            mats_out[i_r] = 0.5 * (A + np.conjugate(A.T))
            visited.add(r)
            continue

        B = mats_out[i_neg]
        A_sym = 0.5 * (A + np.conjugate(B.T))
        mats_out[i_r] = A_sym
        mats_out[i_neg] = np.conjugate(A_sym.T)

        visited.add(r)
        visited.add(r_neg)

    return mats_out
