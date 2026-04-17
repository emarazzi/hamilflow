from __future__ import annotations

import json
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


def write_reduced_info_json(
    input_dir: Path,
    output_dir: Path,
    elements: list[str],
    removed_indices: list[int],
) -> Path:
    """Write updated info.json matching the reduced basis and return the output path."""
    with open(input_dir / DEEPX_INFO_FILENAME, "r", encoding="utf-8") as fr:
        raw_info = json.load(fr)

    raw_map = raw_info["elements_orbital_map"]
    new_map = build_reduced_elements_orbital_map(elements, raw_map, removed_indices)

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
) -> Path:
    """
    Dump reduced-basis real-space matrices to DeepH-like h5.

    Unlike the default serializer, this writer recomputes chunk shapes using
    the kept orbital indices per atom, so reduced basis dimensions are encoded
    correctly for each atom pair block.
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

        mat_R = mats_R[r_to_idx[Rijk]]
        ii = per_atom_reduced_indices[ia]
        jj = per_atom_reduced_indices[ja]
        block = mat_R[np.ix_(ii, jj)]

        chunk_shapes[i_ap] = np.array(block.shape, dtype=np.int64)
        chunk_boundaries[i_ap + 1] = chunk_boundaries[i_ap] + block.size
        entries_chunks.append(block.reshape(-1))

    if entries_chunks:
        entries = np.concatenate(entries_chunks)
    else:
        entries = np.array([], dtype=mats_R.dtype)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("atom_pairs", data=atom_pairs)
        f.create_dataset("chunk_shapes", data=chunk_shapes)
        f.create_dataset("chunk_boundaries", data=chunk_boundaries)
        f.create_dataset("entries", data=np.real(entries).astype(np.float64))

    return out_path


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
