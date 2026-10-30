"""H(k) and S(k) built from the sparse blocks, against the fixture's direct sum and upstream's dense implementation."""

import shutil

import h5py
import numpy as np
import pytest
from deepx_dock.compute.eigen.hamiltonian import HamiltonianObj
from deepx_dock.CONSTANT import (
    DEEPX_HAMILTONIAN_FILENAME,
    DEEPX_INFO_FILENAME,
    DEEPX_OVERLAP_FILENAME,
    DEEPX_POSCAR_FILENAME,
)

from hamilflow.sparse_hamiltonian import SparseHamiltonianObj

KPOINTS = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.13, 0.37, 0.0), (0.21, -0.44, 0.3)]
TOL = 1e-11


@pytest.fixture
def h_obj(synthetic_structure):
    return SparseHamiltonianObj(synthetic_structure.path)


@pytest.mark.parametrize("k", KPOINTS)
def test_matrices_match_direct_sum(synthetic_structure, h_obj, k):
    Sk, Hk = h_obj.Sk_and_Hk(np.array(k))
    S_expected, H_expected = synthetic_structure.Sk_and_Hk(k)
    np.testing.assert_allclose(Sk, S_expected, atol=TOL)
    np.testing.assert_allclose(Hk, H_expected, atol=TOL)


@pytest.mark.parametrize("k", KPOINTS)
def test_matrices_match_upstream(synthetic_structure, h_obj, k):
    Sk, Hk = h_obj.Sk_and_Hk(np.array(k))
    S_upstream, H_upstream = HamiltonianObj(synthetic_structure.path).Sk_and_Hk(np.array(k))
    np.testing.assert_allclose(Sk, S_upstream, atol=TOL)
    np.testing.assert_allclose(Hk, H_upstream, atol=TOL)


def test_rijk_list_matches_upstream(synthetic_structure, h_obj):
    np.testing.assert_array_equal(h_obj.Rijk_list, HamiltonianObj(synthetic_structure.path).Rijk_list)


@pytest.mark.parametrize("k", KPOINTS)
def test_sparse_build_equals_dense_build(h_obj, k):
    k = np.array(k)
    np.testing.assert_allclose(h_obj._sparse_k(k).toarray(), h_obj._dense_k(k), atol=TOL)
    np.testing.assert_allclose(h_obj._s_obj._sparse_k(k).toarray(), h_obj._s_obj._dense_k(k), atol=TOL)


def test_batch_of_k_points_stacks_the_single_results(h_obj):
    ks = np.array(KPOINTS)
    Sk, Hk = h_obj.Sk_and_Hk(ks)
    assert Sk.shape == Hk.shape == (len(ks), h_obj._mat_dim, h_obj._mat_dim)
    S_single, H_single = h_obj.Sk_and_Hk(ks[2])
    np.testing.assert_array_equal(Sk[2], S_single)
    np.testing.assert_array_equal(Hk[2], H_single)


def test_overlap_only_getters_match_sk_and_hk(h_obj):
    ks = np.array(KPOINTS)
    Sk, _ = h_obj.Sk_and_Hk(ks)
    np.testing.assert_array_equal(h_obj.get_Sk(ks), Sk)
    np.testing.assert_array_equal(h_obj.get_Sk(ks[1]), Sk[1])
    np.testing.assert_array_equal(np.stack(h_obj.get_all_Sk(ks, n_jobs=2)), Sk)
    np.testing.assert_array_equal(np.stack(h_obj.get_all_Sk(ks, n_jobs=1, parallel_k=False)), Sk)


def test_dense_real_space_matrices_are_refused(h_obj):
    with pytest.raises(AttributeError):
        _ = h_obj.HR
    with pytest.raises(AttributeError):
        _ = h_obj.SR


def test_load_hamiltonian_false_keeps_structure_and_overlap_only(synthetic_structure, h_obj):
    light = SparseHamiltonianObj(synthetic_structure.path, load_hamiltonian=False)
    np.testing.assert_array_equal(light.Rijk_list, h_obj.Rijk_list)
    np.testing.assert_array_equal(light.get_Sk(np.array(KPOINTS[2])), h_obj.get_Sk(np.array(KPOINTS[2])))
    with pytest.raises(AttributeError, match="load_matrix=False"):
        light._dense_k(np.array(KPOINTS[2]))


def _write_scaled_hamiltonian(source, destination, factor):
    with h5py.File(source, "r") as fr, h5py.File(destination, "w") as fw:
        for name in ("atom_pairs", "chunk_boundaries", "chunk_shapes"):
            fw[name] = fr[name][:]
        fw["entries"] = factor * fr["entries"][:]


def test_share_from_with_same_layout_reuses_arrays_but_keeps_own_values(synthetic_structure, h_obj, tmp_path):
    """Ensemble members share the destination arrays: a member must still get its own entries."""
    member_path = tmp_path / "member.h5"
    _write_scaled_hamiltonian(synthetic_structure.path / DEEPX_HAMILTONIAN_FILENAME, member_path, 2.0)
    member = SparseHamiltonianObj(synthetic_structure.path, H_file_path=member_path, share_from=h_obj)

    assert member._entry_row is h_obj._entry_row
    assert member._s_obj is h_obj._s_obj

    k = np.array(KPOINTS[2])
    S_expected, H_expected = synthetic_structure.Sk_and_Hk(k)
    Sk, Hk = member.Sk_and_Hk(k)
    np.testing.assert_allclose(Sk, S_expected, atol=TOL)
    np.testing.assert_allclose(Hk, 2.0 * H_expected, atol=TOL)
    np.testing.assert_allclose(h_obj.Sk_and_Hk(k)[1], H_expected, atol=TOL)


def test_share_from_with_other_layout_is_ignored(make_synthetic_structure):
    first = make_synthetic_structure("first", seed=1)
    second = make_synthetic_structure("second", seed=2)
    h_first = SparseHamiltonianObj(first.path)
    h_second = SparseHamiltonianObj(second.path, share_from=h_first)

    assert h_second._entry_row is not h_first._entry_row
    assert h_second._s_obj is not h_first._s_obj

    k = np.array(KPOINTS[2])
    S_expected, H_expected = second.Sk_and_Hk(k)
    Sk, Hk = h_second.Sk_and_Hk(k)
    np.testing.assert_allclose(Sk, S_expected, atol=TOL)
    np.testing.assert_allclose(Hk, H_expected, atol=TOL)


def _write_with_two_blocks_swapped(source, destination, first, second):
    """Same matrix with two equally shaped blocks swapped in the file: only atom_pairs tells the layouts apart."""
    with h5py.File(source, "r") as fr, h5py.File(destination, "w") as fw:
        atom_pairs = fr["atom_pairs"][:]
        bounds = fr["chunk_boundaries"][:]
        entries = fr["entries"][:]
        fw["chunk_boundaries"] = bounds
        fw["chunk_shapes"] = fr["chunk_shapes"][:]
    atom_pairs[[first, second]] = atom_pairs[[second, first]]
    first_slice = slice(bounds[first], bounds[first + 1])
    second_slice = slice(bounds[second], bounds[second + 1])
    entries[first_slice], entries[second_slice] = entries[second_slice].copy(), entries[first_slice].copy()
    with h5py.File(destination, "a") as fw:
        fw["atom_pairs"] = atom_pairs
        fw["entries"] = entries


def test_share_from_with_same_shapes_but_other_pair_order_is_ignored(synthetic_structure, h_obj, tmp_path):
    """Files can agree on every chunk size and still place their blocks differently."""
    with h5py.File(synthetic_structure.path / DEEPX_OVERLAP_FILENAME, "r") as f:
        S_shapes = f["chunk_shapes"][:]
    with h5py.File(synthetic_structure.path / DEEPX_HAMILTONIAN_FILENAME, "r") as f:
        H_shapes = f["chunk_shapes"][:]
    first = next(
        i
        for i in range(1, len(H_shapes))
        if np.array_equal(H_shapes[i], H_shapes[0]) and np.array_equal(S_shapes[i], S_shapes[0])
    )
    for name in (DEEPX_INFO_FILENAME, DEEPX_POSCAR_FILENAME):
        shutil.copy(synthetic_structure.path / name, tmp_path / name)
    for name in (DEEPX_OVERLAP_FILENAME, DEEPX_HAMILTONIAN_FILENAME):
        _write_with_two_blocks_swapped(synthetic_structure.path / name, tmp_path / name, 0, first)

    reordered = SparseHamiltonianObj(tmp_path, share_from=h_obj)

    assert reordered._entry_row is not h_obj._entry_row
    k = np.array(KPOINTS[2])
    S_expected, H_expected = synthetic_structure.Sk_and_Hk(k)
    Sk, Hk = reordered.Sk_and_Hk(k)
    np.testing.assert_allclose(Sk, S_expected, atol=TOL)
    np.testing.assert_allclose(Hk, H_expected, atol=TOL)


def test_fill_fraction_counts_structural_nonzeros(synthetic_structure, h_obj):
    expected = np.any(synthetic_structure.H_R != 0, axis=0).mean()
    assert h_obj.fill_fraction >= expected
    assert 0 < h_obj.fill_fraction <= 1
