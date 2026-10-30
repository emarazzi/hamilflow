"""The fixture itself, checked against upstream deepx_dock so that hamilflow tests rest on a valid reference."""

import h5py
import numpy as np
import pytest
from deepx_dock.compute.eigen.hamiltonian import HamiltonianObj
from deepx_dock.CONSTANT import DEEPX_HAMILTONIAN_FILENAME

KPOINTS = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.13, 0.37, 0.0), (0.21, -0.44, 0.3)]


@pytest.mark.parametrize("k", KPOINTS)
def test_reference_is_hermitian(synthetic_structure, k):
    Sk, Hk = synthetic_structure.Sk_and_Hk(k)
    np.testing.assert_allclose(Sk, Sk.conj().T, atol=1e-13)
    np.testing.assert_allclose(Hk, Hk.conj().T, atol=1e-13)


@pytest.mark.parametrize("k", KPOINTS)
def test_overlap_eigenvalues_stay_in_gershgorin_bounds(synthetic_structure, k):
    Sk, _ = synthetic_structure.Sk_and_Hk(k)
    eigvals = np.linalg.eigvalsh(Sk)
    assert eigvals.min() >= 0.5 - 1e-12
    assert eigvals.max() <= 1.5 + 1e-12


def test_reference_is_not_trivial(synthetic_structure):
    """H(k) must really depend on k and be a dense enough problem, or later comparisons prove little."""
    _, H_gamma = synthetic_structure.Sk_and_Hk(KPOINTS[0])
    _, H_general = synthetic_structure.Sk_and_Hk(KPOINTS[2])
    assert np.abs(H_general - H_gamma).max() > 0.1
    assert np.abs(H_general.imag).max() > 0.1


def test_atom_pairs_are_not_grouped_by_R_like_real_files(synthetic_structure):
    with h5py.File(synthetic_structure.path / DEEPX_HAMILTONIAN_FILENAME, "r") as f:
        Rs = f["atom_pairs"][:, :3]
    n_runs = 1 + np.any(Rs[1:] != Rs[:-1], axis=1).sum()
    assert n_runs > len(np.unique(Rs, axis=0))


@pytest.mark.parametrize("k", KPOINTS)
def test_files_parse_with_upstream_to_the_reference(synthetic_structure, k):
    upstream = HamiltonianObj(synthetic_structure.path)
    Sk, Hk = upstream.Sk_and_Hk(np.array(k))
    S_expected, H_expected = synthetic_structure.Sk_and_Hk(k)
    np.testing.assert_allclose(Sk, S_expected, atol=1e-12)
    np.testing.assert_allclose(Hk, H_expected, atol=1e-12)


def test_same_seed_is_reproducible_and_other_seed_differs(make_synthetic_structure):
    first = make_synthetic_structure("first", seed=1)
    again = make_synthetic_structure("again", seed=1)
    other = make_synthetic_structure("other", seed=2)
    np.testing.assert_array_equal(first.H_R, again.H_R)
    assert first.H_R.shape != other.H_R.shape or not np.array_equal(first.H_R, other.H_R)
