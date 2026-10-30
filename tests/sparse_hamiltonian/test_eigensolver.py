"""Eigenpairs of eigh_generalized and SparseHamiltonianObj.diag, against scipy, upstream and their defining equations."""

import numpy as np
import pytest
import scipy.linalg
from deepx_dock.compute.eigen.hamiltonian import HamiltonianObj

from hamilflow.sparse_hamiltonian import SparseHamiltonianObj
from hamilflow.sparse_hamiltonian.eigensolver import eigh_generalized

KPOINTS = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.13, 0.37, 0.0), (0.21, -0.44, 0.3)]


def energy_tol(Hk):
    """Eigenvalue errors scale with the matrix norm, so core levels loosen the tolerance with them."""
    return 1e-12 * max(1.0, np.linalg.norm(Hk, 2))


def assert_eigenpairs(Hk, Sk, vals, vecs):
    """H v = E S v and v^dag S v = 1: true of correct eigenpairs whatever their phase or degeneracy."""
    tol = energy_tol(Hk)
    np.testing.assert_allclose(Hk @ vecs, Sk @ vecs * vals, atol=tol)
    np.testing.assert_allclose(vecs.conj().T @ Sk @ vecs, np.eye(vecs.shape[1]), atol=1e-11)


@pytest.fixture
def h_obj(synthetic_structure):
    return SparseHamiltonianObj(synthetic_structure.path)


@pytest.mark.parametrize("k", KPOINTS)
def test_eigh_generalized_eigenvalues_match_scipy(synthetic_structure, k):
    Sk, Hk = synthetic_structure.Sk_and_Hk(k)
    expected = scipy.linalg.eigh(Hk, Sk, eigvals_only=True)
    np.testing.assert_allclose(eigh_generalized(Hk, Sk, eigvals_only=True), expected, atol=energy_tol(Hk))


@pytest.mark.parametrize("k", KPOINTS)
def test_eigh_generalized_returns_valid_eigenpairs(synthetic_structure, k):
    Sk, Hk = synthetic_structure.Sk_and_Hk(k)
    vals, vecs = eigh_generalized(Hk, Sk)
    assert_eigenpairs(Hk, Sk, vals, vecs)


def test_eigh_generalized_rejects_indefinite_overlap():
    with pytest.raises(np.linalg.LinAlgError):
        eigh_generalized(np.eye(2), np.diag([1.0, -1.0]))


def test_diag_matches_upstream(synthetic_structure, h_obj):
    ks = np.array(KPOINTS)
    expected = HamiltonianObj(synthetic_structure.path).diag(ks)
    vals = h_obj.diag(ks, n_jobs=1)
    assert vals.shape == (h_obj._mat_dim, len(ks))
    tol = energy_tol(synthetic_structure.Sk_and_Hk(ks[0])[1])
    np.testing.assert_allclose(vals, expected, atol=tol)


def test_diag_vectors_are_paired_with_their_values(synthetic_structure, h_obj):
    ks = np.array(KPOINTS)
    vals, vecs = h_obj.diag(ks, n_jobs=1, bands_only=False)
    assert vecs.shape == (h_obj._mat_dim, h_obj._mat_dim, len(ks))
    for i_k, k in enumerate(ks):
        Sk, Hk = synthetic_structure.Sk_and_Hk(k)
        assert_eigenpairs(Hk, Sk, vals[:, i_k], vecs[:, :, i_k])


def test_diag_result_does_not_depend_on_parallelism(synthetic_structure, h_obj):
    ks = np.array(KPOINTS)
    serial = h_obj.diag(ks, n_jobs=1, parallel_k=False)
    threaded = h_obj.diag(ks, n_jobs=2, parallel_k=True)
    tol = energy_tol(synthetic_structure.Sk_and_Hk(ks[0])[1])
    np.testing.assert_allclose(threaded, serial, atol=tol)


def test_diag_keeps_k_point_order_when_threaded(synthetic_structure, h_obj):
    ks = np.array(KPOINTS)
    threaded = h_obj.diag(ks, n_jobs=2, parallel_k=True)
    for i_k, k in enumerate(ks):
        Sk, Hk = synthetic_structure.Sk_and_Hk(k)
        np.testing.assert_allclose(threaded[:, i_k], scipy.linalg.eigh(Hk, Sk, eigvals_only=True), atol=energy_tol(Hk))


def test_diag_accepts_a_generator(h_obj):
    ks = np.array(KPOINTS)
    np.testing.assert_array_equal(h_obj.diag(iter(ks), n_jobs=1), h_obj.diag(ks, n_jobs=1))


@pytest.mark.parametrize("k", KPOINTS)
def test_sparse_calc_finds_the_eigenvalues_nearest_sigma(synthetic_structure, h_obj, k):
    sigma = -0.37  # not an eigenvalue of any of the structures
    Sk, Hk = synthetic_structure.Sk_and_Hk(k)
    exact = scipy.linalg.eigh(Hk, Sk, eigvals_only=True)
    expected = np.sort(exact[np.argsort(np.abs(exact - sigma))[:2]])
    v0 = np.ones(h_obj._mat_dim, dtype=complex)
    vals = h_obj.diag(np.array([k]), n_jobs=1, sparse_calc=True, k=2, sigma=sigma, v0=v0)
    np.testing.assert_allclose(vals[:, 0], expected, atol=1e-8)
