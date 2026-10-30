"""
Orbital removal, against upstream deepx_dock, whose orbital selection it claims to reproduce at lower cost.

Window regularization is not tested: its docstring marks it as not validated and possibly broken.
"""

import logging

import numpy as np
import pytest
import scipy.linalg
from deepx_dock.compute.eigen import ill_conditioned as upstream
from deepx_dock.compute.eigen.hamiltonian import HamiltonianObj

from hamilflow.sparse_hamiltonian import SparseHamiltonianObj, ill_conditioned
from hamilflow.sparse_hamiltonian.ill_conditioned import (
    IllConditionedHandler,
    eig_with_orbital_mask,
    global_orbital_truncation,
)

KPOINTS = np.array([(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.13, 0.37, 0.0), (0.21, -0.44, 0.3)])
THRESHOLD = 1e-3


def near_dependent_overlaps(n, n_k=3, n_dependent=2, seed=0):
    """Random complex S(k) with unit diagonal whose last `n_dependent` orbitals nearly duplicate earlier ones."""
    rng = np.random.default_rng(seed)
    overlaps = []
    for _ in range(n_k):
        A = rng.standard_normal((2 * n, n)) + 1j * rng.standard_normal((2 * n, n))
        for i in range(1, n_dependent + 1):
            noise = rng.standard_normal(2 * n) + 1j * rng.standard_normal(2 * n)
            A[:, -i] = A[:, i - 1] + 1e-4 * rng.uniform(0.5, 1.5) * noise
        A /= np.linalg.norm(A, axis=0)
        overlaps.append(A.conj().T @ A)
    return overlaps


def min_overlap_eigenvalue(overlaps, kept):
    return min(np.linalg.eigvalsh(S[np.ix_(kept, kept)])[0] for S in overlaps)


def test_well_conditioned_overlap_keeps_every_orbital(synthetic_structure):
    overlaps = [synthetic_structure.Sk_and_Hk(k)[0] for k in KPOINTS]
    assert global_orbital_truncation(overlaps, THRESHOLD, verbose=False, n_jobs=1) == list(range(len(overlaps[0])))


@pytest.mark.parametrize("seed", range(5))
def test_truncation_matches_upstream_and_clears_the_threshold(seed):
    overlaps = near_dependent_overlaps(12, seed=seed)
    kept = global_orbital_truncation(overlaps, THRESHOLD, verbose=False, n_jobs=1)
    assert kept == upstream.global_orbital_truncation(overlaps, THRESHOLD, verbose=False)
    assert len(kept) < 12
    assert min_overlap_eigenvalue(overlaps, kept) > THRESHOLD


@pytest.mark.parametrize("seed", range(5))
def test_subspace_iteration_selects_the_same_orbitals_as_upstream(monkeypatch, seed):
    """The fast path normally starts at 300 orbitals: lower the bar to exercise it on a small problem."""
    monkeypatch.setattr(ill_conditioned, "_SUBSPACE_MIN_DIM", 8)
    overlaps = near_dependent_overlaps(40, n_dependent=4, seed=seed)
    kept = global_orbital_truncation(overlaps, THRESHOLD, verbose=False, n_jobs=1)
    assert kept == upstream.global_orbital_truncation(overlaps, THRESHOLD, verbose=False)
    assert len(kept) < 40


def test_subspace_iteration_succeeds_after_orbitals_are_removed(monkeypatch):
    """Failures fall back to LAPACK and still give the right orbitals, so the results alone cannot show this."""
    monkeypatch.setattr(ill_conditioned, "_SUBSPACE_MIN_DIM", 8)
    original = ill_conditioned._lowest_eigpair_subspace
    n_removed_at_success = []

    def spy(state, kept, removed):
        value, vector = original(state, kept, removed)
        if vector is not None:
            n_removed_at_success.append(len(removed))
        return value, vector

    monkeypatch.setattr(ill_conditioned, "_lowest_eigpair_subspace", spy)
    global_orbital_truncation(near_dependent_overlaps(40, n_dependent=4), THRESHOLD, verbose=False, n_jobs=1)
    assert max(n_removed_at_success, default=0) > 0


@pytest.mark.slow
def test_subspace_iteration_at_its_real_size_matches_upstream():
    overlaps = near_dependent_overlaps(320, n_k=2, n_dependent=3)
    kept = global_orbital_truncation(overlaps, THRESHOLD, verbose=False, n_jobs=2)
    assert kept == upstream.global_orbital_truncation(overlaps, THRESHOLD, verbose=False)
    assert len(kept) < 320


def test_truncation_result_does_not_depend_on_threads():
    overlaps = near_dependent_overlaps(12)
    assert global_orbital_truncation(overlaps, THRESHOLD, verbose=False, n_jobs=1) == global_orbital_truncation(
        overlaps, THRESHOLD, verbose=False, n_jobs=2
    )


def _masked_problem(synthetic_structure):
    Sk, Hk = synthetic_structure.Sk_and_Hk(KPOINTS[2])
    kept = [i for i in range(len(Hk)) if i != 2]
    return Hk, Sk, kept


def test_orbital_mask_solves_the_kept_submatrix_and_fills_the_rest(synthetic_structure):
    Hk, Sk, kept = _masked_problem(synthetic_structure)
    idx = np.ix_(kept, kept)
    expected = scipy.linalg.eigh(Hk[idx], Sk[idx], eigvals_only=True)
    vals, vecs = eig_with_orbital_mask(Hk, Sk, kept, fill_value=1e4)
    assert vecs is None
    np.testing.assert_allclose(vals[:-1], expected, atol=1e-12 * max(1.0, np.linalg.norm(Hk, 2)))
    assert vals[-1] == 1e4


def test_orbital_mask_vectors_vanish_on_removed_orbitals_and_solve_the_submatrix(synthetic_structure):
    Hk, Sk, kept = _masked_problem(synthetic_structure)
    vals, vecs = eig_with_orbital_mask(Hk, Sk, kept, return_vecs=True)
    assert vecs.shape == (len(Hk), len(kept))
    np.testing.assert_array_equal(vecs[2], 0.0)
    idx = np.ix_(kept, kept)
    sub = vecs[kept]
    np.testing.assert_allclose(
        Hk[idx] @ sub, Sk[idx] @ sub * vals[: len(kept)], atol=1e-12 * max(1.0, np.linalg.norm(Hk, 2))
    )


def test_orbital_mask_matches_upstream(synthetic_structure):
    Hk, Sk, kept = _masked_problem(synthetic_structure)
    expected, _ = upstream.eig_with_orbital_mask(Hk, Sk, kept)
    vals, _ = eig_with_orbital_mask(Hk, Sk, kept)
    np.testing.assert_allclose(vals, expected, atol=1e-12 * max(1.0, np.linalg.norm(Hk, 2)))


def test_orbital_mask_with_nothing_kept_is_all_fill():
    vals, vecs = eig_with_orbital_mask(np.eye(3), np.eye(3), [], fill_value=7.0, return_vecs=True)
    np.testing.assert_array_equal(vals, [7.0, 7.0, 7.0])
    assert vecs.shape == (3, 0)


def test_orbital_mask_falls_back_and_warns_when_overlap_is_indefinite(caplog):
    with caplog.at_level(logging.WARNING, logger=ill_conditioned.__name__):
        vals, _ = eig_with_orbital_mask(np.eye(2), np.diag([1.0, -1.0]), [0, 1], fill_value=7.0)
    np.testing.assert_array_equal(vals, [7.0, 7.0])
    assert "LinAlgError" in caplog.text


def test_diag_with_orbital_removal_matches_upstream(synthetic_structure):
    """Threshold just above the lowest S(k) eigenvalue, so that removal really happens on this structure."""
    h_obj = SparseHamiltonianObj(synthetic_structure.path)
    overlaps = h_obj.get_all_Sk(KPOINTS, n_jobs=1)
    threshold = min(np.linalg.eigvalsh(S)[0] for S in overlaps) + 0.01

    handler = IllConditionedHandler("orbital_removal", ill_threshold=threshold, verbose=False)
    kept = handler.prepare_orbital_truncation(overlaps, n_jobs=1)
    assert kept == upstream.global_orbital_truncation(overlaps, threshold, verbose=False)
    assert len(kept) < h_obj._mat_dim

    vals = h_obj.diag(KPOINTS, n_jobs=1, ill_handler=handler)
    # Upstream's own diag(kept_orbitals=...) cannot stack its (vals, None) results, so go through its solver
    upstream_h = HamiltonianObj(synthetic_structure.path)
    expected = np.stack(
        [upstream.eig_with_orbital_mask(*upstream_h.Sk_and_Hk(k)[::-1], kept)[0] for k in KPOINTS], axis=1
    )
    np.testing.assert_allclose(vals, expected, atol=1e-12 * max(1.0, np.abs(synthetic_structure.H_R).max()))
    np.testing.assert_allclose(h_obj.diag(KPOINTS, n_jobs=1, kept_orbitals=kept), vals, atol=1e-9)


def test_diag_pads_vectors_of_removed_slots_with_zero_columns(synthetic_structure):
    h_obj = SparseHamiltonianObj(synthetic_structure.path)
    kept = list(range(h_obj._mat_dim - 1))
    vals, vecs = h_obj.diag(KPOINTS, n_jobs=1, bands_only=False, kept_orbitals=kept)
    assert vecs.shape == (h_obj._mat_dim, h_obj._mat_dim, len(KPOINTS))
    np.testing.assert_array_equal(vals[-1], 1e4)
    np.testing.assert_array_equal(vecs[:, -1], 0.0)


def test_handler_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        IllConditionedHandler("no_such_method")


def test_handler_needs_prepare_before_process_k():
    handler = IllConditionedHandler("orbital_removal")
    with pytest.raises(RuntimeError, match="prepare_orbital_truncation"):
        handler.process_k(np.eye(2), np.eye(2))


def test_handler_refuses_truncation_in_window_mode():
    handler = IllConditionedHandler("window_regularization")
    with pytest.raises(RuntimeError, match="only valid for orbital removal"):
        handler.prepare_orbital_truncation([np.eye(2)])


def test_handler_info_reports_the_kept_orbitals():
    handler = IllConditionedHandler("orbital_removal", ill_threshold=THRESHOLD, verbose=False)
    assert "kept_orbitals" not in handler.get_info()
    kept = handler.prepare_orbital_truncation(near_dependent_overlaps(12), n_jobs=1)
    info = handler.get_info()
    assert info["kept_orbitals"] == kept
    assert info["n_kept_orbitals"] == len(kept)
