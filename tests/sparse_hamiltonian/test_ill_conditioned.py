"""
Orbital removal, against upstream deepx_dock, whose orbital selection it claims to reproduce at lower cost, and
canonical orthogonalization (hamilflow only), against the generalized problem in the reduced basis.

Window regularization is not tested: its docstring marks it as not validated and possibly broken.
"""

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg
from deepx_dock.compute.eigen import ill_conditioned as upstream
from deepx_dock.compute.eigen.hamiltonian import HamiltonianObj

from hamilflow.sparse_hamiltonian import SparseHamiltonianObj, ill_conditioned
from hamilflow.sparse_hamiltonian.ill_conditioned import (
    IllConditionedHandler,
    canonical_orthogonalization_eig,
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


# ---------------------------------- canonical orthogonalization ----------------------------------


def dependent_basis_problem(n=8, n_duplicates=1, noise=0.0, seed=0):
    """(H, S) in a basis of n complex functions (columns of A in a 2n-dimensional space) whose last
    `n_duplicates` duplicate the first ones (plus `noise`), with H = A^dag K A for a Hermitian K, so that H
    is consistent with the basis."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((2 * n, n)) + 1j * rng.standard_normal((2 * n, n))
    for i in range(1, n_duplicates + 1):
        A[:, -i] = A[:, i - 1] + noise * (rng.standard_normal(2 * n) + 1j * rng.standard_normal(2 * n))
    A /= np.linalg.norm(A, axis=0)
    K = rng.standard_normal((2 * n, 2 * n)) + 1j * rng.standard_normal((2 * n, 2 * n))
    K = K + K.conj().T
    return A.conj().T @ K @ A, A.conj().T @ A


def assert_s_orthonormal_eigenvectors(Hk, Sk, vals, vecs, atol, full_span=False):
    """C^dag S C = 1 and C^dag H C = diag(E) (the problem projected on the kept subspace); with `full_span`
    (nothing cut, or only exact null directions of S and H) also H C = S C E."""
    n = vecs.shape[1]
    np.testing.assert_allclose(vecs.conj().T @ Sk @ vecs, np.eye(n), atol=1e-10)
    np.testing.assert_allclose(vecs.conj().T @ Hk @ vecs, np.diag(vals[:n]), atol=atol)
    if full_span:
        np.testing.assert_allclose(Hk @ vecs, Sk @ vecs * vals[:n], atol=atol)


@pytest.mark.parametrize("k", KPOINTS.tolist())
def test_canonical_with_a_well_conditioned_overlap_is_the_generalized_problem(synthetic_structure, k):
    Sk, Hk = synthetic_structure.Sk_and_Hk(k)
    atol = 1e-10 * max(1.0, np.linalg.norm(Hk, 2))
    vals, vecs, n_kept = canonical_orthogonalization_eig(Hk, Sk, THRESHOLD, return_vecs=True)
    assert n_kept == len(Hk)
    np.testing.assert_allclose(vals, scipy.linalg.eigh(Hk, Sk, eigvals_only=True), atol=atol)
    assert vecs.shape == Hk.shape
    assert_s_orthonormal_eigenvectors(Hk, Sk, vals, vecs, atol, full_span=True)
    only_vals, no_vecs, _ = canonical_orthogonalization_eig(Hk, Sk, THRESHOLD)
    assert no_vecs is None
    np.testing.assert_allclose(only_vals, vals, atol=atol)


@pytest.mark.parametrize("n_duplicates", [1, 2])
def test_canonical_with_duplicated_orbitals_solves_the_problem_without_them(n_duplicates):
    """An exactly duplicated function adds nothing to the span: the result is the problem without it."""
    Hk, Sk = dependent_basis_problem(n_duplicates=n_duplicates)
    n = len(Hk) - n_duplicates
    expected = scipy.linalg.eigh(Hk[:n, :n], Sk[:n, :n], eigvals_only=True)
    vals, vecs, n_kept = canonical_orthogonalization_eig(Hk, Sk, 1e-6, fill_value=1e4, return_vecs=True)
    assert n_kept == n
    assert len(vals) == len(Hk)
    np.testing.assert_allclose(vals[:n], expected, atol=1e-9)
    np.testing.assert_array_equal(vals[n:], 1e4)
    assert vecs.shape == (len(Hk), n)
    assert_s_orthonormal_eigenvectors(Hk, Sk, vals, vecs, 1e-9, full_span=True)


def test_canonical_cuts_the_ghost_state_of_an_error_along_a_near_null_direction():
    """A small error of H that is not consistent with a nearly dependent basis gives the plain problem a far-off
    (ghost) eigenvalue; canonical orthogonalization keeps the eigenvalues close to the error-free ones."""
    Hk, Sk = dependent_basis_problem(noise=1e-4)
    lam = np.linalg.eigvalsh(Sk)
    assert lam[0] < 1e-6 and lam[1] > 1e-2
    rng = np.random.default_rng(1)
    E = rng.standard_normal(Hk.shape) + 1j * rng.standard_normal(Hk.shape)
    error = 1e-5 * (E + E.conj().T)
    exact, _, _ = canonical_orthogonalization_eig(Hk, Sk, 1e-3)
    plain = scipy.linalg.eigh(Hk + error, Sk, eigvals_only=True)
    vals, _, n_kept = canonical_orthogonalization_eig(Hk + error, Sk, 1e-3)
    assert n_kept == len(Hk) - 1
    assert np.abs(plain).max() > 100 * np.abs(exact[:n_kept]).max()
    np.testing.assert_allclose(vals[:n_kept], exact[:n_kept], atol=1e-2)


def test_canonical_is_invariant_under_a_unitary_change_of_orbital_basis():
    """Orbital removal is not (it picks orbitals); canonical orthogonalization depends on the span only."""
    Hk, Sk = dependent_basis_problem(n_duplicates=2, noise=1e-3)
    Q, _ = np.linalg.qr(np.random.default_rng(2).standard_normal(Hk.shape) + 0j)
    vals, _, n_kept = canonical_orthogonalization_eig(Hk, Sk, 1e-3)
    assert n_kept == len(Hk) - 2
    rotated, vecs, _ = canonical_orthogonalization_eig(Q.conj().T @ Hk @ Q, Q.conj().T @ Sk @ Q, 1e-3,
                                                       return_vecs=True)
    np.testing.assert_allclose(rotated, vals, atol=1e-9)
    # rotated back, the eigenvectors solve the original problem in its kept subspace
    assert_s_orthonormal_eigenvectors(Hk, Sk, vals, Q @ vecs, 1e-8)


def test_canonical_is_invariant_under_orbital_permutation():
    Hk, Sk = dependent_basis_problem(n_duplicates=2, noise=1e-3)
    perm = np.random.default_rng(3).permutation(len(Hk))
    vals, vecs, _ = canonical_orthogonalization_eig(Hk, Sk, 1e-3, return_vecs=True)
    permuted, pvecs, _ = canonical_orthogonalization_eig(Hk[np.ix_(perm, perm)], Sk[np.ix_(perm, perm)], 1e-3,
                                                         return_vecs=True)
    np.testing.assert_allclose(permuted, vals, atol=1e-9)
    # nondegenerate: the eigenvectors agree up to a phase
    overlap = np.abs(np.sum(vecs[perm].conj() * (Sk[np.ix_(perm, perm)] @ pvecs), axis=0))
    np.testing.assert_allclose(overlap, 1.0, atol=1e-8)


def test_canonical_with_nothing_kept_is_all_fill():
    vals, vecs, n_kept = canonical_orthogonalization_eig(np.eye(3), np.eye(3), 2.0, fill_value=7.0, return_vecs=True)
    np.testing.assert_array_equal(vals, [7.0, 7.0, 7.0])
    assert vecs.shape == (3, 0)
    assert n_kept == 0


def test_canonical_falls_back_and_warns_on_linalg_error(monkeypatch, caplog):
    def failing_eigh(a):
        raise scipy.linalg.LinAlgError("no convergence")

    cpu = ill_conditioned.get_backend(2)
    failing = SimpleNamespace(**{**vars(cpu), "eigh": failing_eigh})
    monkeypatch.setattr(ill_conditioned, "get_backend", lambda n: failing)
    handler = IllConditionedHandler("canonical", verbose=False)
    with caplog.at_level(logging.WARNING, logger=ill_conditioned.__name__):
        vals, vecs = handler.process_k(np.eye(2), np.eye(2), return_vecs=True)
    np.testing.assert_array_equal(vals, [1e4, 1e4])
    assert vecs.shape == (2, 0)
    assert "LinAlgError" in caplog.text
    assert handler.n_kept_min is None


@pytest.mark.parametrize("bad", ["overlap", "hamiltonian"])
def test_canonical_treats_non_finite_eigenvalues_as_a_failure(monkeypatch, caplog, bad):
    """cuSOLVER returns NaN instead of raising: the same fallback as a LinAlgError."""
    cpu = ill_conditioned.get_backend(2)
    calls = []

    def nan_eigh(a):
        calls.append(a)
        w, v = cpu.eigh(a)
        if len(calls) == (1 if bad == "overlap" else 2):
            w = w * np.nan
        return w, v

    nan_backend = SimpleNamespace(**{**vars(cpu), "eigh": nan_eigh})
    monkeypatch.setattr(ill_conditioned, "get_backend", lambda n: nan_backend)
    with caplog.at_level(logging.WARNING, logger=ill_conditioned.__name__):
        vals, vecs, n_kept = canonical_orthogonalization_eig(np.eye(2), np.eye(2), 1e-3, fill_value=7.0,
                                                             return_vecs=True)
    np.testing.assert_array_equal(vals, [7.0, 7.0])
    assert vecs.shape == (2, 0) and n_kept is None
    assert "non-finite" in caplog.text


@pytest.mark.parametrize("return_vecs", [False, True])
def test_canonical_falls_back_on_a_hamiltonian_with_nan(caplog, return_vecs):
    """Not mocked: LAPACK's evr returns zero eigenvalues, not NaN, for such a matrix when it computes vectors."""
    Hk, _ = dependent_basis_problem()
    Hk[2, 3] = np.nan
    with caplog.at_level(logging.WARNING, logger=ill_conditioned.__name__):
        vals, vecs, n_kept = canonical_orthogonalization_eig(Hk, 2 * np.eye(len(Hk)), 1e-3, fill_value=7.0,
                                                             return_vecs=return_vecs)
    np.testing.assert_array_equal(vals, 7.0)
    assert n_kept is None
    assert vecs is None if not return_vecs else vecs.shape == (len(Hk), 0)
    assert "non-finite" in caplog.text


def test_canonical_cuts_an_overlap_eigenvalue_equal_to_the_threshold():
    Sk = np.diag([0.25, 1.0, 2.0])
    vals, _, n_kept = canonical_orthogonalization_eig(np.diag([1.0, 2.0, 3.0]), Sk, 0.25)
    assert n_kept == 2
    np.testing.assert_allclose(vals, [1.5, 2.0, 1e4])


def test_canonical_accepts_real_matrices():
    Hk, Sk = dependent_basis_problem(n_duplicates=1, noise=1e-3)
    Hr, Sr = Hk.real, Sk.real  # still symmetric, Sr positive semidefinite
    vals, vecs, n_kept = canonical_orthogonalization_eig(Hr, Sr, 1e-3, return_vecs=True)
    lam, U = np.linalg.eigh(Sr)
    X = U[:, lam > 1e-3] / np.sqrt(lam[lam > 1e-3])
    assert n_kept == X.shape[1]
    np.testing.assert_allclose(vals[:n_kept], np.linalg.eigvalsh(X.T @ Hr @ X), atol=1e-9)
    assert vecs.dtype == np.float64
    assert_s_orthonormal_eigenvectors(Hr, Sr, vals, vecs, 1e-9)


def test_canonical_pairs_eigenvectors_with_eigenvalues_above_the_fill_value():
    """Fill slots are appended after the real eigenvalues, not sorted in: column j of the eigenvectors is
    eigenvalue j even when some exceed fill_value."""
    Hk, Sk = dependent_basis_problem(n_duplicates=1)
    Hk = 1e5 * Hk
    vals, vecs, n_kept = canonical_orthogonalization_eig(Hk, Sk, 1e-6, fill_value=1e4, return_vecs=True)
    assert vals[:n_kept].max() > 1e4
    np.testing.assert_array_equal(vals[n_kept:], 1e4)
    assert np.all(np.diff(vals[:n_kept]) >= 0)
    assert_s_orthonormal_eigenvectors(Hk, Sk, vals, vecs, 1e-9 * np.linalg.norm(Hk, 2), full_span=True)


@pytest.mark.parametrize("threshold", [0.0, -1e-3])
def test_handler_canonical_needs_a_positive_threshold(threshold):
    with pytest.raises(ValueError, match="ill_threshold > 0"):
        IllConditionedHandler("canonical", ill_threshold=threshold)


@pytest.mark.parametrize("method", ["orbital_removal", "canonical"])
def test_handler_survives_a_pickle_round_trip(method):
    import pickle

    handler = IllConditionedHandler(method, ill_threshold=1e-6, verbose=False)
    if method == "orbital_removal":
        handler.prepare_orbital_truncation(near_dependent_overlaps(12), n_jobs=1)
    Hk, Sk = dependent_basis_problem(n_duplicates=1)
    if method == "canonical":
        handler.process_k(Hk, Sk)
    copy = pickle.loads(pickle.dumps(handler))
    assert copy.get_info() == handler.get_info()
    if method == "canonical":
        copy.process_k(Hk, Sk)  # the lock works again
        assert copy.n_kept_min == handler.n_kept_min


def test_handler_canonical_needs_no_preparation_and_tracks_the_smallest_kept_dimension():
    handler = IllConditionedHandler("canonical", ill_threshold=1e-6, verbose=False)
    assert handler.n_kept_min is None
    assert "n_kept_min" not in handler.get_info()
    for n_duplicates in (1, 2, 0):
        Hk, Sk = dependent_basis_problem(n_duplicates=n_duplicates)
        vals = handler.process_k(Hk, Sk)
        assert np.sum(vals == handler.fill_value) == n_duplicates
    assert handler.n_kept_min == 6
    assert handler.kept_orbitals is None
    assert handler.get_info()["n_kept_min"] == 6
    with pytest.raises(RuntimeError, match="only valid for orbital removal"):
        handler.prepare_orbital_truncation([np.eye(2)])


def test_diag_with_canonical_matches_the_per_k_solution(synthetic_structure):
    """Threshold just above the lowest S(k) eigenvalue, so that directions are cut at some k-points only."""
    h_obj = SparseHamiltonianObj(synthetic_structure.path)
    overlaps = h_obj.get_all_Sk(KPOINTS, n_jobs=1)
    lams = [np.linalg.eigvalsh(S) for S in overlaps]
    threshold = min(lam[0] for lam in lams) + 1e-6
    n = h_obj._mat_dim
    n_kept_k = [int(np.sum(lam > threshold)) for lam in lams]
    assert min(n_kept_k) < n == max(n_kept_k)
    handler = IllConditionedHandler("canonical", ill_threshold=threshold, verbose=False)
    vals, vecs = h_obj.diag(KPOINTS, n_jobs=1, bands_only=False, ill_handler=handler)
    assert vals.shape == (n, len(KPOINTS)) and vecs.shape == (n, n, len(KPOINTS))
    assert handler.n_kept_min == min(n_kept_k)
    for i_k, k in enumerate(KPOINTS):
        Sk, Hk = synthetic_structure.Sk_and_Hk(k)
        expected, _, n_kept = canonical_orthogonalization_eig(Hk, Sk, threshold)
        assert n_kept == n_kept_k[i_k]
        np.testing.assert_allclose(vals[:, i_k], expected, atol=1e-10 * max(1.0, np.linalg.norm(Hk, 2)))
        np.testing.assert_array_equal(vecs[:, n_kept:, i_k], 0.0)
        assert_s_orthonormal_eigenvectors(Hk, Sk, vals[:, i_k], vecs[:, :n_kept, i_k],
                                          1e-10 * max(1.0, np.linalg.norm(Hk, 2)))


@pytest.mark.gpu
def test_canonical_on_the_gpu_matches_the_cpu(monkeypatch):
    from hamilflow.sparse_hamiltonian import dense_backend

    Hk, Sk = dependent_basis_problem(n_duplicates=2, noise=1e-3)
    cpu_vals, _, cpu_kept = canonical_orthogonalization_eig(Hk, Sk, 1e-3)
    monkeypatch.setattr(ill_conditioned, "get_backend", lambda n: dense_backend.gpu_backend())
    vals, vecs, n_kept = canonical_orthogonalization_eig(Hk, Sk, 1e-3, return_vecs=True)
    assert n_kept == cpu_kept
    assert isinstance(vals, np.ndarray) and isinstance(vecs, np.ndarray)
    np.testing.assert_allclose(vals, cpu_vals, atol=1e-9)
    assert_s_orthonormal_eigenvectors(Hk, Sk, vals, vecs, 1e-8)
