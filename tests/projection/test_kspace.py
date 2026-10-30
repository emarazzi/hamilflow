"""
The reduction at one k-point, the inverse Fourier transform, and the streaming projection built from them.

The physics: T = [1; -S_rr^-1 S_rk] gives S' = T^dag S T, the Schur complement of S_rr, and its columns are
S-orthogonal to the removed orbitals. So when the removed orbitals span an invariant subspace (a true core
state, H e_r = eps S e_r), the reduced problem keeps every other eigenvalue exactly.
"""

import logging

import numpy as np
import pytest
import scipy.linalg
from deepx_dock.compute.eigen.matrix_obj import AOMatrixK

from hamilflow.projection import kspace
from hamilflow.projection.kspace import (
    apply_custom_kspace_transform,
    apply_custom_kspace_transform_overlap_only,
    apply_tk_projection,
    apply_truncation_kspace_transform,
    apply_truncation_kspace_transform_overlap_only,
    available_memory_bytes,
    build_elimination_tk,
    build_uniform_kmesh,
    hk_and_sk_to_real,
    k_to_r_operator,
    sk_to_real,
    stream_project_to_real_space,
)
from hamilflow.sparse_hamiltonian import SparseHamiltonianObj

NB = 8
RM = [1, 4]
KEEP = [0, 2, 3, 5, 6, 7]


def random_problem(nk=3, nb=NB, seed=0):
    """Complex Hermitian H(k) and positive definite S(k), (nk, nb, nb)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((nk, nb, nb)) + 1j * rng.standard_normal((nk, nb, nb))
    B = rng.standard_normal((nk, nb, nb)) + 1j * rng.standard_normal((nk, nb, nb))
    Hk = A + A.conj().transpose(0, 2, 1)
    Sk = B @ B.conj().transpose(0, 2, 1) + nb * np.eye(nb)
    return Hk, Sk


def schur_complement(S):
    return S[np.ix_(KEEP, KEEP)] - S[np.ix_(KEEP, RM)] @ np.linalg.solve(S[np.ix_(RM, RM)], S[np.ix_(RM, KEEP)])


# ---------------------------------- mesh ----------------------------------


def test_uniform_mesh_has_every_point_in_the_unit_cell_once():
    ks = build_uniform_kmesh((2, 3, 1))
    assert ks.shape == (6, 3)
    assert {tuple(k) for k in np.round(ks, 10)} == {(a / 2, round(b / 3, 10), 0.0) for a in range(2) for b in range(3)}
    assert ks.min() >= 0
    assert ks.max() < 1


# ---------------------------------- Schur elimination ----------------------------------


def test_transform_has_identity_on_kept_rows_and_coefficients_on_removed_rows():
    _, Sk = random_problem()
    Tk, keep, rm = build_elimination_tk(Sk, RM)
    assert (keep, rm) == (KEEP, RM)
    assert Tk.shape == (3, NB, len(KEEP))
    np.testing.assert_array_equal(Tk[:, KEEP, :], np.broadcast_to(np.eye(len(KEEP)), (3, len(KEEP), len(KEEP))))
    expected = -np.linalg.solve(Sk[0][np.ix_(RM, RM)], Sk[0][np.ix_(RM, KEEP)])
    np.testing.assert_allclose(Tk[0][RM], expected)


def test_new_basis_is_overlap_orthogonal_to_the_removed_orbitals():
    _, Sk = random_problem()
    Tk, _, _ = build_elimination_tk(Sk, RM)
    np.testing.assert_allclose(Tk.conj().transpose(0, 2, 1) @ Sk[:, :, RM], 0.0, atol=1e-12)


def test_reduced_overlap_is_the_schur_complement_and_stays_positive_definite():
    Hk, Sk = random_problem()
    _, S_new = apply_custom_kspace_transform(Hk, Sk, RM)
    for S, reduced in zip(Sk, S_new, strict=True):
        np.testing.assert_allclose(reduced, schur_complement(S), atol=1e-12)
        assert np.linalg.eigvalsh(reduced).min() > 0


def test_reduced_hamiltonian_stays_hermitian():
    Hk, Sk = random_problem()
    H_new, _ = apply_custom_kspace_transform(Hk, Sk, RM)
    np.testing.assert_allclose(H_new, H_new.conj().transpose(0, 2, 1), atol=1e-11)


@pytest.mark.parametrize("eps", [-1.0, -20000.0])
def test_removing_a_true_core_state_keeps_every_other_eigenvalue(eps):
    """H e_r = eps S e_r: the removed orbitals overlap the others but are an exact eigenspace."""
    Hk, Sk = random_problem(nk=1, seed=1)
    S = Sk[0]
    H = Hk[0].copy()
    H[RM, :] = eps * S[RM, :]
    H[:, RM] = eps * S[:, RM]
    full = scipy.linalg.eigh(H, S, eigvals_only=True)

    H_new, S_new = apply_custom_kspace_transform(H[None], S[None], RM)
    reduced = scipy.linalg.eigh(H_new[0], S_new[0], eigvals_only=True)

    remaining = list(full)
    for _ in RM:
        remaining.pop(int(np.argmin(np.abs(np.array(remaining) - eps))))
    np.testing.assert_allclose(reduced, remaining, atol=1e-9 * max(1.0, abs(eps)))


def test_nothing_to_remove_returns_the_input():
    Hk, Sk = random_problem()
    assert apply_custom_kspace_transform(Hk, Sk, [])[0] is Hk
    assert apply_truncation_kspace_transform(Hk, Sk, [])[1] is Sk
    assert apply_custom_kspace_transform_overlap_only(Sk, []) is Sk
    assert apply_truncation_kspace_transform_overlap_only(Sk, []) is Sk
    Tk, keep, rm = build_elimination_tk(Sk, [])
    np.testing.assert_array_equal(Tk[0], np.eye(NB))
    assert (keep, rm) == (list(range(NB)), [])


def test_transforms_with_and_without_the_hamiltonian_give_the_same_overlap():
    Hk, Sk = random_problem()
    np.testing.assert_array_equal(
        apply_custom_kspace_transform_overlap_only(Sk, RM), apply_custom_kspace_transform(Hk, Sk, RM)[1]
    )
    np.testing.assert_array_equal(
        apply_truncation_kspace_transform_overlap_only(Sk, RM), apply_truncation_kspace_transform(Hk, Sk, RM)[1]
    )


def test_truncation_only_deletes_rows_and_columns():
    Hk, Sk = random_problem()
    H_new, S_new = apply_truncation_kspace_transform(Hk, Sk, RM)
    np.testing.assert_array_equal(H_new[1], Hk[1][np.ix_(KEEP, KEEP)])
    np.testing.assert_array_equal(S_new[1], Sk[1][np.ix_(KEEP, KEEP)])


def test_duplicate_removed_indices_count_once():
    Hk, Sk = random_problem()
    np.testing.assert_array_equal(
        apply_custom_kspace_transform(Hk, Sk, [4, 1, 1])[0], apply_custom_kspace_transform(Hk, Sk, RM)[0]
    )


@pytest.mark.parametrize(
    "transform",
    [
        lambda H, S, rm: apply_custom_kspace_transform(H, S, rm),
        lambda H, S, rm: apply_truncation_kspace_transform(H, S, rm),
        lambda H, S, rm: apply_custom_kspace_transform_overlap_only(S, rm),
        lambda H, S, rm: apply_truncation_kspace_transform_overlap_only(S, rm),
    ],
    ids=["schur", "truncate", "schur-overlap", "truncate-overlap"],
)
def test_invalid_removal_is_rejected(transform):
    Hk, Sk = random_problem()
    with pytest.raises(ValueError, match="out of range"):
        transform(Hk, Sk, [NB])
    with pytest.raises(ValueError, match="out of range"):
        transform(Hk, Sk, [-1])
    with pytest.raises(ValueError, match="Cannot remove all"):
        transform(Hk, Sk, list(range(NB)))


@pytest.mark.parametrize(
    "transform", [apply_custom_kspace_transform, apply_truncation_kspace_transform], ids=["schur", "truncate"]
)
def test_wrong_shapes_are_rejected(transform):
    Hk, Sk = random_problem()
    with pytest.raises(ValueError, match="shape mismatch"):
        transform(Hk, Sk[:, :-1, :-1], RM)
    with pytest.raises(ValueError, match="must have shape"):
        transform(Hk[0], Sk[0], RM)


@pytest.mark.parametrize("block_rows", [1, 2, NB])
def test_memory_saving_projection_equals_the_explicit_product(monkeypatch, block_rows):
    """`_project_into` never forms T: it must still give T^dag M T, whatever its block size."""
    monkeypatch.setattr(kspace, "_ROW_BLOCK_BYTES", kspace._C16 * NB * block_rows)
    Hk, Sk = random_problem(nk=1)
    keep, rm = np.array(KEEP), np.array(RM)
    C = kspace._schur_coefficients(Sk[0], keep, rm)
    out = np.empty((len(KEEP), len(KEEP)), dtype=complex)
    kspace._project_into(Hk[0], C, keep, rm, out)
    Tk, _, _ = build_elimination_tk(Sk, RM)
    np.testing.assert_allclose(out, apply_tk_projection(Hk, Sk, Tk)[0][0], atol=1e-11)


def test_memory_saving_truncation_equals_the_submatrix():
    Hk, _ = random_problem(nk=1)
    out = np.empty((len(KEEP), len(KEEP)), dtype=complex)
    kspace._project_into(Hk[0], None, np.array(KEEP), np.array(RM), out)
    np.testing.assert_array_equal(out, Hk[0][np.ix_(KEEP, KEEP)])


@pytest.mark.parametrize("mode", ["schur", "truncate"])
def test_reduce_one_k_matches_the_batch_transforms(mode):
    Hk, Sk = random_problem(nk=1)
    H_new, S_new = kspace._reduce_one_k(Hk[0], Sk[0], np.array(KEEP), np.array(RM), mode)
    batch = apply_custom_kspace_transform if mode == "schur" else apply_truncation_kspace_transform
    H_batch, S_batch = batch(Hk, Sk, RM)
    np.testing.assert_allclose(H_new, H_batch[0], atol=1e-11)
    np.testing.assert_allclose(S_new, S_batch[0], atol=1e-11)


def test_reduce_one_k_without_hamiltonian_or_removal():
    Hk, Sk = random_problem(nk=1)
    keep, rm = np.array(KEEP), np.array(RM)
    assert kspace._reduce_one_k(None, Sk[0], keep, rm, "schur")[0] is None
    H, S = Hk[0], Sk[0]
    H_same, S_same = kspace._reduce_one_k(H, S, np.arange(NB), np.array([], dtype=int), "schur")
    assert H_same is H
    assert S_same is S


# ---------------------------------- Fourier transform ----------------------------------


RS = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])


def real_space_matrices(seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((len(RS), 3, 3)) + 1j * rng.standard_normal((len(RS), 3, 3))


def forward(ks, mats):
    return np.exp(2j * np.pi * ks @ RS.T) @ mats.reshape(len(RS), -1)


def test_inverse_transform_recovers_the_real_space_matrices():
    mats = real_space_matrices()
    ks = build_uniform_kmesh((3, 3, 1))
    flat = forward(ks, mats)
    np.testing.assert_allclose(k_to_r_operator(ks, RS, flat.reshape(len(ks), 3, 3)), mats, atol=1e-12)


def test_inverse_transform_matches_upstream():
    ks = build_uniform_kmesh((3, 3, 1))
    flat = forward(ks, real_space_matrices())
    Mk = flat.reshape(len(ks), 3, 3)
    np.testing.assert_allclose(k_to_r_operator(ks, RS, Mk), AOMatrixK(ks, Mk).k2r(RS), atol=1e-12)


def test_mesh_too_coarse_for_the_r_range_aliases():
    """With 2 k-points per direction R and -R are the same: why the mesh must fit the range of the matrices."""
    mats = real_space_matrices()
    ks = build_uniform_kmesh((2, 2, 1))
    flat = forward(ks, mats)
    aliased = k_to_r_operator(ks, RS, flat.reshape(len(ks), 3, 3))
    np.testing.assert_allclose(aliased[1], mats[1] + mats[2], atol=1e-12)


def test_weights_are_applied():
    ks = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    Mk = np.array([np.eye(2), 3 * np.eye(2)])
    R0 = np.array([[0, 0, 0]])
    np.testing.assert_allclose(k_to_r_operator(ks, R0, Mk, weights=np.array([0.25, 0.75]))[0], 2.5 * np.eye(2))
    np.testing.assert_allclose(k_to_r_operator(ks, R0, Mk)[0], 2.0 * np.eye(2))


def test_upstream_based_transforms_agree_with_the_operator():
    ks = build_uniform_kmesh((3, 3, 1))
    flat = forward(ks, real_space_matrices())
    Mk = flat.reshape(len(ks), 3, 3)
    expected = k_to_r_operator(ks, RS, Mk)
    HR, SR = hk_and_sk_to_real(ks, Mk, 2 * Mk, RS)
    np.testing.assert_allclose(HR, expected, atol=1e-12)
    np.testing.assert_allclose(SR, 2 * expected, atol=1e-12)
    np.testing.assert_allclose(sk_to_real(ks, Mk, RS), expected, atol=1e-12)


@pytest.mark.parametrize(
    ("ks", "rs", "mk", "message"),
    [
        (np.zeros((2, 2)), RS, np.zeros((2, 3, 3)), "ks must have"),
        (np.zeros((2, 3)), RS[0], np.zeros((2, 3, 3)), "Rijk_list must have"),
        (np.zeros((2, 3)), RS, np.zeros((3, 3)), "Mk must have|must have shape"),
        (np.zeros((2, 3)), RS, np.zeros((3, 3, 3)), "Nk mismatch"),
    ],
)
def test_wrong_transform_inputs_are_rejected(ks, rs, mk, message):
    with pytest.raises(ValueError, match=message):
        k_to_r_operator(ks, rs, mk)
    with pytest.raises(ValueError, match=message):
        sk_to_real(ks, mk, rs)
    with pytest.raises(ValueError, match=message):
        hk_and_sk_to_real(ks, mk, mk, rs)


def test_wrong_weights_are_rejected():
    with pytest.raises(ValueError, match="weights"):
        k_to_r_operator(np.zeros((2, 3)), RS, np.zeros((2, 3, 3)), weights=np.ones(3))


# ---------------------------------- memory ----------------------------------


@pytest.fixture
def no_other_limits(monkeypatch):
    """Only what each test sets limits the memory; it starts with a process of no size."""
    monkeypatch.setattr(kspace, "_cgroup_memory_limit", lambda: None)
    monkeypatch.setattr(kspace, "_proc_status_bytes", lambda field: 0)
    monkeypatch.delenv("SLURM_MEM_PER_NODE", raising=False)
    monkeypatch.delenv("SLURM_MEM_PER_CPU", raising=False)


def test_cgroup_limit_bounds_the_available_memory(monkeypatch, no_other_limits):
    monkeypatch.setattr(kspace, "_cgroup_memory_limit", lambda: 1000)
    assert available_memory_bytes() == 1000


def test_slurm_node_limit_bounds_the_available_memory(monkeypatch, no_other_limits):
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "1")
    assert available_memory_bytes() == 1024**2


def test_slurm_cpu_limit_is_per_allocated_cpu(monkeypatch, no_other_limits):
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "1")
    monkeypatch.setattr(kspace, "available_cpu_count", lambda: 3)
    assert available_memory_bytes() == 3 * 1024**2


def test_malformed_slurm_limit_is_ignored(monkeypatch, no_other_limits):
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "lots")
    assert available_memory_bytes() is None or available_memory_bytes() > 1024


def test_memory_in_use_is_subtracted_from_the_limit_and_never_goes_negative(monkeypatch, no_other_limits):
    monkeypatch.setattr(kspace, "_cgroup_memory_limit", lambda: 1000)
    monkeypatch.setattr(kspace, "_proc_status_bytes", lambda field: 400 if field == "VmRSS" else 0)
    assert available_memory_bytes() == 600
    monkeypatch.setattr(kspace, "_proc_status_bytes", lambda field: 2000 if field == "VmRSS" else 0)
    assert available_memory_bytes() == 0


def test_worker_estimate_grows_with_the_schur_terms():
    base = kspace._estimate_worker_bytes(100, 80, 0, 1, 2, 1000)
    assert kspace._estimate_worker_bytes(100, 80, 20, 1, 2, 1000) > base
    assert kspace._estimate_worker_bytes(100, 80, 0, 2, 2, 1000) > base
    assert kspace._estimate_worker_bytes(200, 80, 0, 1, 2, 1000) > base


# ---------------------------------- streaming projection ----------------------------------


@pytest.fixture
def structure(make_synthetic_structure):
    return make_synthetic_structure()


def targets_for(keep, obj):
    """Every entry of every R-block as a target, so the result can be compared with dense matrices."""
    nkp = len(keep)
    target_flat = np.tile(np.arange(nkp * nkp), len(obj.Rijk_list))
    r_bounds = np.arange(len(obj.Rijk_list) + 1) * nkp * nkp
    return target_flat, r_bounds


def dense_reference(obj, ks, rm, keep, mode):
    Sk, Hk = obj.Sk_and_Hk(ks)
    transform = apply_custom_kspace_transform if mode == "schur" else apply_truncation_kspace_transform
    H_new, S_new = transform(Hk, Sk, rm)
    return k_to_r_operator(ks, obj.Rijk_list, H_new), k_to_r_operator(ks, obj.Rijk_list, S_new)


@pytest.mark.parametrize("mode", ["schur", "truncate"])
@pytest.mark.parametrize(
    "settings",
    [
        {"n_workers": 1},
        {"n_workers": 2},
        {"n_workers": 1, "k_chunk_size": 1},
        {"n_workers": 2, "k_chunk_size": 2},
    ],
    ids=["serial", "threads", "one-k-per-chunk", "threads-chunks"],
)
def test_streaming_projection_matches_the_dense_transforms(structure, mode, settings):
    obj = SparseHamiltonianObj(structure.path)
    rm, keep = [1, 2, 3], [0, 4, 5]
    ks = build_uniform_kmesh((3, 3, 1))
    target_flat, r_bounds = targets_for(keep, obj)
    HR, SR = stream_project_to_real_space(obj, ks, rm, mode, obj.Rijk_list, target_flat, r_bounds, **settings)
    H_dense, S_dense = dense_reference(obj, ks, rm, keep, mode)
    np.testing.assert_allclose(HR, H_dense.real.reshape(-1), atol=1e-10)
    np.testing.assert_allclose(SR, S_dense.reshape(-1), atol=1e-10)


@pytest.mark.parametrize("density_inv", [10**9, 0], ids=["all-by-block-product", "all-by-gathering"])
@pytest.mark.parametrize("ift_bytes", [kspace._IFT_BLOCK_BYTES, kspace._C16], ids=["big-blocks", "tiny-blocks"])
def test_both_transform_paths_give_the_same_result(monkeypatch, structure, density_inv, ift_bytes):
    monkeypatch.setattr(kspace, "_GATHER_MAX_DENSITY_INV", density_inv)
    monkeypatch.setattr(kspace, "_IFT_BLOCK_BYTES", ift_bytes)
    obj = SparseHamiltonianObj(structure.path)
    rm, keep = [1, 2, 3], [0, 4, 5]
    ks = build_uniform_kmesh((3, 3, 1))
    # a few targets only: a sparse block, with entries spread over the columns
    flat = np.array([0, 4, 8])
    target_flat = np.tile(flat, len(obj.Rijk_list))
    r_bounds = np.arange(len(obj.Rijk_list) + 1) * len(flat)
    HR, SR = stream_project_to_real_space(obj, ks, rm, "schur", obj.Rijk_list, target_flat, r_bounds, n_workers=1)
    H_dense, S_dense = dense_reference(obj, ks, rm, keep, "schur")
    np.testing.assert_allclose(HR, H_dense.real.reshape(len(r_bounds) - 1, -1)[:, flat].reshape(-1), atol=1e-10)
    np.testing.assert_allclose(SR, S_dense.reshape(len(r_bounds) - 1, -1)[:, flat].reshape(-1), atol=1e-10)


def test_empty_r_blocks_are_skipped(structure):
    obj = SparseHamiltonianObj(structure.path)
    n_r = len(obj.Rijk_list)
    r_bounds = np.zeros(n_r + 1, dtype=int)
    r_bounds[1:] = 1  # one target, in the first R
    ks = build_uniform_kmesh((3, 3, 1))
    HR, SR = stream_project_to_real_space(obj, ks, [1, 2, 3], "truncate", obj.Rijk_list, np.array([0]), r_bounds)
    assert HR.shape == SR.shape == (1,)


def test_overlap_only_leaves_out_the_hamiltonian_and_gives_the_same_overlap(structure):
    obj = SparseHamiltonianObj(structure.path)
    ks = build_uniform_kmesh((3, 3, 1))
    target_flat, r_bounds = targets_for([0, 4, 5], obj)
    _, S_full = stream_project_to_real_space(obj, ks, [1, 2, 3], "schur", obj.Rijk_list, target_flat, r_bounds)
    H, S_only = stream_project_to_real_space(
        obj, ks, [1, 2, 3], "schur", obj.Rijk_list, target_flat, r_bounds, overlap_only=True
    )
    assert H is None
    np.testing.assert_allclose(S_only, S_full, atol=1e-12)


def test_low_memory_still_runs_and_warns_when_nothing_fits(monkeypatch, structure, caplog):
    monkeypatch.setattr(kspace, "available_memory_bytes", lambda: 0)
    obj = SparseHamiltonianObj(structure.path)
    ks = build_uniform_kmesh((3, 3, 1))
    target_flat, r_bounds = targets_for([0, 4, 5], obj)
    with caplog.at_level(logging.WARNING, logger=kspace.__name__):
        HR, _ = stream_project_to_real_space(
            obj, ks, [1, 2, 3], "truncate", obj.Rijk_list, target_flat, r_bounds, n_workers=4
        )
    assert "even one worker may not fit" in caplog.text
    H_dense, _ = dense_reference(obj, ks, [1, 2, 3], [0, 4, 5], "truncate")
    np.testing.assert_allclose(HR, H_dense.real.reshape(-1), atol=1e-10)


@pytest.mark.parametrize(
    ("rm", "mode", "message"),
    [
        ([0], "other", "Unsupported reduction_mode"),
        ([6], "schur", "out of range"),
        ([-1], "schur", "out of range"),
        (list(range(6)), "schur", "Cannot remove all"),
    ],
)
def test_invalid_streaming_arguments_are_rejected(structure, rm, mode, message):
    obj = SparseHamiltonianObj(structure.path)
    with pytest.raises(ValueError, match=message):
        stream_project_to_real_space(
            obj, np.zeros((1, 3)), rm, mode, obj.Rijk_list, np.array([0]), np.zeros(len(obj.Rijk_list) + 1)
        )
