"""Real harmonics in the converted FHI-aims convention and their rotation matrices."""

import numpy as np
import pytest

from hamilflow.overlap.rotations import real_sph_harm, real_wigner, rotation_from_z


def random_rotations(n, seed=0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(n, 3, 3)))
    q *= np.sign(np.linalg.det(q))[:, None, None]
    return q


def aims_sheval4(v):
    """FHI-aims ``SHEval4`` (src/basis_sets/SHEval.f90) for unit vectors, (n, 25), then the
    deepx_dock phase: -1 on odd m > 0."""
    x, y, z = v.T
    z2 = z * z
    p = np.zeros((len(v), 25))
    p[:, 0] = 0.28209479177387814347
    p[:, 2] = 0.48860251190291992159 * z
    p[:, 6] = 0.94617469575756001809 * z2 - 0.31539156525252000603
    p[:, 12] = z * (1.8658816629505769571 * z2 - 1.1195289977703461742)
    p[:, 20] = 1.9843134832984429429 * z * p[:, 12] - 1.0062305898749053634 * p[:, 6]
    c0, s0 = x, y
    for idx_c, idx_s, f in [
        (3, 1, -0.48860251190291992159 * np.ones_like(z)),
        (7, 5, -1.0925484305920790705 * z),
        (13, 11, -2.2852289973223286808 * z2 + 0.45704579946446573616),
        (21, 19, z * (-4.6833258049010241757 * z2 + 2.0071396306718675039)),
    ]:
        p[:, idx_c] = f * c0
        p[:, idx_s] = -f * s0
    c1, s1 = x * c0 - y * s0, x * s0 + y * c0
    for idx_c, idx_s, f in [
        (8, 4, 0.54627421529603953527 * np.ones_like(z)),
        (14, 10, 1.4453057213202770277 * z),
        (22, 18, 3.3116114351514600633 * z2 - 0.47308734787878000905),
    ]:
        p[:, idx_c] = f * c1
        p[:, idx_s] = f * s1
    c0, s0 = x * c1 - y * s1, x * s1 + y * c1
    for idx_c, idx_s, f in [
        (15, 9, -0.59004358992664351035 * np.ones_like(z)),
        (23, 17, -1.7701307697799305310 * z),
    ]:
        p[:, idx_c] = f * c0
        p[:, idx_s] = -f * s0
    c1, s1 = x * c0 - y * s0, x * s0 + y * c0
    p[:, 24] = 0.62583573544917613459 * c1
    p[:, 16] = 0.62583573544917613459 * s1
    for l in range(5):
        for m in range(1, l + 1, 2):
            p[:, l * l + l + m] *= -1
    return p


def test_matches_aims_sheval_with_deepx_phase():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(50, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    ref = aims_sheval4(v)
    for l in range(5):
        np.testing.assert_allclose(real_sph_harm(l, v), ref[:, l * l:(l + 1) ** 2], atol=1e-13)


def test_p_orbitals_are_y_z_x():
    c = np.sqrt(3 / (4 * np.pi))
    np.testing.assert_allclose(real_sph_harm(1, np.eye(3)), c * np.eye(3)[:, [1, 2, 0]], atol=1e-15)


def test_unnormalized_input_and_zero_vector():
    np.testing.assert_allclose(real_sph_harm(2, [[0, 0, 3.0]]), real_sph_harm(2, [[0, 0, 1.0]]))
    with pytest.raises(ValueError, match="zero vector"):
        real_sph_harm(1, [[0.0, 0.0, 0.0]])


@pytest.mark.parametrize("l", range(7))
def test_wigner_orthogonal_and_reproduces_rotated_harmonics(l):
    Q = random_rotations(6)
    M = real_wigner(l, Q)
    eye = np.eye(2 * l + 1)
    np.testing.assert_allclose(M @ np.swapaxes(M, -1, -2), np.broadcast_to(eye, M.shape), atol=1e-12)
    v = np.random.default_rng(2).normal(size=(10, 3))
    lhs = real_sph_harm(l, np.einsum("nij,sj->nsi", Q, v))  # Y(Q v)
    rhs = np.einsum("nkm,sm->nsk", M, real_sph_harm(l, v))
    np.testing.assert_allclose(lhs, rhs, atol=1e-12)


@pytest.mark.parametrize("l", range(5))
def test_wigner_composition(l):
    Q1, Q2 = random_rotations(4, seed=3), random_rotations(4, seed=4)
    np.testing.assert_allclose(real_wigner(l, Q1 @ Q2), real_wigner(l, Q1) @ real_wigner(l, Q2), atol=1e-12)


def test_wigner_l1_is_permuted_rotation():
    Q = random_rotations(3)
    perm = [1, 2, 0]  # (y, z, x)
    np.testing.assert_allclose(real_wigner(1, Q), Q[:, perm][:, :, perm], atol=1e-13)


def test_rotation_from_z():
    rng = np.random.default_rng(5)
    d = np.concatenate([rng.normal(size=(20, 3)), np.eye(3), -np.eye(3)]) * 2.5
    Q = rotation_from_z(d)
    np.testing.assert_allclose(Q @ np.swapaxes(Q, -1, -2), np.broadcast_to(np.eye(3), Q.shape), atol=1e-13)
    np.testing.assert_allclose(np.linalg.det(Q), 1.0, atol=1e-13)
    np.testing.assert_allclose(Q[..., 2], d / np.linalg.norm(d, axis=-1, keepdims=True), atol=1e-14)
