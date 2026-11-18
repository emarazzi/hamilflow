"""Real spherical harmonics in the DeepH (converted FHI-aims) convention and their rotation matrices.

Convention. FHI-aims evaluates its real harmonics with ``SHEval`` (``fast_ylm``, the default):
``Y_{l,m>0} = (-1)^m N_lm P_l^m(cos t) cos(m p)`` and ``Y_{l,-m} = N_lm P_l^m(cos t) sin(m p)``
(``P_l^m`` without the Condon-Shortley phase). deepx_dock's conversion multiplies every basis
function with odd ``m > 0`` by -1, so the converted (DeepH) basis uses the Wikipedia real
harmonics, with every prefactor positive:

    Y_{l,m}  = c_lm (d^m P_l / dz^m)(z) Re[(x + iy)^m]      (m > 0)
    Y_{l,-m} = c_lm (d^m P_l / dz^m)(z) Im[(x + iy)^m]      (m > 0)

on unit vectors (x, y, z), with ``c_lm > 0`` the orthonormalizing constant, ordered
``m = -l .. l``. For l = 1 this is ``sqrt(3 / 4 pi) (y, z, x)``.

Rotations. For a rotation ``Q`` (3x3, det +1), ``M^l(Q)`` is the orthogonal matrix with
``Y_l(Q v) = M^l(Q) Y_l(v)`` for every unit vector ``v``. It is built by least squares on a fixed
set of sample directions, so it is exact for whatever ``real_sph_harm`` returns. If ``f`` is an
orbital centred on atom A, then the two-centre block for the bond vector ``d = r Q z`` is

    S(d) = M^{la}(Q) S(r z) M^{lb}(Q)^T,

where ``S(r z)`` is the block of the same pair of shells along +z (diagonal in m, see `tables`).

All functions take an ``xp`` (NumPy or CuPy) array module and work batched over leading axes.
"""

from __future__ import annotations

from functools import cache
from math import factorial, pi, sqrt
from typing import Any

import numpy as np

L_MAX = 6

_N_SAMPLES = 64  # sample directions of the least-squares rotation fit; >= 2 L_MAX + 1


@cache
def _legendre_derivative_coeffs(l: int, m: int) -> np.ndarray:
    """Power-series coefficients (ascending) of ``c_lm d^m P_l(z) / dz^m``, normalized so that
    the real harmonics are orthonormal on the sphere."""
    p_l = np.polynomial.legendre.leg2poly([0] * l + [1])
    coeffs = np.polynomial.polynomial.polyder(p_l, m) if m > 0 else p_l
    norm = sqrt((2 * l + 1) / (4 * pi) * factorial(l - m) / factorial(l + m))
    if m > 0:
        norm *= sqrt(2.0)
    return np.asarray(coeffs, dtype=np.float64) * norm


def real_sph_harm(l: int, vecs: Any, xp: Any = np) -> Any:
    """Real harmonics ``Y_{l,m}``, ``m = -l..l``, of the directions ``vecs`` (..., 3), in the
    DeepH convention (module docstring). ``vecs`` need not be normalized. Returns (..., 2l+1)."""
    if not 0 <= l <= L_MAX:
        raise ValueError(f"l must be in [0, {L_MAX}], got {l}")
    vecs = xp.asarray(vecs, dtype=xp.float64)
    norm = xp.linalg.norm(vecs, axis=-1, keepdims=True)
    if bool(xp.any(norm == 0)):
        raise ValueError("real_sph_harm: zero vector has no direction")
    u = vecs / norm
    x, y, z = u[..., 0], u[..., 1], u[..., 2]
    out = xp.empty(vecs.shape[:-1] + (2 * l + 1,), dtype=xp.float64)
    # (x + iy)^m as real/imaginary parts, built up recursively
    cm, sm = xp.ones_like(x), xp.zeros_like(x)
    for m in range(l + 1):
        if m > 0:
            cm, sm = cm * x - sm * y, cm * y + sm * x
        a = xp.zeros_like(z)
        for c in _legendre_derivative_coeffs(l, m)[::-1]:  # Horner
            a = a * z + float(c)
        if m == 0:
            out[..., l] = a
        else:
            out[..., l + m] = a * cm
            out[..., l - m] = a * sm
    return out


@cache
def _sample_directions() -> np.ndarray:
    """Fixed, well-spread unit vectors (Fibonacci sphere)."""
    k = np.arange(_N_SAMPLES) + 0.5
    z = 1.0 - 2.0 * k / _N_SAMPLES
    phi = pi * (1.0 + sqrt(5.0)) * k
    rho = np.sqrt(1.0 - z * z)
    return np.stack([rho * np.cos(phi), rho * np.sin(phi), z], axis=-1)


@cache
def _sample_pinv(l: int) -> np.ndarray:
    """``pinv(Y_l(samples))``, (2l+1, n_samples)."""
    return np.linalg.pinv(real_sph_harm(l, _sample_directions()))


def real_wigner(l: int, Q: Any, xp: Any = np) -> Any:
    """``M^l(Q)`` (..., 2l+1, 2l+1) with ``Y_l(Q v) = M^l(Q) Y_l(v)``, for rotations ``Q``
    (..., 3, 3)."""
    Q = xp.asarray(Q, dtype=xp.float64)
    if l == 0:
        return xp.ones(Q.shape[:-2] + (1, 1), dtype=xp.float64)
    samples = xp.asarray(_sample_directions())
    rotated = xp.einsum("...ij,sj->...si", Q, samples)  # Q v_s
    B = real_sph_harm(l, rotated, xp=xp)  # (..., s, 2l+1) = Y(v_s) M^T
    Mt = xp.einsum("ks,...sn->...kn", xp.asarray(_sample_pinv(l)), B)
    return xp.swapaxes(Mt, -1, -2)


def rotation_from_z(dirs: Any, xp: Any = np) -> Any:
    """Rotations ``Q`` (..., 3, 3) with ``Q @ z = d / |d|`` for the vectors ``dirs`` (..., 3).

    The rotation about the bond axis is arbitrary (bond-frame blocks are invariant under it);
    it is chosen smooth away from ``d ~ -x``."""
    d = xp.asarray(dirs, dtype=xp.float64)
    u = d / xp.linalg.norm(d, axis=-1, keepdims=True)
    # helper axis: x, or y where u is nearly parallel to x
    helper = xp.zeros_like(u)
    near_x = xp.abs(u[..., 0]) > 0.9
    helper[..., 0] = xp.where(near_x, 0.0, 1.0)
    helper[..., 1] = xp.where(near_x, 1.0, 0.0)
    e2 = xp.cross(u, helper)
    e2 /= xp.linalg.norm(e2, axis=-1, keepdims=True)
    e1 = xp.cross(e2, u)
    return xp.stack([e1, e2, u], axis=-1)  # columns: Q x, Q y, Q z
