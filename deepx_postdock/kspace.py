from __future__ import annotations

import numpy as np


def build_uniform_kmesh(nk: tuple[int, int, int]) -> np.ndarray:
    """Build a uniform fractional k-mesh in [0, 1)."""
    nx, ny, nz = nk
    xs = np.arange(nx, dtype=float) / nx
    ys = np.arange(ny, dtype=float) / ny
    zs = np.arange(nz, dtype=float) / nz
    kx, ky, kz = np.meshgrid(xs, ys, zs, indexing="ij")
    ks = np.column_stack([kx.ravel(), ky.ravel(), kz.ravel()])
    return ks


def apply_custom_kspace_transform(
    Hk: np.ndarray,
    Sk: np.ndarray,
    remove_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Schur-style elimination transform in k-space."""
    if Hk.shape != Sk.shape:
        raise ValueError(f"Hk/Sk shape mismatch: {Hk.shape} vs {Sk.shape}")
    if Hk.ndim != 3:
        raise ValueError(f"Hk/Sk must have shape (Nk, Nb, Nb), got {Hk.shape}")

    rm = sorted(set(int(i) for i in remove_indices))
    if len(rm) == 0:
        return Hk, Sk

    Tk, _, _ = build_elimination_tk(Sk, rm)
    Hk_new, Sk_new = apply_tk_projection(Hk, Sk, Tk)
    return Hk_new, Sk_new


def apply_truncation_kspace_transform(
    Hk: np.ndarray,
    Sk: np.ndarray,
    remove_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce k-space matrices by direct truncation (delete rows/cols) on all k points."""
    if Hk.shape != Sk.shape:
        raise ValueError(f"Hk/Sk shape mismatch: {Hk.shape} vs {Sk.shape}")
    if Hk.ndim != 3:
        raise ValueError(f"Hk/Sk must have shape (Nk, Nb, Nb), got {Hk.shape}")

    _nk, nb, nb2 = Hk.shape
    if nb != nb2:
        raise ValueError(f"Hk/Sk must be square in last two dims, got {Hk.shape}")

    rm = sorted(set(int(i) for i in remove_indices))
    if len(rm) == 0:
        return Hk, Sk
    if rm[0] < 0 or rm[-1] >= nb:
        raise ValueError(f"remove_indices out of range for Nb={nb}: {rm}")

    rm_set = set(rm)
    keep = np.array([i for i in range(nb) if i not in rm_set], dtype=int)
    if keep.size == 0:
        raise ValueError("Cannot remove all orbitals")

    Hk_new = Hk[:, keep, :][:, :, keep]
    Sk_new = Sk[:, keep, :][:, :, keep]
    return Hk_new, Sk_new


def build_elimination_tk(
    Sk: np.ndarray,
    remove_indices: list[int],
) -> tuple[np.ndarray, list[int], list[int]]:
    """Build elimination transform T(k) for removing a set of orbitals."""
    if Sk.ndim != 3:
        raise ValueError(f"Sk must have shape (Nk, Nb, Nb), got {Sk.shape}")

    nk, nb, nb2 = Sk.shape
    if nb != nb2:
        raise ValueError(f"Sk must be square in last two dims, got {Sk.shape}")

    rm = sorted(set(int(i) for i in remove_indices))
    if len(rm) == 0:
        eye = np.eye(nb, dtype=np.complex128)
        return np.broadcast_to(eye, (nk, nb, nb)).copy(), list(range(nb)), []
    if rm[0] < 0 or rm[-1] >= nb:
        raise ValueError(f"remove_indices out of range for Nb={nb}: {rm}")

    keep = [i for i in range(nb) if i not in rm]
    nkp = len(keep)
    if nkp == 0:
        raise ValueError("Cannot remove all orbitals")

    S_mm = Sk[:, rm, :][:, :, rm]
    S_mk = Sk[:, rm, :][:, :, keep]
    coeff = -np.linalg.solve(S_mm, S_mk)

    Tk = np.zeros((nk, nb, nkp), dtype=np.complex128)
    Tk[:, keep, :] = np.eye(nkp, dtype=np.complex128)[None, :, :]
    Tk[:, rm, :] = coeff
    return Tk, keep, rm


def apply_tk_projection(
    Hk: np.ndarray,
    Sk: np.ndarray,
    Tk: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply H'(k)=T(k)^dagger H(k) T(k), S'(k)=T(k)^dagger S(k) T(k)."""
    Tc = np.conjugate(np.swapaxes(Tk, 1, 2))
    Hk_new = np.matmul(np.matmul(Tc, Hk), Tk)
    Sk_new = np.matmul(np.matmul(Tc, Sk), Tk)
    return Hk_new, Sk_new


def k_to_r_operator(
    ks: np.ndarray,
    Rijk_list: np.ndarray,
    Mk: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Inverse transform operator blocks from k-space to real-space on a target R list."""
    ks = np.asarray(ks, dtype=float)
    Rs = np.asarray(Rijk_list, dtype=float)
    Mk = np.asarray(Mk)

    if ks.ndim != 2 or ks.shape[1] != 3:
        raise ValueError(f"ks must have shape (Nk, 3), got {ks.shape}")
    if Rs.ndim != 2 or Rs.shape[1] != 3:
        raise ValueError(f"Rijk_list must have shape (NR, 3), got {Rs.shape}")
    if Mk.ndim != 3:
        raise ValueError(f"Mk must have shape (Nk, Nrow, Ncol), got {Mk.shape}")
    if Mk.shape[0] != ks.shape[0]:
        raise ValueError(f"Nk mismatch between ks and Mk: {ks.shape[0]} vs {Mk.shape[0]}")

    nk = ks.shape[0]
    if weights is None:
        w = np.full(nk, 1.0 / nk, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1 or w.shape[0] != nk:
            raise ValueError(f"weights must have shape (Nk,), got {w.shape}")

    phase = np.exp(-2j * np.pi * np.matmul(Rs, ks.T))
    wr = phase * w[None, :]

    Mk_flat = Mk.reshape(nk, -1)
    MR_flat = np.matmul(wr, Mk_flat)
    return MR_flat.reshape(len(Rs), Mk.shape[1], Mk.shape[2])


def hk_and_sk_to_real(
    ks: np.ndarray,
    Hk: np.ndarray,
    Sk: np.ndarray,
    Rijk_list: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inverse Fourier transform k-space H/S matrices back to real space.

    Parameters
    ----------
    ks : np.ndarray, shape (Nk, 3)
        k-points in fractional coordinates.
    Hk : np.ndarray, shape (Nk, Nb, Nb)
        Hamiltonian matrices in reciprocal space.
    Sk : np.ndarray, shape (Nk, Nb, Nb)
        Overlap matrices in reciprocal space.
    Rijk_list : np.ndarray, shape (N_R, 3), dtype=int
        Lattice displacements for inter-cell hoppings.
    weights : np.ndarray, shape (Nk,), optional
        Weights for k-points. Default uses uniform 1/Nk weights.

    Returns
    -------
    HR : np.ndarray, shape (N_R, Nb, Nb)
        Hamiltonian matrices in real space on Rijk_list.
    SR : np.ndarray, shape (N_R, Nb, Nb)
        Overlap matrices in real space on Rijk_list.
    """
    from deepx_dock.compute.eigen.matrix_obj import AOMatrixK

    ks = np.asarray(ks)
    Hk = np.asarray(Hk)
    Sk = np.asarray(Sk)
    Rs = np.asarray(Rijk_list)

    if ks.ndim != 2 or ks.shape[1] != 3:
        raise ValueError(f"ks must have shape (Nk, 3), got {ks.shape}")
    if Rs.ndim != 2 or Rs.shape[1] != 3:
        raise ValueError(f"Rijk_list must have shape (N_R, 3), got {Rs.shape}")
    if Hk.shape != Sk.shape:
        raise ValueError(f"Hk/Sk shape mismatch: {Hk.shape} vs {Sk.shape}")
    if Hk.ndim != 3:
        raise ValueError(f"Hk/Sk must have shape (Nk, Nb, Nb), got {Hk.shape}")
    if Hk.shape[0] != ks.shape[0]:
        raise ValueError(f"Nk mismatch between ks and Hk: {ks.shape[0]} vs {Hk.shape[0]}")

    HR = AOMatrixK(ks, Hk).k2r(Rs, weights=weights)
    SR = AOMatrixK(ks, Sk).k2r(Rs, weights=weights)
    return HR, SR
