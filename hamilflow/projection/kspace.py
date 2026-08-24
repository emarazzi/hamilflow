from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import threadpoolctl


def available_cpu_count() -> int:
    """
    Cores actually usable by this process, respecting cgroup/cpuset limits
    (e.g. SLURM's --cpus-per-task). os.cpu_count() reports the node's full
    physical core count regardless of allocation, which under a SLURM
    cpuset causes BLAS thread limits to be set far too high.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        # sched_getaffinity is POSIX-only (no Windows/macOS support).
        return os.cpu_count() or 1


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


def _schur_transform_chunk(
    Hk_chunk: np.ndarray,
    Sk_chunk: np.ndarray,
    remove_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Elimination transform for one k-chunk (Hamiltonian + overlap), run in a thread."""
    Tk, _, _ = build_elimination_tk(Sk_chunk, remove_indices)
    return apply_tk_projection(Hk_chunk, Sk_chunk, Tk)


def _schur_transform_chunk_overlap_only(
    Sk_chunk: np.ndarray,
    remove_indices: list[int],
) -> np.ndarray:
    """Overlap-only counterpart of _schur_transform_chunk."""
    Tk, _, _ = build_elimination_tk(Sk_chunk, remove_indices)
    Tc = np.conjugate(np.swapaxes(Tk, 1, 2))
    return np.matmul(np.matmul(Tc, Sk_chunk), Tk)


def _truncation_transform_chunk(
    Hk_chunk: np.ndarray,
    Sk_chunk: np.ndarray,
    remove_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Direct truncation for one k-chunk (Hamiltonian + overlap), run in a thread."""
    _nk, nb, nb2 = Hk_chunk.shape
    if nb != nb2:
        raise ValueError(f"Hk/Sk must be square in last two dims, got {Hk_chunk.shape}")

    rm = sorted(set(int(i) for i in remove_indices))
    if len(rm) == 0:
        return Hk_chunk, Sk_chunk
    if rm[0] < 0 or rm[-1] >= nb:
        raise ValueError(f"remove_indices out of range for Nb={nb}: {rm}")

    rm_set = set(rm)
    keep = np.array([i for i in range(nb) if i not in rm_set], dtype=int)
    if keep.size == 0:
        raise ValueError("Cannot remove all orbitals")

    Hk_new = Hk_chunk[:, keep, :][:, :, keep]
    Sk_new = Sk_chunk[:, keep, :][:, :, keep]
    return Hk_new, Sk_new


def _truncation_transform_chunk_overlap_only(
    Sk_chunk: np.ndarray,
    remove_indices: list[int],
) -> np.ndarray:
    """Overlap-only counterpart of _truncation_transform_chunk."""
    _nk, nb, nb2 = Sk_chunk.shape
    if nb != nb2:
        raise ValueError(f"Sk must be square in last two dims, got {Sk_chunk.shape}")

    rm = sorted(set(int(i) for i in remove_indices))
    if len(rm) == 0:
        return Sk_chunk
    if rm[0] < 0 or rm[-1] >= nb:
        raise ValueError(f"remove_indices out of range for Nb={nb}: {rm}")

    rm_set = set(rm)
    keep = np.array([i for i in range(nb) if i not in rm_set], dtype=int)
    if keep.size == 0:
        raise ValueError("Cannot remove all orbitals")

    return Sk_chunk[:, keep, :][:, :, keep]


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


def apply_custom_kspace_transform_overlap_only(
    Sk: np.ndarray,
    remove_indices: list[int],
) -> np.ndarray:
    """Apply Schur-style elimination transform to overlap only in k-space."""
    if Sk.ndim != 3:
        raise ValueError(f"Sk must have shape (Nk, Nb, Nb), got {Sk.shape}")

    rm = sorted(set(int(i) for i in remove_indices))
    if len(rm) == 0:
        return Sk

    Tk, _, _ = build_elimination_tk(Sk, rm)
    Tc = np.conjugate(np.swapaxes(Tk, 1, 2))
    Sk_new = np.matmul(np.matmul(Tc, Sk), Tk)
    return Sk_new


def apply_truncation_kspace_transform_overlap_only(
    Sk: np.ndarray,
    remove_indices: list[int],
) -> np.ndarray:
    """Reduce overlap matrix by direct truncation (delete rows/cols) on all k points."""
    if Sk.ndim != 3:
        raise ValueError(f"Sk must have shape (Nk, Nb, Nb), got {Sk.shape}")

    _nk, nb, nb2 = Sk.shape
    if nb != nb2:
        raise ValueError(f"Sk must be square in last two dims, got {Sk.shape}")

    rm = sorted(set(int(i) for i in remove_indices))
    if len(rm) == 0:
        return Sk
    if rm[0] < 0 or rm[-1] >= nb:
        raise ValueError(f"remove_indices out of range for Nb={nb}: {rm}")

    rm_set = set(rm)
    keep = np.array([i for i in range(nb) if i not in rm_set], dtype=int)
    if keep.size == 0:
        raise ValueError("Cannot remove all orbitals")

    Sk_new = Sk[:, keep, :][:, :, keep]
    return Sk_new


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


def sk_to_real(
    ks: np.ndarray,
    Sk: np.ndarray,
    Rijk_list: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Inverse Fourier transform k-space overlap matrix back to real space.

    Parameters
    ----------
    ks : np.ndarray, shape (Nk, 3)
        k-points in fractional coordinates.
    Sk : np.ndarray, shape (Nk, Nb, Nb)
        Overlap matrices in reciprocal space.
    Rijk_list : np.ndarray, shape (N_R, 3), dtype=int
        Lattice displacements for inter-cell hoppings.
    weights : np.ndarray, shape (Nk,), optional
        Weights for k-points. Default uses uniform 1/Nk weights.

    Returns
    -------
    SR : np.ndarray, shape (N_R, Nb, Nb)
        Overlap matrices in real space on Rijk_list.
    """
    from deepx_dock.compute.eigen.matrix_obj import AOMatrixK

    ks = np.asarray(ks)
    Sk = np.asarray(Sk)
    Rs = np.asarray(Rijk_list)

    if ks.ndim != 2 or ks.shape[1] != 3:
        raise ValueError(f"ks must have shape (Nk, 3), got {ks.shape}")
    if Rs.ndim != 2 or Rs.shape[1] != 3:
        raise ValueError(f"Rijk_list must have shape (N_R, 3), got {Rs.shape}")
    if Sk.ndim != 3:
        raise ValueError(f"Sk must have shape (Nk, Nb, Nb), got {Sk.shape}")
    if Sk.shape[0] != ks.shape[0]:
        raise ValueError(f"Nk mismatch between ks and Sk: {ks.shape[0]} vs {Sk.shape[0]}")

    SR = AOMatrixK(ks, Sk).k2r(Rs, weights=weights)
    return SR


def stream_project_to_real_space(
    obj,
    ks: np.ndarray,
    remove_indices: list[int],
    reduction_mode: str,
    Rijk_list: np.ndarray,
    n_workers: int = 1,
    overlap_only: bool = False,
) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Project the k-mesh and inverse-Fourier-transform back to real space,
    streaming over small k-chunks instead of materializing dense Hk/Sk for
    the whole mesh at once.

    ``obj.Sk_and_Hk(ks_chunk)`` (see ``hamilflow.sparse_hamiltonian``) only
    ever builds one dense (Nb, Nb) matrix per requested k-point, so calling
    it with the *entire* mesh at once would recreate an (Nk, Nb, Nb) dense
    array — the same order-of-magnitude OOM this module exists to avoid.

    The k-space transform (schur/truncate) is parallelized across chunks --
    its output is O(chunk_size * Nb_kept^2), so more workers cost little.
    The real-space IFT is deliberately run *serially*, one chunk at a time,
    in this (the calling) thread instead of inside the worker pool: its
    output is O(R_quantity * Nb_kept^2) *regardless of chunk size* (it's the
    contribution to the full R-grid, not proportional to how many k's went
    in), so running N of them concurrently costs N times that -- for
    structures where R_quantity is not small relative to Nb_kept (real
    materials with modest orbital-removal fractions routinely have this),
    that multiplies to more memory than the dense path ever used, which is
    the opposite of the point. Serializing the IFT bounds peak real-space
    memory to ~2x one chunk's contribution (the running accumulator plus
    the one currently being folded in), independent of n_workers.

    Correctness: the IFT is a weighted linear sum over k
    (``AOMatrixK.k2r``, weights default to uniform ``1/Nk``), so summing
    per-chunk partial IFTs is exactly equivalent to one IFT over the full
    mesh -- but only if each chunk uses ``weight = 1/total_nk`` (the full
    mesh size), not ``1/chunk_size``. Chunk results are summed, not
    averaged or concatenated.
    """
    rm = sorted(set(int(i) for i in remove_indices))
    total_nk = len(ks)
    n_chunks = max(1, min(int(n_workers), total_nk))
    k_chunks = [c for c in np.array_split(ks, n_chunks) if len(c) > 0]

    if reduction_mode == "schur":
        transform = _schur_transform_chunk_overlap_only if overlap_only else _schur_transform_chunk
    elif reduction_mode == "truncate":
        transform = _truncation_transform_chunk_overlap_only if overlap_only else _truncation_transform_chunk
    else:
        raise ValueError(f"Unsupported reduction_mode '{reduction_mode}'. Expected 'schur' or 'truncate'.")

    threads_per_worker = max(1, available_cpu_count() // max(1, len(k_chunks)))

    def process(ks_chunk: np.ndarray):
        """K-space only: build Hk/Sk for this chunk and reduce them. No IFT here."""
        if overlap_only:
            Sk_c = obj.get_Sk(ks_chunk)  # skips building Hk entirely -- unused in overlap_only mode
            return transform(Sk_c, rm)
        Sk_c, Hk_c = obj.Sk_and_Hk(ks_chunk)
        return transform(Hk_c, Sk_c, rm)

    # threadpoolctl mutates process-global BLAS thread state, not per-thread
    # state, so it must be set once around the whole pool (single caller,
    # single mutation) rather than inside each worker thread — calling it
    # concurrently from multiple threads races on that global state and can
    # crash the underlying BLAS library (seen as a segfault on some HPC
    # MKL/OpenBLAS builds, even when it happens to survive elsewhere).
    HR_new = None
    SR_new = None
    with threadpoolctl.threadpool_limits(limits=threads_per_worker):
        with ThreadPoolExecutor(max_workers=len(k_chunks)) as ex:
            future_to_chunk = {ex.submit(process, chunk): chunk for chunk in k_chunks}
            for fut in as_completed(future_to_chunk):
                ks_chunk = future_to_chunk[fut]
                weights_chunk = np.full(len(ks_chunk), 1.0 / total_nk)
                if overlap_only:
                    Sk_new_c = fut.result()
                    sr_c = sk_to_real(ks_chunk, Sk_new_c, Rijk_list, weights=weights_chunk)
                else:
                    Hk_new_c, Sk_new_c = fut.result()
                    hr_c, sr_c = hk_and_sk_to_real(ks_chunk, Hk_new_c, Sk_new_c, Rijk_list, weights=weights_chunk)
                    HR_new = hr_c if HR_new is None else HR_new + hr_c
                SR_new = sr_c if SR_new is None else SR_new + sr_c

    return HR_new, SR_new
