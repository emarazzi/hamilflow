from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

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


# Target size of one chunk's reduced k-space matrices (H' and S') per worker.
_K_CHUNK_BYTES = 256 * 1024**2
# Max size of the per-worker temporary real-space block (a group of full
# reduced R-blocks) produced by one IFT gemm.
_IFT_BLOCK_BYTES = 64 * 1024**2
# R-blocks with fewer than nkp^2 / this many targets are IFT'd by gathering.
_GATHER_MAX_DENSITY_INV = 32


def _reduce_one_k(
    Hk: np.ndarray | None,
    Sk: np.ndarray,
    keep: np.ndarray,
    rm: np.ndarray,
    reduction_mode: str,
) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Reduce one k-point's dense (Nb, Nb) H/S to the kept basis.

    Schur: same T(k) as ``build_elimination_tk`` (identity on kept rows,
    C = -S_rr^{-1} S_rk on removed rows), but T^dagger M T is evaluated
    blockwise as X = M[:, keep] + M[:, rm] C, then X[keep] + C^dagger X[rm],
    so T(k) is never built and the cost is O(Nb^2 * N_rm) instead of O(Nb^3).
    """
    if len(rm) == 0:
        return Hk, Sk
    if reduction_mode == "truncate":
        H_new = None if Hk is None else Hk[np.ix_(keep, keep)]
        return H_new, Sk[np.ix_(keep, keep)]

    C = -np.linalg.solve(Sk[np.ix_(rm, rm)], Sk[np.ix_(rm, keep)])
    Ch = C.conj().T

    def project(M: np.ndarray) -> np.ndarray:
        X = M[:, keep]
        X += M[:, rm] @ C
        out = X[keep]
        out += Ch @ X[rm]
        return out

    H_new = None if Hk is None else project(Hk)
    return H_new, project(Sk)


def stream_project_to_real_space(
    obj,
    ks: np.ndarray,
    remove_indices: list[int],
    reduction_mode: str,
    Rijk_list: np.ndarray,
    target_flat: np.ndarray,
    r_bounds: np.ndarray,
    n_workers: int = 1,
    overlap_only: bool = False,
    k_chunk_size: int | None = None,
) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Project the k-mesh and inverse-Fourier-transform back to real space,
    accumulating only a sparse set of real-space target entries.

    Returns ``(HR_re, SR)``: for every target (see
    ``io.ReducedOutputLayout``; entries of R index ``r`` are
    ``target_flat[r_bounds[r]:r_bounds[r + 1]]``, flat ``a * nkp + b``
    indices into the reduced matrix), the real part of the IFT'd reduced
    Hamiltonian (``None`` with ``overlap_only``) and the complex IFT'd
    reduced overlap. Only the real part of H is kept because the output
    files only store real entries.

    Memory: the dense real-space (N_R, Nb_kept, Nb_kept) arrays of the
    straightforward approach are never formed -- the accumulators have one
    scalar per target, i.e. the size of the output files. In k-space, each
    worker holds one k-chunk of *reduced* matrices (sized by
    ``k_chunk_size``, by default ~``_K_CHUNK_BYTES``, independent of
    ``n_workers`` and of the mesh size), one dense (Nb, Nb) H/S pair at a
    time (``obj._dense_k`` builds a single k-point from the sparse blocks),
    and one IFT temporary of at most ~``_IFT_BLOCK_BYTES``.

    Parallelism: chunks are processed by ``n_workers`` threads; each worker
    also folds its own chunk into the shared accumulators (one lock per R
    block), so the IFT runs in parallel without per-worker copies of the
    real-space result.

    Correctness: the IFT is a weighted linear sum over k, so summing
    per-chunk partial IFTs with ``weight = 1/total_nk`` is exactly the full
    mesh IFT (uniform weights, as in ``AOMatrixK.k2r``).
    """
    if reduction_mode not in ("schur", "truncate"):
        raise ValueError(f"Unsupported reduction_mode '{reduction_mode}'. Expected 'schur' or 'truncate'.")

    nb = int(obj._mat_dim)
    rm = np.array(sorted(set(int(i) for i in remove_indices)), dtype=np.int64)
    if len(rm) and (rm[0] < 0 or rm[-1] >= nb):
        raise ValueError(f"remove_indices out of range for Nb={nb}: {rm.tolist()}")
    keep = np.setdiff1d(np.arange(nb, dtype=np.int64), rm)
    if keep.size == 0:
        raise ValueError("Cannot remove all orbitals")
    nkp = len(keep)

    ks = np.asarray(ks, dtype=np.float64)
    total_nk = len(ks)
    n_workers = max(1, min(int(n_workers), total_nk))
    if k_chunk_size is None:
        bytes_per_k = (1 if overlap_only else 2) * nkp * nkp * np.dtype(np.complex128).itemsize
        k_chunk_size = max(1, _K_CHUNK_BYTES // max(1, bytes_per_k))
        k_chunk_size = min(k_chunk_size, -(-total_nk // n_workers))
    k_chunk_size = max(1, int(k_chunk_size))
    k_chunks = [ks[i : i + k_chunk_size] for i in range(0, total_nk, k_chunk_size)]

    Rs = np.asarray(Rijk_list, dtype=np.float64)
    r_bounds = np.asarray(r_bounds, dtype=np.int64)
    target_flat = np.asarray(target_flat, dtype=np.int64)
    n_targets = len(target_flat)
    active_r = [r for r in range(len(Rs)) if r_bounds[r + 1] > r_bounds[r]]
    r_locks = {r: threading.Lock() for r in active_r}

    # Per R, the IFT either gathers the targets' columns for every k-point
    # (cost ~ n_targets_r * Nk, scattered reads) or contracts the whole
    # reduced R-block with BLAS and picks the targets afterwards (cost
    # ~ nkp^2 * Nk, but at gemm speed). The block product wins unless the
    # R-block is very sparse (large supercells, distant R).
    nkp2 = nkp * nkp
    dense_r = [r for r in active_r if (r_bounds[r + 1] - r_bounds[r]) * _GATHER_MAX_DENSITY_INV >= nkp2]
    dense_set = set(dense_r)
    sparse_r = [r for r in active_r if r not in dense_set]
    bytes_per_r = (1 if overlap_only else 2) * nkp2 * np.dtype(np.complex128).itemsize
    r_group = max(1, _IFT_BLOCK_BYTES // bytes_per_r)

    HR_acc = None if overlap_only else np.zeros(n_targets, dtype=np.float64)
    SR_acc = np.zeros(n_targets, dtype=np.complex128)

    s_obj = obj._s_obj

    def process(ks_chunk: np.ndarray) -> None:
        nkc = len(ks_chunk)
        Sk_red = np.empty((nkc, nkp, nkp), dtype=np.complex128)
        Hk_red = None if overlap_only else np.empty((nkc, nkp, nkp), dtype=np.complex128)
        for i, k in enumerate(ks_chunk):
            Sk = s_obj._dense_k(k)
            Hk = None if overlap_only else obj._dense_k(k)
            H_new, S_new = _reduce_one_k(Hk, Sk, keep, rm, reduction_mode)
            del Hk, Sk
            Sk_red[i] = S_new
            if Hk_red is not None:
                Hk_red[i] = H_new

        S_flat = Sk_red.reshape(nkc, -1)
        H_flat = None if Hk_red is None else Hk_red.reshape(nkc, -1)
        phase = np.exp(-2j * np.pi * (Rs @ ks_chunk.T)) / total_nk

        def fold(r: int, s_part: np.ndarray, h_part: np.ndarray | None) -> None:
            s, e = r_bounds[r], r_bounds[r + 1]
            with r_locks[r]:
                SR_acc[s:e] += s_part
                if h_part is not None:
                    HR_acc[s:e] += h_part

        # Sparse R blocks: gather the targets' k-columns and contract them.
        for r in sparse_r:
            idx = target_flat[r_bounds[r] : r_bounds[r + 1]]
            ph = phase[r]
            fold(r, ph @ S_flat[:, idx], None if H_flat is None else (ph @ H_flat[:, idx]).real)

        # Dense R blocks: one zgemm per group of R (the full reduced R-blocks,
        # bounded by _IFT_BLOCK_BYTES), then pick the targets out of it.
        for g in range(0, len(dense_r), r_group):
            rs = dense_r[g : g + r_group]
            ph = phase[rs]
            S_R = ph @ S_flat
            H_R = None if H_flat is None else ph @ H_flat
            for j, r in enumerate(rs):
                idx = target_flat[r_bounds[r] : r_bounds[r + 1]]
                fold(r, S_R[j, idx], None if H_R is None else H_R[j, idx].real)
            del S_R, H_R

    # threadpoolctl mutates process-global BLAS thread state, not per-thread
    # state, so it must be set once around the whole pool (single caller,
    # single mutation) rather than inside each worker thread — calling it
    # concurrently from multiple threads races on that global state and can
    # crash the underlying BLAS library (seen as a segfault on some HPC
    # MKL/OpenBLAS builds, even when it happens to survive elsewhere).
    threads_per_worker = max(1, available_cpu_count() // n_workers)
    with threadpoolctl.threadpool_limits(limits=threads_per_worker):
        if n_workers == 1:
            for chunk in k_chunks:
                process(chunk)
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                for _ in ex.map(process, k_chunks):
                    pass

    return HR_acc, SR_acc
