from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import threadpoolctl

from ..cpus import available_cpu_count  # noqa: F401  (re-exported)

logger = logging.getLogger(__name__)


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
# Never fewer than one k-point per chunk, so for very large Nb a chunk is
# simply one k-point (see _estimate_worker_bytes for the true per-worker cost).
_K_CHUNK_BYTES = 256 * 1024**2
# Max size of the per-worker IFT temporary (a group of R vectors times a
# slice of the reduced matrix columns).
_IFT_BLOCK_BYTES = 64 * 1024**2
# Max size of the per-worker row-block temporaries in the Schur/truncate step.
_ROW_BLOCK_BYTES = 128 * 1024**2
# R-blocks with fewer than nkp^2 / this many targets are IFT'd by gathering.
_GATHER_MAX_DENSITY_INV = 32
# Fraction of the available memory the workers are allowed to plan for.
_MEMORY_SAFETY = 0.8
# Schur reductions removing at least this many orbitals are dominated by
# large gemms (~16 * nkp^2 * n_rm flops per matrix and k-point vs O(Nb^2)
# for everything else). Concurrent multi-threaded calls scale badly with
# OpenBLAS (pthreads): for this gemm shape, 32 cores gave 953 / 646 / 352
# GFLOP/s total with 1 x 32 / 2 x 16 / 4 x 8 workers x BLAS threads. So by
# default such runs use one worker with every core in BLAS.
_BLAS_BOUND_MIN_REMOVED = 128

_C16 = np.dtype(np.complex128).itemsize


def _cgroup_memory_limit() -> int | None:
    """Tightest memory limit of this process's cgroup or any ancestor (v2 or v1), in bytes."""
    try:
        with open("/proc/self/cgroup", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return None

    candidates = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, rel = parts
        if controllers == "":
            candidates.append(("/sys/fs/cgroup", rel, "memory.max"))
        elif "memory" in controllers.split(","):
            candidates.append(("/sys/fs/cgroup/memory", rel, "memory.limit_in_bytes"))

    best = None
    for root, rel, fname in candidates:
        rel_parts = [p for p in rel.split("/") if p]
        for depth in range(len(rel_parts), -1, -1):
            path = os.path.join(root, *rel_parts[:depth], fname)
            try:
                with open(path, encoding="utf-8") as f:
                    raw = f.read().strip()
            except OSError:
                continue
            if raw == "max":
                continue
            try:
                val = int(raw)
            except ValueError:
                continue
            # cgroup v1 reports "unlimited" as a huge page-aligned number.
            if val < (1 << 60):
                best = val if best is None else min(best, val)
    return best


def _proc_status_bytes(field: str) -> int:
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def available_memory_bytes() -> int | None:
    """
    Memory this process can still use, respecting cgroup limits (what SLURM
    --mem sets), the SLURM env vars, the node's MemAvailable and RLIMIT_AS.
    ``None`` if nothing could be determined.
    """
    rss = _proc_status_bytes("VmRSS")
    bounds = []

    limit = _cgroup_memory_limit()
    if limit is not None:
        bounds.append(limit - rss)

    mem_node = os.environ.get("SLURM_MEM_PER_NODE")
    mem_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    try:
        if mem_node:
            bounds.append(int(mem_node) * 1024**2 - rss)
        elif mem_cpu:
            bounds.append(int(mem_cpu) * 1024**2 * available_cpu_count() - rss)
    except ValueError:
        pass

    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    bounds.append(int(line.split()[1]) * 1024)
                    break
    except (OSError, ValueError, IndexError):
        pass

    try:
        import resource

        soft, _ = resource.getrlimit(resource.RLIMIT_AS)
        if soft != resource.RLIM_INFINITY:
            bounds.append(soft - _proc_status_bytes("VmSize"))
    except (ImportError, OSError, ValueError):
        pass

    return max(0, min(bounds)) if bounds else None


def _estimate_worker_bytes(nb: int, nkp: int, n_rm: int, k_chunk_size: int, n_mats: int, nnz: int) -> int:
    """Peak extra memory of one worker in stream_project_to_real_space."""
    return int(
        _C16 * nb * nb  # one dense H(k) or S(k) at a time
        + _C16 * nnz  # phase-weighted entries while building it
        + _C16 * n_mats * k_chunk_size * nkp * nkp  # reduced chunk buffers
        + 3 * _C16 * n_rm * nkp  # Schur C, (M T)[rm] and the solve/product temporaries
        + 3 * _ROW_BLOCK_BYTES
        + _IFT_BLOCK_BYTES
    )


def _project_into(
    M: np.ndarray,
    C: np.ndarray | None,
    keep: np.ndarray,
    rm: np.ndarray,
    out: np.ndarray,
) -> None:
    """
    Write the reduced (nkp, nkp) matrix of one dense (Nb, Nb) M into ``out``.

    ``C is None``: truncation, out = M[keep][:, keep].
    Otherwise Schur with T(k) = [identity on kept rows; C on removed rows]
    (same T as ``build_elimination_tk``): out = T^dagger M T, evaluated as
    X = M[:, keep] + M[:, rm] C, out = X[keep] + C^dagger X[rm], without ever
    forming T(k) or a full (Nb, nkp) X. Rows of ``out`` are produced in
    blocks of at most ~``_ROW_BLOCK_BYTES``, so the only full-size arrays are
    ``M`` itself and ``out``.
    """
    nb = M.shape[0]
    nkp = len(keep)
    X_rm = None
    if C is not None:
        X_rm = M[np.ix_(rm, keep)]
        X_rm += M[np.ix_(rm, rm)] @ C

    block = max(1, _ROW_BLOCK_BYTES // (_C16 * max(nb, 1)))
    for b0 in range(0, nkp, block):
        b1 = min(b0 + block, nkp)
        M_rows = M[keep[b0:b1]]
        o = out[b0:b1]
        # mode='clip' (indices are valid): the default 'raise' buffers `out`.
        np.take(M_rows, keep, axis=1, out=o, mode="clip")
        if C is not None:
            o += M_rows[:, rm] @ C
            o += C[:, b0:b1].conj().T @ X_rm
        del M_rows


def _schur_coefficients(Sk: np.ndarray, keep: np.ndarray, rm: np.ndarray) -> np.ndarray:
    """C = -S_rr^{-1} S_rk, the removed-orbital rows of T(k)."""
    return -np.linalg.solve(Sk[np.ix_(rm, rm)], Sk[np.ix_(rm, keep)])


def _reduce_one_k(
    Hk: np.ndarray | None,
    Sk: np.ndarray,
    keep: np.ndarray,
    rm: np.ndarray,
    reduction_mode: str,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Reduce one k-point's dense (Nb, Nb) H/S to the kept basis (allocating the outputs)."""
    if len(rm) == 0:
        return Hk, Sk
    C = _schur_coefficients(Sk, keep, rm) if reduction_mode == "schur" else None
    nkp = len(keep)
    S_new = np.empty((nkp, nkp), dtype=np.complex128)
    _project_into(Sk, C, keep, rm, S_new)
    H_new = None
    if Hk is not None:
        H_new = np.empty((nkp, nkp), dtype=np.complex128)
        _project_into(Hk, C, keep, rm, H_new)
    return H_new, S_new


def stream_project_to_real_space(
    obj,
    ks: np.ndarray,
    remove_indices: list[int],
    reduction_mode: str,
    Rijk_list: np.ndarray,
    target_flat: np.ndarray,
    r_bounds: np.ndarray,
    n_workers: int | None = None,
    overlap_only: bool = False,
    k_chunk_size: int | None = None,
) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Project the k-mesh and inverse-Fourier-transform back to real space,
    accumulating only a sparse set of real-space target entries.

    Returns ``(HR_re, SR)``: for every target (see
    ``io.ReducedOutputLayout``; entries of R index ``r`` are
    ``target_flat[r_bounds[r]:r_bounds[r + 1]]``, flat ``a * nkp + b``
    indices into the reduced matrix, sorted within each R), the real part of
    the IFT'd reduced Hamiltonian (``None`` with ``overlap_only``) and the
    complex IFT'd reduced overlap. Only the real part of H is kept because
    the output files only store real entries.

    Memory: the dense real-space (N_R, Nb_kept, Nb_kept) arrays of the
    straightforward approach are never formed -- the accumulators have one
    scalar per target, i.e. the size of the output files. Per worker, the
    peak is one dense (Nb, Nb) H(k) or S(k) (built from the sparse blocks
    by ``obj._dense_k``, freed right after its reduction), the chunk of
    reduced matrices (``k_chunk_size`` k-points, by default
    ~``_K_CHUNK_BYTES`` but at least one), the Schur coefficients, and
    row/IFT blocks of bounded size; see ``_estimate_worker_bytes``. For
    large Nb that is several dense matrices per worker, so ``n_workers`` is
    capped to what fits in ``available_memory_bytes()``; spare CPUs go to
    BLAS inside each worker instead.

    ``n_workers=None`` picks the split automatically: one worker with all
    CPUs in BLAS for gemm-bound Schur reductions (at least
    ``_BLAS_BOUND_MIN_REMOVED`` removed orbitals), otherwise one worker per
    CPU. An explicit ``n_workers`` is honoured (still capped by memory and by
    the number of k-points); BLAS gets ``available_cpu_count() // n_workers``
    threads per worker.

    Parallelism: chunks are processed by worker threads; each worker also
    folds its own chunk into the shared accumulators (one lock per R
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
    nkp2 = nkp * nkp
    n_mats = 1 if overlap_only else 2
    schur = reduction_mode == "schur" and len(rm) > 0

    ks = np.asarray(ks, dtype=np.float64)
    total_nk = len(ks)
    if n_workers is None:
        blas_bound = schur and len(rm) >= _BLAS_BOUND_MIN_REMOVED
        n_workers = 1 if blas_bound else available_cpu_count()
        worker_mode = "auto, BLAS-bound" if blas_bound else "auto"
    else:
        worker_mode = "requested"
    n_workers = max(1, min(int(n_workers), total_nk))
    if k_chunk_size is None:
        k_chunk_size = max(1, _K_CHUNK_BYTES // max(1, n_mats * nkp2 * _C16))
        k_chunk_size = min(k_chunk_size, -(-total_nk // n_workers))
    k_chunk_size = max(1, int(k_chunk_size))

    Rs = np.asarray(Rijk_list, dtype=np.float64)
    r_bounds = np.asarray(r_bounds, dtype=np.int64)
    target_flat = np.asarray(target_flat)
    n_targets = len(target_flat)
    active_r = [r for r in range(len(Rs)) if r_bounds[r + 1] > r_bounds[r]]
    r_locks = {r: threading.Lock() for r in active_r}

    HR_acc = None if overlap_only else np.zeros(n_targets, dtype=np.float64)
    SR_acc = np.zeros(n_targets, dtype=np.complex128)

    # Cap the number of concurrent workers by memory (accumulators above are
    # already allocated, so available_memory_bytes() accounts for them).
    s_obj = obj._s_obj
    nnz = max(
        len(s_obj._entries_expanded),
        0 if overlap_only else len(obj._entries_expanded),
    )
    worker_bytes = _estimate_worker_bytes(nb, nkp, len(rm) if schur else 0, k_chunk_size, n_mats, nnz)
    avail = available_memory_bytes()
    mem_note = "memory limit unknown"
    log_level = logging.INFO
    if avail is not None:
        fit = int(_MEMORY_SAFETY * avail // max(1, worker_bytes))
        mem_note = f"{avail / 1024**3:.1f} GiB available, ~{worker_bytes / 1024**3:.2f} GiB per worker"
        if fit < n_workers:
            n_workers = max(1, fit)
            if fit < 1:
                mem_note += " -- WARNING: even one worker may not fit"
                log_level = logging.WARNING
            k_chunk_size = min(k_chunk_size, max(1, -(-total_nk // n_workers)))
    k_chunks = [ks[i : i + k_chunk_size] for i in range(0, total_nk, k_chunk_size)]
    n_workers = max(1, min(n_workers, len(k_chunks)))
    threads_per_worker = max(1, available_cpu_count() // n_workers)
    logger.log(
        log_level,
        f"[projection] Nb={nb} kept={nkp} removed={len(rm)} Nk={total_nk} "
        f"chunk={k_chunk_size} workers={n_workers} ({worker_mode}) blas_threads={threads_per_worker} ({mem_note})",
    )

    # Per R, the IFT either gathers the targets' columns for every k-point
    # (cost ~ n_targets_r * Nk, scattered reads) or contracts reduced
    # R-blocks with BLAS and picks the targets afterwards (cost ~ nkp^2 * Nk,
    # but at gemm speed). The block product wins unless the R-block is very
    # sparse (large supercells, distant R). Block products are done for
    # groups of R and slices of columns so the temporary stays within
    # _IFT_BLOCK_BYTES.
    dense_r = [r for r in active_r if (r_bounds[r + 1] - r_bounds[r]) * _GATHER_MAX_DENSITY_INV >= nkp2]
    dense_set = set(dense_r)
    sparse_r = [r for r in active_r if r not in dense_set]
    bytes_per_r = n_mats * nkp2 * _C16
    r_group = max(1, _IFT_BLOCK_BYTES // bytes_per_r)
    col_block = nkp2 if r_group > 1 else max(1, _IFT_BLOCK_BYTES // (n_mats * _C16))

    def process(ks_chunk: np.ndarray) -> None:
        nkc = len(ks_chunk)
        Sk_red = np.empty((nkc, nkp, nkp), dtype=np.complex128)
        Hk_red = None if overlap_only else np.empty((nkc, nkp, nkp), dtype=np.complex128)
        for i, k in enumerate(ks_chunk):
            Sk = s_obj._dense_k(k)
            C = _schur_coefficients(Sk, keep, rm) if schur else None
            _project_into(Sk, C, keep, rm, Sk_red[i])
            del Sk
            if Hk_red is not None:
                Hk = obj._dense_k(k)
                _project_into(Hk, C, keep, rm, Hk_red[i])
                del Hk
            del C

        S_flat = Sk_red.reshape(nkc, -1)
        H_flat = None if Hk_red is None else Hk_red.reshape(nkc, -1)
        phase = np.exp(-2j * np.pi * (Rs @ ks_chunk.T)) / total_nk

        def fold(r: int, s: int, e: int, s_part: np.ndarray, h_part: np.ndarray | None) -> None:
            with r_locks[r]:
                SR_acc[s:e] += s_part
                if h_part is not None:
                    HR_acc[s:e] += h_part

        # Sparse R blocks: gather the targets' k-columns and contract them.
        for r in sparse_r:
            s, e = r_bounds[r], r_bounds[r + 1]
            idx = target_flat[s:e]
            ph = phase[r]
            fold(r, s, e, ph @ S_flat[:, idx], None if H_flat is None else (ph @ H_flat[:, idx]).real)

        # Dense R blocks: one gemm per (group of R, column slice), then pick
        # the targets falling in that slice (targets are sorted within an R).
        for g in range(0, len(dense_r), r_group):
            rs = dense_r[g : g + r_group]
            ph = phase[rs]
            for c0 in range(0, nkp2, col_block):
                c1 = min(c0 + col_block, nkp2)
                S_R = ph @ S_flat[:, c0:c1]
                H_R = None if H_flat is None else ph @ H_flat[:, c0:c1]
                for j, r in enumerate(rs):
                    rb0, rb1 = r_bounds[r], r_bounds[r + 1]
                    idx_r = target_flat[rb0:rb1]
                    s = rb0 + int(np.searchsorted(idx_r, c0))
                    e = rb0 + int(np.searchsorted(idx_r, c1))
                    if s == e:
                        continue
                    idx = target_flat[s:e] - c0
                    fold(r, s, e, S_R[j, idx], None if H_R is None else H_R[j, idx].real)
                del S_R, H_R

    # threadpoolctl mutates process-global BLAS thread state, not per-thread
    # state, so it must be set once around the whole pool (single caller,
    # single mutation) rather than inside each worker thread — calling it
    # concurrently from multiple threads races on that global state and can
    # crash the underlying BLAS library (seen as a segfault on some HPC
    # MKL/OpenBLAS builds, even when it happens to survive elsewhere).
    with threadpoolctl.threadpool_limits(limits=threads_per_worker):
        if n_workers == 1:
            for chunk in k_chunks:
                process(chunk)
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                for _ in ex.map(process, k_chunks):
                    pass

    return HR_acc, SR_acc
