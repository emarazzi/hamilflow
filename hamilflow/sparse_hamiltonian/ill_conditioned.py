"""
Handling of an ill-conditioned overlap in the generalized eigenproblem H psi = E S psi.

Adapted from ``deepx_dock.compute.eigen.ill_conditioned`` (deepx-dock 0.9.11,
GPL-3.0-or-later), with the same interface. Two methods:

1. Window regularization: projects out the nearly linearly dependent
   combinations of the eigenvectors in an energy window.
2. Orbital removal: removes, globally over all k-points, the orbitals that make
   S(k) ill-conditioned.

``global_orbital_truncation`` selects the same orbitals as the deepx-dock
version, at a fraction of the cost: it computes only the lowest eigenpair of
S(k), by a warm-started block subspace iteration instead of a full dense
diagonalization, runs the k-points in threads, and stops tracking a k-point
once its S(k) is well conditioned (see its docstring).
"""

from __future__ import annotations

import concurrent.futures
import os
from typing import List, Optional, Tuple

import numpy as np
import threadpoolctl
from scipy.linalg import LinAlgError, cho_factor, cho_solve, cholesky, eigh

from .eigensolver import eigh_generalized


def window_regularization_eig(
    Hk: np.ndarray,
    Sk: np.ndarray,
    emin: float,
    emax: float,
    ill_threshold: float,
    fill_value: float = 1e4,
    return_vecs: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Eigenvalues of (Hk, Sk) with the ill-conditioned combinations in [emin, emax] projected out.

    The eigenvectors in the window (absolute energies) form the quality matrix
    M = Psi^dag Psi; its eigenvectors with eigenvalue <= ill_threshold are nearly
    linearly dependent combinations, and the window problem is re-solved without
    them. The removed slots are filled with ``fill_value`` and the result is sorted.
    """
    norb = Hk.shape[0]

    eigvals, eigvecs = eigh_generalized(Hk, Sk)

    mask = (eigvals >= emin) & (eigvals <= emax)
    n_window = np.sum(mask)

    if n_window == 0:
        return eigvals, (eigvecs if return_vecs else None)

    filtered_vals = eigvals[mask]
    filtered_vecs = eigvecs[:, mask]

    mass_matrix = filtered_vecs.T.conj() @ filtered_vecs
    mass_eigvals, mass_eigvecs = eigh(mass_matrix)

    project_mask = mass_eigvals > ill_threshold
    n_good = np.sum(project_mask)

    if n_good < n_window:
        good_mass_vecs = mass_eigvecs[:, project_mask]

        H_sub = np.diag(filtered_vals)
        M_sub = good_mass_vecs.T.conj() @ good_mass_vecs

        try:
            sub_vals, sub_vecs = eigh(H_sub, M_sub)
        except LinAlgError:
            return eigvals, (eigvecs if return_vecs else None)

        final_vecs = filtered_vecs @ (good_mass_vecs @ sub_vecs)

        final_vals = np.full(norb, fill_value)
        final_vals[:n_good] = sub_vals

        sort_idx = np.argsort(final_vals)
        final_vals = final_vals[sort_idx]
        final_vecs = final_vecs[:, sort_idx]

        return final_vals, (final_vecs if return_vecs else None)

    return eigvals, (eigvecs if return_vecs else None)


def _map_threads(fn, items: list, n_jobs: int = -1, parallel_k: bool = True) -> list:
    """`[fn(x) for x in items]`, spread over threads with the leftover CPU budget given to
    each thread's BLAS (as `SparseHamiltonianObj.diag`)."""
    if n_jobs < 0:
        n_jobs = os.cpu_count() or 1
    n_workers = max(1, min(n_jobs, len(items))) if parallel_k else 1
    n_blas_threads = max(1, n_jobs // n_workers)
    with threadpoolctl.threadpool_limits(limits=n_blas_threads, user_api="blas"):
        if n_workers <= 1:
            return [fn(x) for x in items]
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            return list(ex.map(fn, items))


# Below this size the dense LAPACK solver is cheap enough on its own.
_SUBSPACE_MIN_DIM = 300
# Block size and iteration cap of the subspace iteration.
_SUBSPACE_BLOCK = 32
_SUBSPACE_MAX_ITER = 100


def _lowest_eigpair_subspace(
    Sk: np.ndarray, x0: Optional[np.ndarray] = None
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """Lowest eigenpair of a positive-definite Sk by block inverse subspace iteration on a
    Cholesky factor of Sk: only multi-vector solves and products, which (unlike the memory-bound
    tridiagonal reduction of a dense eigh, or single-vector Lanczos) run at BLAS-3 speed.

    `x0` warm-starts the block (e.g. the previous block with the removed orbital's row
    deleted). Returns (value, vector, block), or (nan, None, None) unless the vector is accurate
    enough to pick the same largest-amplitude orbital as an exact solver: the Davis-Kahan bound
    `|r| / gap` on its error must stay well below the gap between its two largest amplitudes.
    """
    n = len(Sk)
    b = min(_SUBSPACE_BLOCK, n // 4)
    try:
        factor = cho_factor(Sk, lower=True, check_finite=False)
    except LinAlgError:
        return np.nan, None, None
    if x0 is None or x0.shape != (n, b):
        rng = np.random.default_rng(0)
        x0 = rng.standard_normal((n, b)) + 1j * rng.standard_normal((n, b))
    x = x0.astype(Sk.dtype, copy=False)
    s_norm = np.linalg.norm(Sk, ord=1)
    for _ in range(_SUBSPACE_MAX_ITER):
        q, _ = np.linalg.qr(cho_solve(factor, x, check_finite=False))
        sq = Sk @ q
        t = q.conj().T @ sq
        w, u = np.linalg.eigh(0.5 * (t + t.conj().T))
        x = q @ u
        r = np.linalg.norm(sq @ u[:, :2] - x[:, :2] * w[:2], axis=0)
        gap = w[1] - w[0] - r[1]
        amp = np.sort(np.abs(x[:, 0]))
        if gap > 0 and r[0] / gap < 0.1 * (amp[-1] - amp[-2]):
            return float(w[0]), x[:, 0], x
        if r[0] < 1e-13 * s_norm:  # converged, but too close to a tie to be sure
            break
    return np.nan, None, None


def _lowest_eigpair(
    Sk: np.ndarray, ill_threshold: float, x0: Optional[np.ndarray] = None
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """Lowest eigenvalue and eigenvector of Sk (and the subspace-iteration block, to warm-start
    the next call), or (inf, None, None) if all eigenvalues are above `ill_threshold` (checked
    with a Cholesky factorization of Sk - ill_threshold * 1).

    A Ritz value is an upper bound on the lowest eigenvalue, so a subspace-iteration value at or
    below the threshold already proves Sk ill-conditioned and the Cholesky test is skipped. With
    a warm start (Sk was ill-conditioned at the previous iteration and likely still is) the
    subspace iteration runs first; on a first visit the cheaper Cholesky test does."""

    def well_conditioned() -> bool:
        try:
            cholesky(Sk - ill_threshold * np.eye(len(Sk)), lower=True, check_finite=False)
            return True
        except LinAlgError:
            return False

    use_subspace = len(Sk) >= _SUBSPACE_MIN_DIM
    if (not use_subspace or x0 is None) and well_conditioned():
        return np.inf, None, None
    if use_subspace:
        val, vec, block = _lowest_eigpair_subspace(Sk, x0)
        if vec is not None:
            if val <= ill_threshold:
                return val, vec, block
            # above the threshold, not yet a proof (or Cholesky failed by round-off only)
            if x0 is None or well_conditioned():
                return np.inf, None, None
    vals, vecs = eigh(Sk, subset_by_index=[0, 0], check_finite=False)
    if vals[0] > ill_threshold:  # Cholesky failed by round-off only
        return np.inf, None, None
    return float(vals[0]), vecs[:, 0], None


def global_orbital_truncation(
    Sk_list: List[np.ndarray],
    ill_threshold: float,
    verbose: bool = True,
    n_jobs: int = -1,
    parallel_k: bool = True,
) -> List[int]:
    """
    Orbitals to keep so that S(k) has all eigenvalues above `ill_threshold` at every k-point.

    One orbital per iteration is removed from all k-points: the one with the largest
    amplitude in the lowest eigenvector of S(k), at the k-point where that eigenvalue is
    the lowest, until every S(k) is above the threshold.

    Removing an orbital takes a principal submatrix of S(k), which by Cauchy interlacing
    cannot lower its smallest eigenvalue. A k-point whose S(k) is above the threshold
    therefore stays above it and cannot be the worst k-point again, so it is dropped from
    the following iterations; this does not change which orbitals are removed. At the
    remaining k-points, a Cholesky factorization of S(k) - ill_threshold * 1 checks whether
    S(k) is above the threshold, and only otherwise the lowest eigenpair is computed, by a
    subspace iteration warm-started from the previous iteration's (see
    `_lowest_eigpair_subspace`; it falls back to LAPACK when its answer could differ).

    Returns the kept orbital indices (in the original numbering).
    """
    norb = Sk_list[0].shape[0]
    remaining = list(range(norb))

    # k-points that can still be (or become) the worst one; the others are never touched again
    active = list(range(len(Sk_list)))
    Sk_work = {ik: Sk_list[ik] for ik in active}
    blocks = {}  # subspace-iteration block of each k-point, to warm-start the next iteration

    iteration = 0
    while True:
        pairs = _map_threads(lambda ik: _lowest_eigpair(Sk_work[ik], ill_threshold, blocks.get(ik)),
                             active, n_jobs, parallel_k)
        ill = [(ik, val, vec) for ik, (val, vec, _) in zip(active, pairs) if vec is not None]
        blocks = {ik: block for ik, (_, _, block) in zip(active, pairs) if block is not None}
        if not ill:
            break
        k_worst, global_min, v = min(ill, key=lambda t: t[1])

        worst_local_idx = int(np.argmax(np.abs(v)))
        orbital_to_remove = remaining[worst_local_idx]

        if verbose:
            print(f"[iter {iteration}] ill_value = {global_min:.2e}, removing orbital {orbital_to_remove}")

        active = [ik for ik, _, _ in ill]
        for ik in list(Sk_work):
            if ik in active:
                Sk = np.delete(np.delete(Sk_work[ik], worst_local_idx, axis=0), worst_local_idx, axis=1)
                Sk_work[ik] = Sk
                if ik in blocks:
                    blocks[ik] = np.delete(blocks[ik], worst_local_idx, axis=0)
            else:
                del Sk_work[ik]

        remaining.remove(orbital_to_remove)

        if len(remaining) == 0:
            if verbose:
                print("[warning] All orbitals removed, stopping")
            break

        iteration += 1

    if verbose:
        print(f"[done] Kept {len(remaining)}/{norb} orbitals")

    return remaining


def eig_with_orbital_mask(
    Hk: np.ndarray, Sk: np.ndarray, kept_orbitals: List[int], fill_value: float = 1e4, return_vecs: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Eigenvalues of (Hk, Sk) restricted to `kept_orbitals`, the removed slots filled with
    `fill_value` (all sorted). The eigenvectors, if requested, are (Norb, N_kept): rows in the
    full orbital basis, zero on the removed orbitals.
    """
    norb_total = Hk.shape[0]
    n_kept = len(kept_orbitals)

    if n_kept == 0:
        eigvals = np.full(norb_total, fill_value)
        return eigvals, (np.zeros((norb_total, 0), dtype=Hk.dtype) if return_vecs else None)

    idx = np.ix_(kept_orbitals, kept_orbitals)
    Hk_sub = Hk[idx]
    Sk_sub = Sk[idx]

    try:
        if return_vecs:
            vals, vecs_sub = eigh_generalized(Hk_sub, Sk_sub)

            full_vecs = np.zeros((norb_total, n_kept), dtype=vecs_sub.dtype)
            full_vecs[kept_orbitals, :] = vecs_sub

            full_vals = np.full(norb_total, fill_value)
            full_vals[:n_kept] = vals

            return np.sort(full_vals), full_vecs[:, np.argsort(vals)]

        vals = eigh_generalized(Hk_sub, Sk_sub, eigvals_only=True)
        full_vals = np.full(norb_total, fill_value)
        full_vals[:n_kept] = vals
        return np.sort(full_vals), None

    except LinAlgError:
        eigvals = np.full(norb_total, fill_value)
        return eigvals, (np.zeros((norb_total, 0), dtype=Hk.dtype) if return_vecs else None)


class IllConditionedHandler:
    """
    Ill-conditioning treatment of the generalized eigenproblem, by window regularization or
    orbital removal (same interface as deepx-dock's `IllConditionedHandler`).

    - `method`: "window_regularization" or "orbital_removal".
    - `ill_threshold`: threshold on the quality-matrix (window) or S(k) (orbital removal)
      eigenvalues.
    - `window_emin`, `window_emax`: window relative to `fermi_energy` (eV), window
      regularization only.
    - `fill_value`: value of the removed eigenvalue slots.

    With orbital removal, call `prepare_orbital_truncation` with S(k) on the whole k-grid
    before `process_k`.
    """

    METHOD_WINDOW = "window_regularization"
    METHOD_ORBITAL = "orbital_removal"

    def __init__(
        self,
        method: str,
        ill_threshold: float = 1e-3,
        window_emin: float = -1000.0,
        window_emax: float = 6.0,
        fermi_energy: float = 0.0,
        fill_value: float = 1e4,
        verbose: bool = True,
    ):
        if method not in (self.METHOD_WINDOW, self.METHOD_ORBITAL):
            raise ValueError(f"Unknown method: {method}. Choose from '{self.METHOD_WINDOW}' or '{self.METHOD_ORBITAL}'")

        self.method = method
        self.ill_threshold = ill_threshold
        self.window_emin = window_emin + fermi_energy
        self.window_emax = window_emax + fermi_energy
        self.fill_value = fill_value
        self.verbose = verbose

        self._kept_orbitals: Optional[List[int]] = None

    @property
    def kept_orbitals(self) -> Optional[List[int]]:
        """Indices of the orbitals kept by `prepare_orbital_truncation`."""
        return self._kept_orbitals

    def prepare_orbital_truncation(
        self, Sk_list: List[np.ndarray], n_jobs: int = -1, parallel_k: bool = True
    ) -> List[int]:
        """Select the kept orbitals from S(k) at every k-point (see `global_orbital_truncation`)."""
        if self.method != self.METHOD_ORBITAL:
            raise RuntimeError(
                f"prepare_orbital_truncation() is only valid for orbital removal mode. Current method: {self.method}"
            )
        self._kept_orbitals = global_orbital_truncation(
            Sk_list, self.ill_threshold, self.verbose, n_jobs=n_jobs, parallel_k=parallel_k
        )
        return self._kept_orbitals

    def process_k(self, Hk: np.ndarray, Sk: np.ndarray, return_vecs: bool = False):
        """Eigenvalues (and, with `return_vecs`, eigenvectors) of one k-point."""
        if self.method == self.METHOD_WINDOW:
            eigvals, eigvecs = window_regularization_eig(
                Hk, Sk, self.window_emin, self.window_emax, self.ill_threshold, self.fill_value, return_vecs
            )
        else:
            if self._kept_orbitals is None:
                raise RuntimeError("Orbital removal mode requires calling prepare_orbital_truncation() first")
            eigvals, eigvecs = eig_with_orbital_mask(Hk, Sk, self._kept_orbitals, self.fill_value, return_vecs)

        if return_vecs:
            return eigvals, eigvecs
        return eigvals

    def get_info(self) -> dict:
        """Method parameters and, after orbital removal, the kept orbitals."""
        info = {
            "method": self.method,
            "ill_threshold": self.ill_threshold,
            "window_emin": self.window_emin,
            "window_emax": self.window_emax,
            "fill_value": self.fill_value,
        }
        if self._kept_orbitals is not None:
            info["n_kept_orbitals"] = len(self._kept_orbitals)
            info["kept_orbitals"] = self._kept_orbitals
        return info
