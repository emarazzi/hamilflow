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
diagonalization, factorizes each S(k) only once, runs the k-points in threads,
and stops tracking a k-point once its S(k) is well conditioned (see its
docstring). It runs on a GPU when one is available (see ``dense_backend``).
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading

import numpy as np
import threadpoolctl
from scipy.linalg import LinAlgError, eigh

from ..cpus import available_cpu_count
from .dense_backend import get_backend
from .eigensolver import eigh_generalized

logger = logging.getLogger(__name__)

# .failures: the list `_tracking_linalg_failures` collects this thread's per-k-point
# LinAlgError fallbacks in, or None outside such a k-point loop
_tracking = threading.local()


def _linalg_failure(what: str, consequence: str) -> None:
    """Report the LinAlgError fallback of one k-point: collected (and logged at DEBUG) inside a
    `_tracking_linalg_failures` k-point loop, which warns once for all of them; a WARNING of its
    own otherwise (e.g. `IllConditionedHandler.process_k` called directly)."""
    failures = getattr(_tracking, "failures", None)
    if failures is None:
        logger.warning(f"{what} at a k-point: {consequence}")
    else:
        failures.append((what, consequence))
        logger.debug(f"{what} at a k-point: {consequence}")


def _tracking_linalg_failures(fn, failed: dict):
    """`wrapped(i_k, *args) = fn(*args)` that records the LinAlgError fallbacks of k-point `i_k`
    in `failed[i_k]` instead of warning for each; call `_warn_linalg_failures` after the loop.
    Thread-safe: each k-point runs in one thread, and `failed` gets distinct keys."""
    def wrapped(i_k, *args):
        previous = getattr(_tracking, "failures", None)
        _tracking.failures = failures = []
        try:
            return fn(*args)
        finally:
            _tracking.failures = previous
            if failures:
                failed[i_k] = failures
    return wrapped


def _warn_linalg_failures(failed: dict, n_k: int) -> None:
    """One WARNING per kind of LinAlgError fallback recorded by `_tracking_linalg_failures`."""
    by_kind: dict[tuple[str, str], list[int]] = {}
    for i_k in sorted(failed):
        for kind in dict.fromkeys(failed[i_k]):
            by_kind.setdefault(kind, []).append(i_k)
    for (what, consequence), ks in by_kind.items():
        logger.warning(f"{what} at {len(ks)} of {n_k} k-points (indices {ks[:5]}{' ...' if len(ks) > 5 else ''}): "
                       f"{consequence} there")


def window_regularization_eig(
    Hk: np.ndarray,
    Sk: np.ndarray,
    emin: float,
    emax: float,
    ill_threshold: float,
    fill_value: float = 1e4,
    return_vecs: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
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
            _linalg_failure("Window regularization failed (LinAlgError)", "unregularized eigenvalues returned")
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
        n_jobs = available_cpu_count()
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


class _KPointState:
    """S(k) of one k-point during `global_orbital_truncation`, never copied or shrunk: the
    current S(k) is its principal submatrix on the kept orbitals. Arrays live on the device of
    `be` (see `dense_backend`).

    - `L`: lower Cholesky factor of the full S(k) (None until needed, or if S(k) is not
      positive definite, in which case the k-point uses LAPACK on the submatrix).
    - `W`: columns of S(k)^-1 at the removed orbitals, (n, n_removed).
    - `block`: subspace-iteration block on the kept orbitals, warm-starting the next iteration.
    """

    def __init__(self, S, be):
        self.be = be
        self.S = be.to_device(S)
        self.L = None
        self.W = be.xp.zeros((len(S), 0), dtype=self.S.dtype)
        self.block = None

    def submatrix(self, kept: np.ndarray):
        if len(kept) == len(self.S):
            return self.S
        k = self.be.xp.asarray(kept)
        return self.S[self.be.xp.ix_(k, k)]

    def remove(self, orbital: int, local_idx: int) -> None:
        """Account for removing `orbital` (index in the full S(k), at position `local_idx`
        among the kept orbitals)."""
        xp = self.be.xp
        if self.L is not None:
            e = xp.zeros((len(self.S), 1), dtype=self.S.dtype)
            e[orbital] = 1.0
            self.W = xp.hstack([self.W, self.be.cho_solve(self.L, e)])
        if self.block is not None:
            keep = np.ones(len(self.block), dtype=bool)
            keep[local_idx] = False
            self.block = self.block[xp.asarray(keep)]


def _lowest_eigpair_subspace(
    state: _KPointState, kept: np.ndarray, removed: np.ndarray
) -> tuple[float, np.ndarray | None]:
    """Lowest eigenpair of the kept-orbital submatrix S_KK of S(k) by block inverse subspace
    iteration: only multi-vector solves and products, which (unlike the memory-bound tridiagonal
    reduction of a dense eigh, or single-vector Lanczos) run at BLAS-3 speed.

    Neither S_KK nor a factor of it is formed. With A = S^-1 (from the Cholesky factor of the
    full S(k)) and R the removed orbitals, (S_KK)^-1 = A_KK - A_KR (A_RR)^-1 A_RK, where
    A_KR = W_K and A_RR = W_R (`_KPointState.W`); S_KK q is S applied to q padded with zeros.
    The block warm-starts from `state.block` and is stored back there.

    Returns (value, vector), or (nan, None) unless the vector is accurate enough to pick the
    same largest-amplitude orbital as an exact solver: the Davis-Kahan bound `|r| / gap` on its
    error must stay well below the gap between its two largest amplitudes. The residual r is
    computed with S_KK itself, so the accuracy of the solves only affects the convergence.
    The vector is a NumPy array.
    """
    be = state.be
    xp = be.xp
    S, L, W = state.S, state.L, state.W
    n_full, n = len(S), len(kept)
    kept_d, removed_d = xp.asarray(kept), xp.asarray(removed)
    b = min(_SUBSPACE_BLOCK, n // 4)
    x = state.block
    if x is None or x.shape != (n, b):
        rng = np.random.default_rng(0)
        x = be.to_device(rng.standard_normal((n, b)) + 1j * rng.standard_normal((n, b)))
    x = x.astype(S.dtype, copy=False)
    try:
        c_rr = be.cholesky(W[removed_d]) if len(removed) else None
    except LinAlgError:
        return np.nan, None
    w_k = W[kept_d]

    def padded(y):
        full = xp.zeros((n_full, y.shape[1]), dtype=S.dtype)
        full[kept_d] = y
        return full

    def solve(y):  # (S_KK)^-1 y
        z = be.cho_solve(L, padded(y))
        if c_rr is None:
            return z[kept_d]
        return z[kept_d] - w_k @ be.cho_solve(c_rr, z[removed_d])

    s_norm = float(xp.abs(S).sum(axis=0).max())  # 1-norm
    for _ in range(_SUBSPACE_MAX_ITER):
        q, _ = xp.linalg.qr(solve(x))
        sq = (S @ padded(q))[kept_d]
        t = q.conj().T @ sq
        w, u = xp.linalg.eigh(0.5 * (t + t.conj().T))
        x = q @ u
        r = be.to_host(xp.linalg.norm(sq @ u[:, :2] - x[:, :2] * w[:2], axis=0))
        w2 = be.to_host(w[:2])
        gap = w2[1] - w2[0] - r[1]
        x0 = be.to_host(x[:, 0])
        amp = np.sort(np.abs(x0))
        if gap > 0 and r[0] / gap < 0.1 * (amp[-1] - amp[-2]):
            state.block = x
            return float(w2[0]), x0
        if r[0] < 1e-13 * s_norm:  # converged, but too close to a tie to be sure
            break
    state.block = x
    return np.nan, None


def _lowest_eigpair(
    state: _KPointState, kept: np.ndarray, removed: np.ndarray, ill_threshold: float
) -> tuple[float, np.ndarray | None]:
    """Lowest eigenvalue and eigenvector (NumPy) of the kept-orbital S(k), or (inf, None) if all
    its eigenvalues are above `ill_threshold` (checked with a Cholesky factorization of
    S_KK - ill_threshold * 1).

    A Ritz value is an upper bound on the lowest eigenvalue, so a subspace-iteration value at or
    below the threshold already proves S(k) ill-conditioned and the Cholesky test is skipped.
    From the second visit on (S(k) was ill-conditioned at the previous iteration and likely still
    is) the subspace iteration runs first; on the first visit the Cholesky test does, and the
    factor of the full S(k) is computed if S(k) is ill-conditioned."""
    be = state.be

    def well_conditioned() -> bool:
        # S_kk - ill_threshold * 1, built in one copy (no identity matrix or second copy)
        S_kk = state.submatrix(kept)
        S_kk = S_kk.copy() if S_kk is state.S else S_kk
        diag = be.xp.arange(len(S_kk))
        S_kk[diag, diag] -= ill_threshold
        try:
            be.cholesky(S_kk)
            return True
        except LinAlgError:
            return False

    first_visit = state.block is None
    use_subspace = len(kept) >= _SUBSPACE_MIN_DIM
    if (first_visit or not use_subspace) and well_conditioned():
        return np.inf, None
    if use_subspace and first_visit and state.L is None and not len(removed):
        try:
            state.L = be.cholesky(state.S)
        except LinAlgError:
            pass
    if use_subspace and state.L is not None:
        val, vec = _lowest_eigpair_subspace(state, kept, removed)
        if vec is not None:
            if val <= ill_threshold:
                return val, vec
            # above the threshold, not yet a proof (or Cholesky failed by round-off only)
            if first_visit or well_conditioned():
                return np.inf, None
    vals, vecs = be.lowest_eigh(state.submatrix(kept))
    val = float(vals[0])
    if val > ill_threshold:  # Cholesky failed by round-off only
        return np.inf, None
    return val, be.to_host(vecs[:, 0])


def global_orbital_truncation(
    Sk_list: list[np.ndarray],
    ill_threshold: float,
    verbose: bool = True,
    n_jobs: int = -1,
    parallel_k: bool = True,
) -> list[int]:
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
    `_lowest_eigpair_subspace`; it falls back to LAPACK when its answer could differ). That
    iteration works with the full S(k) and its Cholesky factor, computed once: the removed
    orbitals enter through a low-rank correction, so S(k) is never copied or refactorized.

    On a GPU (see `dense_backend`) every S(k) is kept on the device and the k-points run one
    after the other (`n_jobs` and `parallel_k` then only set the CPU threads).

    Returns the kept orbital indices (in the original numbering).
    """
    norb = Sk_list[0].shape[0]
    kept = np.arange(norb)
    removed = np.zeros(0, dtype=int)
    be = get_backend(norb)
    parallel_k = parallel_k and be.name == "cpu"

    # k-points that can still be (or become) the worst one; the others are never touched again
    states = {ik: _KPointState(Sk, be) for ik, Sk in enumerate(Sk_list)}
    active = list(states)

    iteration = 0
    while True:
        pairs = _map_threads(lambda ik: _lowest_eigpair(states[ik], kept, removed, ill_threshold),
                             active, n_jobs, parallel_k)
        ill = [(ik, val, vec) for ik, (val, vec) in zip(active, pairs) if vec is not None]
        if not ill:
            break
        k_worst, global_min, v = min(ill, key=lambda t: t[1])

        worst_local_idx = int(np.argmax(np.abs(v)))
        orbital_to_remove = int(kept[worst_local_idx])

        if verbose:
            logger.info(f"[iter {iteration}] ill_value = {global_min:.2e}, removing orbital {orbital_to_remove}")

        active = [ik for ik, _, _ in ill]
        for ik in list(states):
            if ik not in active:
                del states[ik]
        _map_threads(lambda ik: states[ik].remove(orbital_to_remove, worst_local_idx), active, n_jobs, parallel_k)
        kept = np.delete(kept, worst_local_idx)
        removed = np.append(removed, orbital_to_remove)

        if len(kept) == 0:
            if verbose:
                logger.warning("All orbitals removed, stopping")
            break

        iteration += 1

    if verbose:
        logger.info(f"[done] Kept {len(kept)}/{norb} orbitals")

    return [int(i) for i in kept]


def eig_with_orbital_mask(
    Hk: np.ndarray, Sk: np.ndarray, kept_orbitals: list[int], fill_value: float = 1e4, return_vecs: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
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
        _linalg_failure(f"Diagonalization with {n_kept}/{norb_total} kept orbitals failed (LinAlgError)",
                        f"all eigenvalues set to fill_value={fill_value}")
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

        self._kept_orbitals: list[int] | None = None

    @property
    def kept_orbitals(self) -> list[int] | None:
        """Indices of the orbitals kept by `prepare_orbital_truncation`."""
        return self._kept_orbitals

    def prepare_orbital_truncation(
        self, Sk_list: list[np.ndarray], n_jobs: int = -1, parallel_k: bool = True
    ) -> list[int]:
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
