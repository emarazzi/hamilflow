from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, eigh, solve_triangular


def eigh_generalized(H: np.ndarray, S: np.ndarray, eigvals_only: bool = False):
    """
    All eigenvalues (and S-orthonormal eigenvectors, as columns) of the Hermitian generalized
    problem H psi = E S psi, with S positive definite: the drop-in for
    ``scipy.linalg.eigh(H, S)`` used by every dense diagonalization in hamilflow.

    The problem is reduced to the standard one L^-1 H L^-dag (S = L L^dag) and solved with
    LAPACK's MRRR driver (``evr``). scipy's default for the generalized problem
    (``gvd``, divide and conquer) is several times slower on these Hamiltonians and does not
    speed up with more BLAS threads; e.g. for n = 1856, 12.5 s vs 6.8 s on one thread and
    13.6 s vs 3.4 s on four, eigenvalues equal to 1e-12 eV. Raises ``LinAlgError`` if S is
    not positive definite, as ``scipy.linalg.eigh`` does.
    """
    L = cholesky(S, lower=True, check_finite=False)
    # L^-1 H, then L^-1 (L^-1 H)^dag = L^-1 H L^-dag (H Hermitian)
    A = solve_triangular(L, H, lower=True, check_finite=False)
    A = solve_triangular(L, A.conj().T, lower=True, check_finite=False)
    A = 0.5 * (A + A.conj().T)
    if eigvals_only:
        return eigh(A, eigvals_only=True, driver="evr", check_finite=False)
    w, v = eigh(A, driver="evr", check_finite=False)
    return w, solve_triangular(L, v, lower=True, trans="C", check_finite=False)
