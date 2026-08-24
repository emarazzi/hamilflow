from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
import threadpoolctl
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from .matrix_obj import SparseAOMatrixObj


class SparseHamiltonianObj(SparseAOMatrixObj):
    """
    Memory-efficient counterpart to ``deepx_dock``'s ``HamiltonianObj``.

    Drop-in for the subset of the ``HamiltonianObj`` API hamilflow actually
    uses: metadata attributes (``elements``, ``elements_orbital_map``,
    ``lattice``, ``frac_coords``, ``occupation``, ``spinful``,
    ``info_dir_path``, ``Rijk_list``, ``atom_pairs``, ``atom_num_orbits_cumsum``),
    plus ``Sk_and_Hk(ks)`` and ``diag(ks, ...)``. Never materializes a dense
    real-space ``HR``/``SR`` array — see ``SparseAOMatrixObj`` for why.

    ``self`` acts as the Hamiltonian's own sparse block set (mirroring how
    ``HamiltonianObj`` extends ``AOMatrixObj`` by construction with
    ``matrix_type="hamiltonian"``); the overlap's sparse block set is held
    separately as ``self._s_obj``.
    """

    def __init__(self, data_path, H_file_path=None):
        super().__init__(data_path, H_file_path, matrix_type="hamiltonian")
        s_obj = SparseAOMatrixObj(data_path, matrix_type="overlap")
        self.assert_compatible(s_obj)
        self._s_obj = s_obj

    @property
    def HR(self):
        raise AttributeError(
            "SparseHamiltonianObj never materializes a dense real-space Hamiltonian "
            "(that's the point of this class). Use Sk_and_Hk(ks) or diag(ks) instead."
        )

    @property
    def SR(self):
        raise AttributeError(
            "SparseHamiltonianObj never materializes a dense real-space overlap matrix "
            "(that's the point of this class). Use Sk_and_Hk(ks) or diag(ks) instead."
        )

    def Sk_and_Hk(self, k):
        """
        Get overlap and Hamiltonian matrices at given k-point(s).

        Same contract as ``HamiltonianObj.Sk_and_Hk``: shape (3,) in / (Nb, Nb)
        out for a single k-point, (Nk, 3) in / (Nk, Nb, Nb) out for several.
        Callers should pass small batches — this still builds one dense
        (Nb, Nb) array per requested k-point, so requesting the whole k-mesh
        at once reproduces the same OOM this class exists to avoid.
        """
        k = np.asarray(k, dtype=np.float64)
        if k.ndim == 1:
            ks = k[None, :]
            squeeze = True
        else:
            ks = k
            squeeze = False

        Hk = np.stack([self._dense_k(kk) for kk in ks], axis=0)
        Sk = np.stack([self._s_obj._dense_k(kk) for kk in ks], axis=0)

        if squeeze:
            return Sk[0], Hk[0]
        return Sk, Hk

    def get_Sk(self, k):
        """
        Overlap matrix only, at given k-point(s) -- same shape contract as
        ``Sk_and_Hk`` but never touches the Hamiltonian's sparse blocks.
        For overlap-only workflows, calling ``Sk_and_Hk`` would build and
        immediately discard a dense Hk for every k-point for nothing.
        """
        k = np.asarray(k, dtype=np.float64)
        if k.ndim == 1:
            return self._s_obj._dense_k(k)
        return np.stack([self._s_obj._dense_k(kk) for kk in k], axis=0)

    def diag(
        self,
        ks,
        n_jobs: int = -1,
        parallel_k: bool = True,
        sparse_calc: bool = False,
        bands_only: bool = True,
        ill_handler=None,
        kept_orbitals: Optional[List[int]] = None,
        **kwargs,
    ):
        """
        Diagonalize the Hamiltonian at specified k-points.

        Ported from ``HamiltonianObj.diag`` with the same signature and
        semantics; the two differences are: (1) each k-point's dense Hk/Sk
        comes from the sparse blocks (``self._dense_k`` / ``self._s_obj._dense_k``)
        instead of a pre-materialized dense HR/SR, and (2) parallelism uses
        ``concurrent.futures.ThreadPoolExecutor`` directly (matching
        ``hamilflow.projection.kspace``) instead of ``deepx_dock.parallel.parallel_map``,
        to avoid depending on deepx_dock internals beyond ``AOMatrixObj``.
        ``process_k`` runs one k-point at a time, so this is safe regardless
        of how many k-points are requested.
        """
        if n_jobs < 0:
            n_jobs = os.cpu_count() or 1

        def process_k(k):
            Sk = self._s_obj._dense_k(k)
            Hk = self._dense_k(k)

            if ill_handler is not None:
                return ill_handler.process_k(Hk, Sk, return_vecs=not bands_only)

            if kept_orbitals is not None:
                from deepx_dock.compute.eigen.ill_conditioned import eig_with_orbital_mask

                return eig_with_orbital_mask(Hk, Sk, kept_orbitals, return_vecs=not bands_only)

            if sparse_calc:
                if bands_only:
                    vals = eigsh(Hk, M=Sk, return_eigenvectors=False, **kwargs)
                    return np.sort(vals)
                else:
                    vals, vecs = eigsh(Hk, M=Sk, **kwargs)
                    idx = np.argsort(vals)
                    return vals[idx], vecs[:, idx]
            else:
                if bands_only:
                    return eigh(Hk, Sk, eigvals_only=True)
                else:
                    return eigh(Hk, Sk)

        n_blas_threads = 1 if parallel_k else n_jobs
        n_workers = n_jobs if parallel_k else 1
        with threadpoolctl.threadpool_limits(limits=n_blas_threads, user_api="blas"):
            if n_workers <= 1:
                results = [process_k(k) for k in ks]
            else:
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    results = list(ex.map(process_k, ks))

        if bands_only:
            return np.stack(results, axis=1)
        else:
            eigvals = np.stack([r[0] for r in results], axis=1)
            eigvecs = np.stack([r[1] for r in results], axis=2)
            return eigvals, eigvecs
