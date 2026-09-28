from __future__ import annotations

import numpy as np
from deepx_dock.compute.eigen.band import BandDataGenerator

from hamilflow.sparse_hamiltonian import SparseHamiltonianObj


class SparseBandDataGenerator(BandDataGenerator):
    """``BandDataGenerator`` specialized for hamilflow's ``SparseHamiltonianObj``.

    k-path handling, Fermi-level shifting and HDF5 dump are reused unchanged
    from the base class. ``calc_band_data`` is overridden so that it calls
    ``SparseHamiltonianObj.diag`` with hamilflow's own signature
    (``n_jobs``/``parallel_k``) regardless of which ``deepx_dock`` version is
    installed (older releases pass ``k_process_num`` instead), and so that
    ``sparse_calc=True`` goes through the fully sparse path: each k-point's
    Hk/Sk is built as a scipy sparse matrix and only ``num_band`` eigenvalues
    near ``fermi_energy + lowest_band_energy`` are computed with shift-invert
    ``eigsh``, without ever allocating a dense (Nb, Nb) matrix.

    The constructor fails fast on a non-sparse Hamiltonian instead of
    silently accepting an object that would OOM by materializing a dense
    ``HR``/``SR``.
    """

    def __init__(self, obj_H: SparseHamiltonianObj, band_conf):
        if not isinstance(obj_H, SparseHamiltonianObj):
            raise TypeError(
                "SparseBandDataGenerator requires a SparseHamiltonianObj "
                f"(e.g. from hamilflow.band_structures.get_hamiltonian), got {type(obj_H).__name__}."
            )
        super().__init__(obj_H, band_conf)

    def calc_band_data(
        self,
        n_jobs: int = -1,
        parallel_k: bool = True,
        sparse_calc: bool = False,
        ill_method=None,
        ill_threshold=None,
        window_emin=None,
        window_emax=None,
    ):
        """
        Diagonalize along the high-symmetry path and store ``self.band_data``
        (shape ``(band_quantity, kpoints_quantity)``, shifted so E_F = 0).

        Parameters
        ----------
        n_jobs : int
            Total worker/BLAS thread budget (-1: all cores).
        parallel_k : bool
            Parallelize over k-points (True) or within each diagonalization (False).
        sparse_calc : bool
            If True, compute only ``band_conf["num_band"]`` bands closest above
            ``fermi_energy + band_conf["lowest_band_energy"]`` with sparse
            shift-invert ``eigsh``; Hk/Sk are never densified. Requires
            ``ill_method=None`` (ill-conditioning handling needs dense matrices).
        ill_method : str or None
            None, 'window_regularization' or 'orbital_removal' (dense path only).
        ill_threshold, window_emin, window_emax : float or None
            Forwarded to ``IllConditionedHandler``; see the base class.
        """
        if ill_method == "none":
            ill_method = None
        if sparse_calc and ill_method is not None:
            raise ValueError(
                "sparse_calc=True cannot be combined with ill_method: ill-conditioning "
                "handling needs dense Hk/Sk. Use sparse_calc=False, or ill_method=None."
            )

        ill_handler = None
        if ill_method is not None:
            from deepx_dock.compute.eigen.ill_conditioned import IllConditionedHandler

            ill_handler = IllConditionedHandler(
                method=ill_method,
                ill_threshold=ill_threshold if ill_threshold is not None else 1e-3,
                window_emin=window_emin if window_emin is not None else -1000.0,
                window_emax=window_emax if window_emax is not None else 6.0,
                fermi_energy=self.fermi_energy,
            )
            if ill_method == "orbital_removal":
                ks = np.array(self.kpoints_frac_list)
                Sk_list = self.obj_H.get_all_Sk(ks, n_jobs=n_jobs, parallel_k=parallel_k)
                ill_handler.prepare_orbital_truncation(Sk_list)

        if sparse_calc:
            num_band = self.band_conf.get("num_band", 50)
            lowest_band_energy = self.band_conf.get("lowest_band_energy", -0.5)
            maxiter = self.band_conf.get("maxiter", 300)
            mat_dim = self.obj_H._mat_dim
            if num_band >= mat_dim - 1:
                raise ValueError(
                    f"num_band={num_band} must be < {mat_dim - 1} (matrix dimension - 1) for sparse "
                    "diagonalization; use sparse_calc=False to get all bands."
                )
            self.band_quantity = num_band
            print(
                f"Sparse calculation with num_band={num_band}, "
                f"lowest_band_energy={lowest_band_energy}, maxiter={maxiter} ..."
            )
            # Single-vector Lanczos can miss one member of an exactly degenerate
            # pair (e.g. Kramers pairs at TRIM points with SOC), returning the
            # next level instead. Requesting a couple of extra eigenvalues with
            # a larger Krylov space and trimming back to num_band avoids it.
            n_eig = min(num_band + 2, mat_dim - 2)
            kwargs = {
                "k": n_eig,
                "ncv": min(mat_dim, max(4 * n_eig, 20)),
                "sigma": self.fermi_energy + lowest_band_energy,
                "which": "LA",
                "maxiter": maxiter,
                "tol": 1e-5,
                "mode": "normal",
            }
        else:
            kwargs = {}

        self.band_data = self.obj_H.diag(
            self.kpoints_frac_list,
            n_jobs=n_jobs,
            parallel_k=parallel_k,
            sparse_calc=sparse_calc,
            bands_only=True,
            ill_handler=ill_handler,
            **kwargs,
        )
        if sparse_calc:
            self.band_data = self.band_data[: self.band_quantity]

        if self.band_data.shape[0] != self.band_quantity:
            print(f"Warn: Only {self.band_data.shape[0]} bands are calculated, not {self.band_quantity} in input.")
            self.band_quantity = self.band_data.shape[0]
        self._shift_band_data_to_fermi_zero()
