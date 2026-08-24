from __future__ import annotations

from deepx_dock.compute.eigen.band import BandDataGenerator

from hamilflow.sparse_hamiltonian import SparseHamiltonianObj


class SparseBandDataGenerator(BandDataGenerator):
    """``BandDataGenerator`` specialized for hamilflow's ``SparseHamiltonianObj``.

    ``SparseHamiltonianObj`` exposes the same ``reciprocal_lattice`` /
    ``spinful`` / ``orbits_quantity`` / ``fermi_energy`` attributes and the
    same ``diag(ks, n_jobs=, parallel_k=, ...)`` / ``get_all_Sk(ks, n_jobs=,
    parallel_k=)`` signatures as ``HamiltonianObj``, so every method of the
    base class -- k-path handling, diagonalization, Fermi-level shifting,
    HDF5 dump -- is reused unchanged. This subclass only narrows the
    accepted Hamiltonian type and fails fast on a mismatch, instead of
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
