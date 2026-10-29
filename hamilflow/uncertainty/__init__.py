"""Uncertainty utilities for hamilflow.

Provides:
- ``read_deeph_hamiltonian``, ``write_deeph_hamiltonian``, ``average_predicted_hamiltonians``
- ``discover_structures``, ``link_ensemble_files``
- ``BandUncertaintyCalculator``
- ``BandUncertaintyPostProcessor``, ``UncertaintyCalibrator``
- ``MatrixUncertaintyCalculator``
- ``fit_rigid_shifts``, ``write_aligned_hamiltonian`` (rigid-shift alignment, see ``alignment``)
"""

from .alignment import fit_rigid_shifts, write_aligned_hamiltonian
from .band_postprocess import BandUncertaintyPostProcessor, UncertaintyCalibrator
from .band_uncertainty import BandUncertaintyCalculator
from .ensemble_io import discover_structures, link_ensemble_files
from .hamiltonian_io import (
    average_predicted_hamiltonians,
    read_deeph_hamiltonian,
    write_deeph_hamiltonian,
)
from .matrix_uncertainty import MatrixUncertaintyCalculator

__all__ = [
    "read_deeph_hamiltonian",
    "write_deeph_hamiltonian",
    "average_predicted_hamiltonians",
    "discover_structures",
    "link_ensemble_files",
    "BandUncertaintyCalculator",
    "BandUncertaintyPostProcessor",
    "UncertaintyCalibrator",
    "MatrixUncertaintyCalculator",
    "fit_rigid_shifts",
    "write_aligned_hamiltonian",
]
