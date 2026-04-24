from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy import ndarray
from scipy.optimize import minimize_scalar


def _validate_bands_input(bands: list | ndarray, param_name: str = "bands") -> ndarray:
    """
    Validate and convert bands input to numpy array.

    Parameters
    ----------
    bands : list | ndarray
        Band structure data
    param_name : str
        Parameter name for error messages

    Returns
    -------
    ndarray
        Validated numpy array

    Raises
    ------
    ValueError
        If input is empty, contains NaN, or has invalid shape
    """
    bands_array = np.asarray(bands, dtype=float)

    if bands_array.size == 0:
        raise ValueError(f"{param_name} cannot be empty")

    if np.any(np.isnan(bands_array)):
        raise ValueError(f"{param_name} contains NaN values")

    if bands_array.ndim < 1:
        raise ValueError(f"{param_name} must be at least 1-dimensional")

    return bands_array


def band_comparison(band1: list | ndarray, band2: list | ndarray) -> float:
    """
    Compare two band structures using root mean square error.

    Computes the RMSE between two band structures, normalized by the number of
    k-points and the minimum number of bands.

    Parameters
    ----------
    band1 : list | ndarray
        First band structure with shape (n_bands, n_kpoints)
    band2 : list | ndarray
        Second band structure with shape (m_bands, n_kpoints)

    Returns
    -------
    float
        Total RMSE across all bands

    Raises
    ------
    ValueError
        If the bands have different numbers of k-points

    Examples
    --------
    >>> band1 = np.array([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]])
    >>> band2 = np.array([[0.01, 0.11, 0.21], [1.01, 1.11, 1.21]])
    >>> total_rmse = band_comparison(band1, band2)
    """
    band1 = _validate_bands_input(band1, "band1")
    band2 = _validate_bands_input(band2, "band2")

    if band1.ndim < 2 or band2.ndim < 2:
        raise ValueError("Bands must be at least 2-dimensional (n_bands, n_kpoints)")

    if band1.shape[1] != band2.shape[1]:
        raise ValueError(
            f"Bands have different numbers of k-points: "
            f"{band1.shape[1]} vs {band2.shape[1]}"
        )

    n_bands_min = min(band1.shape[0], band2.shape[0])
    n_kpoints = band1.shape[1]

    diff_squared = np.abs(band1[:n_bands_min] - band2[:n_bands_min]) ** 2
    total_mse = np.sum(diff_squared) / (n_kpoints * n_bands_min)

    return float(np.sqrt(total_mse))


def shift_vbm(bands: list | ndarray, fermie: float) -> ndarray:
    """
    Shift band energies so the valence band maximum (VBM) is at zero.

    Finds the highest energy band below the Fermi level and shifts all bands
    so that this energy becomes zero. Returns a copy of the shifted bands.
    """
    bands_array = _validate_bands_input(bands, "bands")

    below_fermi_mask = bands_array < fermie

    if not np.any(below_fermi_mask):
        raise ValueError(
            f"No bands found below Fermi level ({fermie}). "
            f"Band range: [{np.min(bands_array):.3f}, {np.max(bands_array):.3f}]"
        )

    top_vb = np.max(bands_array[below_fermi_mask])
    return bands_array - top_vb


def shift_cbm(bands: list | ndarray, fermie: float) -> ndarray:
    """
    Shift band energies so the conduction band minimum (CBM) is at zero.

    Finds the lowest energy band above the Fermi level and shifts all bands
    so that this energy becomes zero. Returns a copy of the shifted bands.
    """
    bands_array = _validate_bands_input(bands, "bands")

    above_fermi_mask = bands_array > fermie

    if not np.any(above_fermi_mask):
        raise ValueError(
            f"No bands found above Fermi level ({fermie}). "
            f"Band range: [{np.min(bands_array):.3f}, {np.max(bands_array):.3f}]"
        )

    bottom_cb = np.min(bands_array[above_fermi_mask])
    return bands_array - bottom_cb


def shift_midgap(bands: list | ndarray, fermie: float) -> ndarray:
    """Shift band energies so the mid-gap energy is at zero."""
    bands_array = _validate_bands_input(bands, "bands")

    below_fermi_mask = bands_array < fermie
    above_fermi_mask = bands_array > fermie

    if not np.any(below_fermi_mask):
        raise ValueError(
            f"No bands found below Fermi level ({fermie}). "
            f"Band range: [{np.min(bands_array):.3f}, {np.max(bands_array):.3f}]"
        )

    if not np.any(above_fermi_mask):
        raise ValueError(
            f"No bands found above Fermi level ({fermie}). "
            f"Band range: [{np.min(bands_array):.3f}, {np.max(bands_array):.3f}]"
        )

    top_vb = np.max(bands_array[below_fermi_mask])
    bottom_cb = np.min(bands_array[above_fermi_mask])
    mid_gap = 0.5 * (top_vb + bottom_cb)

    return bands_array - mid_gap


def get_bandgap(bands: list | ndarray, fermie: float) -> float:
    """Calculate the band gap energy."""
    bands_array = _validate_bands_input(bands, "bands")

    below_fermi_mask = bands_array < fermie
    above_fermi_mask = bands_array > fermie

    if not np.any(below_fermi_mask):
        raise ValueError(
            f"No bands found below Fermi level ({fermie}). "
            f"Band range: [{np.min(bands_array):.3f}, {np.max(bands_array):.3f}]"
        )

    if not np.any(above_fermi_mask):
        raise ValueError(
            f"No bands found above Fermi level ({fermie}). "
            f"Band range: [{np.min(bands_array):.3f}, {np.max(bands_array):.3f}]"
        )

    top_vb = np.max(bands_array[below_fermi_mask])
    bottom_cb = np.min(bands_array[above_fermi_mask])

    return float(np.abs(bottom_cb - top_vb))


def get_shift(
    band_ref: list | ndarray,
    band_new: list | ndarray,
    shift_range: tuple[float, float],
) -> tuple[float, float]:
    """Find the optimal energy shift to align two band structures."""
    band_ref = _validate_bands_input(band_ref, "band_ref")
    band_new = _validate_bands_input(band_new, "band_new")

    min_shift, max_shift = shift_range

    def objective(shift: float) -> float:
        return band_comparison(band_ref, band_new + shift)

    result = minimize_scalar(objective, bounds=(min_shift, max_shift), method="bounded")

    if not (min_shift <= result.x <= max_shift):
        raise ValueError(
            f"Optimization result {result.x:.6f} is outside range {shift_range}"
        )

    return float(result.x), float(result.fun)


def correct_k_points(bands: list | ndarray, remove_indices: Sequence[int]) -> ndarray:
    """Remove k-point indices from band structure along the second axis."""
    bands_array = _validate_bands_input(bands, "bands")
    remove_set = set(int(i) for i in remove_indices)

    if bands_array.ndim < 2:
        raise ValueError("bands must be at least 2-dimensional (n_bands, n_kpoints)")

    n_kpoints = bands_array.shape[1]
    keep_indices = [i for i in range(n_kpoints) if i not in remove_set]

    return bands_array[:, keep_indices].copy()
