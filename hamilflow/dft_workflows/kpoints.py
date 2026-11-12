from __future__ import annotations

import copy
import math
from typing import Any, cast

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.symmetry.bandstructure import HighSymmKpath

DEFAULT_VACUUM_GAP = 8.3
"""Default vacuum threshold in Å for :func:`detect_vacuum_axes`.

The tight C basis has a confinement potential with onset at 4 Å, so 2 × 4 Å = 8 Å is roughly
where the overlap across a gap becomes negligible. FHI-aims's own k-grid checks, observed
empirically: gaps of 6.6–7.97 Å are treated as periodic (k = 1 along them aborts with the
density-ratio error), while a gap of 8.69 Å is treated as vacuum (k > 1 along it aborts).
8.3 Å lies inside that interval. The value is empirical, not taken from the FHI-aims source,
so it is exposed as the ``vacuum_gap`` setting rather than hard-coded.
"""


def detect_vacuum_axes(structure: Structure, vacuum_gap: float = DEFAULT_VACUUM_GAP) -> list[int]:
    """Return the indices of the lattice axes along which the structure has a vacuum gap.

    For each axis ``i`` the fractional coordinates along ``i`` are wrapped into [0, 1) and
    sorted; the largest consecutive difference, including the periodic wrap gap
    ``s[0] + 1 - s[-1]``, is converted to Å with the interplanar spacing
    ``d_i = 2π / |b_i|`` (the distance between the lattice planes spanned by the other two
    axes, equal to ``|a_i|`` in orthogonal cells and correct in skewed ones). The axis is a
    vacuum axis if that gap is larger than ``vacuum_gap`` (Å).

    Examples: ``[2]`` for a z-slab, ``[0, 1]`` for a wire along z, ``[0, 1, 2]`` for a molecule
    in a box. See :data:`DEFAULT_VACUUM_GAP` for the rationale of the default threshold.
    """
    frac = np.mod(np.asarray(structure.frac_coords, dtype=float), 1.0)
    spacings = 2.0 * np.pi / np.asarray(structure.lattice.reciprocal_lattice.abc, dtype=float)
    vacuum_axes: list[int] = []
    for axis in range(3):
        s = np.sort(frac[:, axis])
        gaps = np.append(np.diff(s), s[0] + 1.0 - s[-1])
        if float(gaps.max()) * float(spacings[axis]) > vacuum_gap:
            vacuum_axes.append(axis)
    return vacuum_axes


def _mask_vacuum_axes(
    structure: Structure, k_grid: Any, vacuum_gap: float | None
) -> list[int]:
    """Return ``k_grid`` as three ints with every vacuum axis set to 1 (no-op if gap is None)."""
    k_grid_values = [int(value) for value in k_grid]
    if vacuum_gap is not None:
        for axis in detect_vacuum_axes(structure, float(vacuum_gap)):
            k_grid_values[axis] = 1
    return k_grid_values


def get_ksampling(
    structure: Structure,
    kpoints_updates: dict[str, Any] | None = None,
    user_kpoints_settings: dict[str, Any] | None = None,
    force_gamma: bool = True,
    force_2d: bool = False,
    force_1d: bool = False,
    symprec: float = 0.01,
) -> dict[str, Any] | None:
    """Resolve k-point settings into a single payload for AIMS input generation.

    The settings dict is resolved by the first matching key, in this order:

    - ``k_grid`` / ``kgrid``: explicit Γ-centred grid.
    - ``grid_density``: pymatgen ``Kpoints.automatic_density`` (k-points per reciprocal atom).
    - ``reciprocal_density``: pymatgen ``Kpoints.automatic_density_by_vol`` (k-points per Å⁻³).
    - ``kspacing``: uniform linear spacing in Å⁻¹, **including the 2π factor**;
      ``k_i = max(1, ceil(|b_i| / kspacing))`` with ``|b_i| = 2π / d_i``. Optional
      ``vacuum_gap`` (Å, default :data:`DEFAULT_VACUUM_GAP`; ``None`` disables detection)
      sets ``k_i = 1`` along every axis found by :func:`detect_vacuum_axes`.
    - ``line_density``: band-structure path (no vacuum detection).
    - ``explicit`` / ``added_kpoints``: returned unchanged (no vacuum detection).

    ``grid_density`` and ``reciprocal_density`` run vacuum detection only if the settings
    contain ``vacuum_gap`` (opt-in, so results without it are unchanged). Known limitation:
    pymatgen distributes points using the full cell volume, vacuum included; masking the vacuum
    axes afterwards removes the wasted points, but the in-plane density stays lower than the bulk
    density for the same ``grid_density`` / ``reciprocal_density``. Use ``kspacing`` when the
    density must be consistent across dimensionalities.

    ``force_2d`` (k_z = 1) and ``force_1d`` (k_x = k_y = 1) are manual overrides applied after
    vacuum masking; they only ever set components to 1, so they combine with detection.
    """

    if force_2d and force_1d:
        raise ValueError("force_2d and force_1d are mutually exclusive.")

    def normalize_k_grid(k_grid: tuple[int, int, int] | list[int] | tuple[Any, ...]) -> list[int]:
        k_grid_values = [int(value) for value in k_grid]
        if len(k_grid_values) != 3:
            raise ValueError(f"k_grid must contain exactly three integers, got: {k_grid_values}")
        if force_2d:
            k_grid_values[2] = 1
        elif force_1d:
            k_grid_values[0] = 1
            k_grid_values[1] = 1
        return k_grid_values

    if user_kpoints_settings not in (None, {}):
        kconfig: dict[str, Any] = copy.deepcopy(user_kpoints_settings)
    elif kpoints_updates:
        kconfig = copy.deepcopy(kpoints_updates)
    else:
        return None

    if not isinstance(kconfig, dict):
        if hasattr(kconfig, "as_dict"):
            kconfig = cast(dict[str, Any], kconfig.as_dict())
        else:
            raise TypeError(
                "k-point settings must be provided as a dict or an object exposing as_dict()."
            )

    if "k_grid" in kconfig or "kgrid" in kconfig:
        k_grid = kconfig.get("k_grid", kconfig.get("kgrid"))
        if k_grid is None:
            raise ValueError("k_grid resolution failed.")
        return {"k_grid": normalize_k_grid(k_grid)}

    if kconfig.get("grid_density"):
        kpoints = Kpoints.automatic_density(structure, int(kconfig["grid_density"]), force_gamma)
        k_grid = _mask_vacuum_axes(structure, kpoints.kpts[0], kconfig.get("vacuum_gap"))
        return {"k_grid": normalize_k_grid(k_grid)}

    if kconfig.get("reciprocal_density"):
        kpoints = Kpoints.automatic_density_by_vol(
            structure, kconfig["reciprocal_density"], force_gamma
        )
        k_grid = _mask_vacuum_axes(structure, kpoints.kpts[0], kconfig.get("vacuum_gap"))
        return {"k_grid": normalize_k_grid(k_grid)}

    if kconfig.get("kspacing"):
        kspacing = float(kconfig["kspacing"])
        if kspacing <= 0:
            raise ValueError(f"kspacing must be positive, got: {kspacing}")
        # pymatgen's reciprocal_lattice includes the 2π factor: |b_i| = 2π / d_i.
        k_grid = [
            max(1, math.ceil(b_length / kspacing))
            for b_length in structure.lattice.reciprocal_lattice.abc
        ]
        k_grid = _mask_vacuum_axes(
            structure, k_grid, kconfig.get("vacuum_gap", DEFAULT_VACUUM_GAP)
        )
        return {"k_grid": normalize_k_grid(k_grid)}

    if kconfig.get("line_density"):
        kpath_kwargs = dict(kconfig.get("kpath_kwargs", {}))
        kpath_kwargs.setdefault("symprec", symprec)
        kpath = HighSymmKpath(structure, **kpath_kwargs)
        frac_k_points, k_points_labels = kpath.get_kpoints(
            line_density=kconfig["line_density"], coords_are_cartesian=False
        )
        return {
            "kpoints_mode": "line",
            "line_density": kconfig["line_density"],
            "kpath_kwargs": kpath_kwargs,
            "kpts": frac_k_points,
            "kpts_weights": [1] * len(frac_k_points),
            "kpts_labels": k_points_labels,
            "comment": "Non SCF run along symmetry lines",
        }

    if kconfig.get("explicit") or kconfig.get("added_kpoints"):
        return kconfig

    raise ValueError(
        "Unsupported k-point sampling settings for AIMS. Provide k_grid, grid_density, "
        "reciprocal_density, kspacing, line_density, or explicit k-point settings."
    )
