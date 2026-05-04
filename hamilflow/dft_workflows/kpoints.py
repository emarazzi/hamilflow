from __future__ import annotations

import copy
from typing import Any

from typing import cast

from pymatgen.core import SETTINGS, Structure
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.symmetry.bandstructure import HighSymmKpath


def get_ksampling(
    structure: Structure,
    kpoints_updates: dict[str, Any] | None = None,
    user_kpoints_settings: dict[str, Any] | None = None,
    force_gamma: bool = True,
    symprec: float = SETTINGS.SYMPREC,
) -> dict[str, Any] | None:
    """Resolve k-point settings into a single payload for AIMS input generation."""
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
        k_grid_values = tuple(int(value) for value in k_grid)
        if len(k_grid_values) != 3:
            raise ValueError(f"k_grid must contain exactly three integers, got: {k_grid_values}")
        return {"k_grid": [int(k_grid_values[0]), int(k_grid_values[1]), int(k_grid_values[2])]}

    if kconfig.get("grid_density"):
        kpoints = Kpoints.automatic_density(structure, int(kconfig["grid_density"]), force_gamma)
        k_grid = kpoints.kpts[0]
        return {"k_grid": [int(k_grid[0]), int(k_grid[1]), int(k_grid[2])]}

    if kconfig.get("reciprocal_density"):
        kpoints = Kpoints.automatic_density_by_vol(
            structure, kconfig["reciprocal_density"], force_gamma
        )
        k_grid = kpoints.kpts[0]
        return {"k_grid": [int(k_grid[0]), int(k_grid[1]), int(k_grid[2])]}

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
        "reciprocal_density, line_density, or explicit k-point settings."
    )