import math

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Kpoints

from hamilflow.dft_workflows.kpoints import detect_vacuum_axes, get_ksampling

KSPACING = {"kspacing": 0.16}


def diamond():
    return Structure(
        Lattice.cubic(3.567),
        ["C"] * 8,
        [
            [0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0],
            [0.25, 0.25, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25],
        ],
    )


def graphene_slab(c=20.0, z=0.5):
    unit = Structure(
        Lattice.hexagonal(2.46, c), ["C", "C"], [[0, 0, z], [1 / 3, 2 / 3, z]]
    )
    return unit * (2, 2, 1)


def x_slab():
    """The z-slab with its axes permuted (a, b, c) -> (c, a, b): vacuum along a."""
    slab = graphene_slab()
    m = slab.lattice.matrix
    return Structure(Lattice([m[2], m[0], m[1]]), slab.species, slab.frac_coords[:, [2, 0, 1]])


def layers(gap, thickness=3.0):
    """Two atomic planes ``thickness`` apart, then a gap of ``gap`` Å along z."""
    c = thickness + gap
    return Structure(
        Lattice.orthorhombic(2.5, 2.5, c), ["C", "C"], [[0, 0, 0], [0.5, 0.5, thickness / c]]
    )


def kgrid(structure, settings, **kwargs):
    return get_ksampling(structure, user_kpoints_settings=settings, **kwargs)["k_grid"]


def test_bulk():
    s = diamond()
    assert detect_vacuum_axes(s, 8.3) == []
    assert math.ceil(2 * math.pi / 3.567 / 0.16) == 12
    assert kgrid(s, KSPACING) == [12, 12, 12]


def test_z_slab():
    s = graphene_slab()
    assert detect_vacuum_axes(s, 8.3) == [2]
    k = kgrid(s, KSPACING)
    assert k[2] == 1 and k[0] > 1 and k[0] == k[1]


def test_x_slab():
    s = x_slab()
    assert detect_vacuum_axes(s, 8.3) == [0]
    k = kgrid(s, KSPACING)
    assert k[0] == 1 and k[1] > 1 and k[1] == k[2]
    assert k[1:] == kgrid(graphene_slab(), KSPACING)[:2]


def test_wire():
    s = Structure(
        Lattice.orthorhombic(15, 15, 2.5), ["C", "C"], [[0.5, 0.5, 0], [0.5, 0.5, 0.5]]
    )
    assert detect_vacuum_axes(s, 8.3) == [0, 1]
    k = kgrid(s, KSPACING)
    assert k[:2] == [1, 1] and k[2] == math.ceil(2 * math.pi / 2.5 / 0.16)


def test_molecule():
    s = Structure(Lattice.cubic(15), ["C", "C"], [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5 + 1.24 / 15]])
    assert detect_vacuum_axes(s, 8.3) == [0, 1, 2]
    assert kgrid(s, KSPACING) == [1, 1, 1]


@pytest.mark.parametrize(("gap", "vacuum"), [(7.5, False), (9.0, True)])
def test_near_threshold(gap, vacuum):
    s = layers(gap)
    assert (2 in detect_vacuum_axes(s, 8.3)) is vacuum
    assert (kgrid(s, KSPACING)[2] == 1) is vacuum


def test_wrap():
    c = 20.0
    s = Structure(
        Lattice.orthorhombic(2.5, 2.5, c), ["C", "C"], [[0, 0, 0.02], [0.5, 0.5, 0.98]]
    )
    # The atoms are 0.8 Å apart through the boundary; the vacuum is the 19.2 Å in between.
    assert detect_vacuum_axes(s, 8.3) == [2]
    assert detect_vacuum_axes(s, 19.0) == [2]
    assert detect_vacuum_axes(s, 19.3) == []
    # Coordinates outside [0, 1) are wrapped first.
    shifted = Structure(s.lattice, s.species, s.frac_coords + [0, 0, 3.0], to_unit_cell=False)
    assert detect_vacuum_axes(shifted, 8.3) == [2]


def test_skewed_cell_uses_interplanar_spacing():
    # c tilted strongly towards a: |c| = 12 Å but the planes are only ~6 Å apart.
    lattice = Lattice([[3, 0, 0], [0, 3, 0], [10.392, 0, 6.0]])
    s = Structure(lattice, ["C"], [[0, 0, 0]])
    assert lattice.c > 8.3
    assert detect_vacuum_axes(s, 8.3) == []


def test_reciprocal_density_backward_compat():
    s = graphene_slab()
    expected = [int(v) for v in Kpoints.automatic_density_by_vol(s, 1000, True).kpts[0]]
    assert kgrid(s, {"reciprocal_density": 1000}) == expected


def test_reciprocal_density_opt_in():
    s = graphene_slab()
    plain = kgrid(s, {"reciprocal_density": 1000})
    assert plain[2] > 1  # pymatgen puts points along the vacuum
    masked = kgrid(s, {"reciprocal_density": 1000, "vacuum_gap": 8.3})
    assert masked[:2] == plain[:2] and masked[2] == 1


def test_grid_density_opt_in():
    s = graphene_slab()
    plain = kgrid(s, {"grid_density": 1000})
    expected = [int(v) for v in Kpoints.automatic_density(s, 1000, True).kpts[0]]
    assert plain == expected
    masked = kgrid(s, {"grid_density": 1000, "vacuum_gap": 8.3})
    assert masked[:2] == plain[:2] and masked[2] == 1


def test_kspacing_vacuum_gap_none_disables_detection():
    s = graphene_slab()
    k = kgrid(s, {"kspacing": 0.16, "vacuum_gap": None})
    assert k[2] == math.ceil(2 * math.pi / 20.0 / 0.16)


def test_overrides_combine_with_detection():
    assert kgrid(diamond(), KSPACING, force_2d=True) == [12, 12, 1]
    assert kgrid(diamond(), KSPACING, force_1d=True) == [1, 1, 12]
    assert kgrid(graphene_slab(), KSPACING, force_1d=True) == [1, 1, 1]
    with pytest.raises(ValueError):
        get_ksampling(diamond(), user_kpoints_settings=KSPACING, force_2d=True, force_1d=True)


def test_kspacing_must_be_positive():
    with pytest.raises(ValueError):
        get_ksampling(diamond(), user_kpoints_settings={"kspacing": -0.1})


@pytest.mark.parametrize(
    "settings",
    [
        {"k_grid": np.array([2, 2, 1])},
        {"grid_density": 500, "vacuum_gap": 8.3},
        {"reciprocal_density": 100},
        {"reciprocal_density": 100, "vacuum_gap": 8.3},
        KSPACING,
    ],
)
def test_types_are_python_int(settings):
    k = kgrid(graphene_slab(), settings)
    assert len(k) == 3 and all(type(v) is int for v in k)
