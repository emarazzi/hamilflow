"""Two-centre (Slater-Koster) overlap tables from FHI-aims dimer runs.

Bond frame. For two atoms with the bond vector along +z, the block between a shell ``a``
(angular momentum ``la``) on the first atom and a shell ``b`` (``lb``) on the second, in the
real harmonics of `rotations`, is diagonal in m with equal entries for ``+m`` and ``-m``
(invariance under rotations about z and under the mirror y -> -y). So the full block is set by
the radial functions

    t_{a b |m|}(r),   |m| = 0 .. min(la, lb)     (sigma, pi, delta, phi, gamma)

stored here on a grid of distances and interpolated with cubic splines. The block for any bond
vector is the rotation of the bond-frame block (`rotations`).

FHI-aims stores a basis-function pair only when the distance is below the sum of the two
functions' outer radii (``pbc_lists.f90``), so every channel is exactly zero from
``outer_radius[a] + outer_radius[b]`` on, and two atoms form a pair of the DeepH files when
their distance is below ``2 max(outer_radius)``. These radii are the last radial grid point
where the function exceeds FHI-aims' ``wave_threshold`` (``shrink_fixed_basis.f90``), NOT the
"outer radius [A]" printed in ``aims.out``; they are calibrated by bisection on dimer runs
(tight C: pair cutoff 11.34466 Å from dimers, inside the bracket (11.34422, 11.34524) Å set by
the pair sets of three real structures; the provenance is kept in ``TwoCentreTable.meta``).

On-site block: ``<phi_A mu | phi_A nu>`` is the same for every atom of an element in exact
arithmetic. FHI-aims' grid integration is not: its overlap carries a non-two-centre,
environment-dependent error of ~1e-4 off-site (bond lengths < 7 Å) and up to ~1e-3 on-site
(diffuse shells), which no two-centre table reproduces. The on-site block should be the mean
over the atoms of condensed structures (`onsite_from_structures`), not the dimer value.

Tables come from dimers (two atoms along z in a large periodic box, Gamma only), run with the
production overlap-only FHI-aims settings and converted with the production conversion, so the
basis, the conventions and the integration settings are those of the reference data.
`bond_frame_samples` reads the same quantities from the pairs of any converted structure, which
is used to check a table against real structures.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

from .io import deeph_paths, read_blocks, read_info, read_layout, read_poscar
from .rotations import real_wigner, rotation_from_z

# FHI-aims settings of the production overlap-only runs (examples/dft_workflows/
# aims_overlap_only_to_projected_deeph.py plus the flow's output_rs_matrices_stop). The dimer
# runs must use the same ones.
OVL_ONLY_AIMS_KWARGS: dict[str, Any] = {
    "species_dir": "tight",
    "density_update_method": "density_matrix",
    "output_rs_matrices": "plain",
    "occupation_type": "gaussian 0.01",
    "use_local_index": "false",
    "KS_method": "parallel",
    "collect_eigenvectors": "false",
    "sc_accuracy_etot": "1.0e-03 eV",
    "sc_accuracy_rho": 1.0e-2,
    "output_rs_matrices_stop": "",
}

DIMER_BOX = 30.0  # Å: periodic images of the partner atom are >= 30 - 11.35 Å away, beyond the pair cutoff (11.345 Å, tight C)


def shell_offsets(shells: list[int] | tuple[int, ...]) -> np.ndarray:
    return np.concatenate([[0], np.cumsum([2 * l + 1 for l in shells])]).astype(np.int64)


def channel_list(shells_a: tuple[int, ...], shells_b: tuple[int, ...]) -> np.ndarray:
    """Channels ``(a, b, |m|)``, (n_channels, 3)."""
    return np.array([(a, b, m) for a, la in enumerate(shells_a) for b, lb in enumerate(shells_b)
                     for m in range(min(la, lb) + 1)], dtype=np.int64)


def wigner_blockdiag(shells: tuple[int, ...], Q: Any, xp: Any = np) -> Any:
    """Block-diagonal rotation of all shells of an atom, (..., n_orb, n_orb)."""
    off = shell_offsets(shells)
    M = xp.zeros(Q.shape[:-2] + (int(off[-1]), int(off[-1])), dtype=xp.float64)
    cache: dict[int, Any] = {}
    for s, l in enumerate(shells):
        if l not in cache:
            cache[l] = real_wigner(l, Q, xp=xp)
        M[..., off[s]:off[s + 1], off[s]:off[s + 1]] = cache[l]
    return M


def channels_to_bond_blocks(values: Any, shells_a: tuple[int, ...], shells_b: tuple[int, ...],
                            xp: Any = np) -> Any:
    """Bond-frame blocks (n, n_a, n_b) from channel values (n, n_channels)."""
    oa, ob = shell_offsets(shells_a), shell_offsets(shells_b)
    chans = channel_list(shells_a, shells_b)
    rows, cols, src = [], [], []
    for c, (a, b, m) in enumerate(chans):
        la, lb = shells_a[a], shells_b[b]
        for mm in {m, -m}:
            rows.append(oa[a] + la + mm)
            cols.append(ob[b] + lb + mm)
            src.append(c)
    out = xp.zeros((values.shape[0], int(oa[-1]), int(ob[-1])), dtype=xp.float64)
    out[:, xp.asarray(rows), xp.asarray(cols)] = values[:, xp.asarray(src)]
    return out


def bond_blocks_to_channels(blocks: np.ndarray, shells_a: tuple[int, ...],
                            shells_b: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Channel values (n, n_channels), averaged over +-m, from bond-frame blocks, and the largest
    deviation of each block from the bond-frame form (off-diagonal entries and +m/-m spread)."""
    oa, ob = shell_offsets(shells_a), shell_offsets(shells_b)
    chans = channel_list(shells_a, shells_b)
    vals = np.zeros((blocks.shape[0], len(chans)))
    model = np.zeros_like(blocks)
    for c, (a, b, m) in enumerate(chans):
        la, lb = shells_a[a], shells_b[b]
        r_p, c_p = oa[a] + la + m, ob[b] + lb + m
        r_m, c_m = oa[a] + la - m, ob[b] + lb - m
        vals[:, c] = 0.5 * (blocks[:, r_p, c_p] + blocks[:, r_m, c_m])
        model[:, r_p, c_p] = vals[:, c]
        model[:, r_m, c_m] = vals[:, c]
    resid = np.abs(blocks - model).reshape(len(blocks), -1).max(axis=1) if len(blocks) else np.zeros(0)
    return vals, resid


@dataclass
class TwoCentreTable:
    """Splined two-centre overlap channels of one element pair, plus the on-site blocks.

    ``values[i, c]`` is channel ``channels[c] = (a, b, |m|)`` at distance ``r[i]`` (Å).
    ``outer_radius_a/b`` (Å, per shell) set the cutoff ``outer_radius_a[a] + outer_radius_b[b]``
    of every channel, beyond which it is exactly zero (the radius where the function vanishes on
    FHI-aims' grid). ``atom_pair_cutoff``: see `pair_cutoff`. ``onsite`` maps an element to its
    on-site block.
    """

    element_a: str
    element_b: str
    shells_a: tuple[int, ...]
    shells_b: tuple[int, ...]
    r: np.ndarray
    values: np.ndarray
    outer_radius_a: np.ndarray
    outer_radius_b: np.ndarray
    onsite: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    atom_pair_cutoff: float | None = None

    def __post_init__(self) -> None:
        self.shells_a = tuple(int(v) for v in self.shells_a)
        self.shells_b = tuple(int(v) for v in self.shells_b)
        self.r = np.asarray(self.r, dtype=np.float64)
        self.values = np.asarray(self.values, dtype=np.float64)
        self.outer_radius_a = np.asarray(self.outer_radius_a, dtype=np.float64)
        self.outer_radius_b = np.asarray(self.outer_radius_b, dtype=np.float64)
        self.channels = channel_list(self.shells_a, self.shells_b)
        if self.values.shape != (len(self.r), len(self.channels)):
            raise ValueError(f"values must be (n_r, n_channels) = ({len(self.r)}, {len(self.channels)}), "
                             f"got {self.values.shape}")
        if np.any(np.diff(self.r) <= 0):
            raise ValueError("r grid must be strictly increasing")
        self.channel_cutoff = (self.outer_radius_a[self.channels[:, 0]]
                               + self.outer_radius_b[self.channels[:, 1]])
        if self.r[-1] < self.channel_cutoff.max() - 0.05:
            warnings.warn(f"table grid ends at {self.r[-1]:.3f} Å, before the largest channel cutoff "
                          f"{self.channel_cutoff.max():.3f} Å: channels are set to 0 beyond the grid", stacklevel=2)
        self._spline = CubicSpline(self.r, self.values, axis=0, bc_type="not-a-knot", extrapolate=False)

    @property
    def pair_cutoff(self) -> float:
        """Atoms closer than this form a pair of the DeepH files: ``atom_pair_cutoff`` when set
        (FHI-aims' sparsity radius, larger than the support radii: blocks just inside it are
        stored but exactly zero), else ``max(outer_radius_a) + max(outer_radius_b)``."""
        if self.atom_pair_cutoff is not None:
            return float(self.atom_pair_cutoff)
        return float(self.outer_radius_a.max() + self.outer_radius_b.max())

    def evaluate(self, dist: np.ndarray) -> np.ndarray:
        """Channel values (n, n_channels) at distances ``dist`` (Å). Zero at and beyond each
        channel's cutoff; an error below the first grid point."""
        dist = np.asarray(dist, dtype=np.float64)
        if dist.size and dist.min() < self.r[0] - 1e-9:
            raise ValueError(f"distance {dist.min():.4f} Å below the table range (r_min = {self.r[0]:.4f} Å)")
        out = self._spline(np.minimum(dist, self.r[-1]))
        out[dist > self.r[-1]] = 0.0
        out[dist[:, None] >= self.channel_cutoff[None, :]] = 0.0
        return out

    def bond_blocks(self, dist: np.ndarray, xp: Any = np) -> Any:
        return channels_to_bond_blocks(xp.asarray(self.evaluate(dist)), self.shells_a, self.shells_b, xp=xp)

    def blocks(self, vecs: np.ndarray, xp: Any = np) -> Any:
        """Blocks (n, n_a, n_b) for bond vectors ``vecs`` (n, 3) in Å (from atom a to atom b)."""
        vecs = np.asarray(vecs, dtype=np.float64)
        dist = np.linalg.norm(vecs, axis=1)
        if np.any(dist < 1e-8):
            raise ValueError("blocks: zero bond vector; on-site blocks come from `onsite`")
        B0 = self.bond_blocks(dist, xp=xp)
        Q = rotation_from_z(xp.asarray(vecs), xp=xp)
        Ma = wigner_blockdiag(self.shells_a, Q, xp=xp)
        Mb = Ma if self.shells_b == self.shells_a else wigner_blockdiag(self.shells_b, Q, xp=xp)
        return Ma @ B0 @ xp.swapaxes(Mb, -1, -2)

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        np.savez(
            path,
            element_a=self.element_a, element_b=self.element_b,
            shells_a=np.array(self.shells_a), shells_b=np.array(self.shells_b),
            r=self.r, values=self.values,
            outer_radius_a=self.outer_radius_a, outer_radius_b=self.outer_radius_b,
            onsite_elements=np.array(list(self.onsite.keys())),
            **{f"onsite_{el}": blk for el, blk in self.onsite.items()},
            meta=np.array(repr(self.meta)),
            atom_pair_cutoff=np.array(np.nan if self.atom_pair_cutoff is None else self.atom_pair_cutoff),
        )
        return path

    @classmethod
    def load(cls, path: Path | str) -> TwoCentreTable:
        with np.load(path, allow_pickle=False) as z:
            onsite = {str(el): np.array(z[f"onsite_{el}"]) for el in z["onsite_elements"]}
            return cls(
                element_a=str(z["element_a"]), element_b=str(z["element_b"]),
                shells_a=tuple(z["shells_a"]), shells_b=tuple(z["shells_b"]),
                r=z["r"], values=z["values"],
                outer_radius_a=z["outer_radius_a"], outer_radius_b=z["outer_radius_b"],
                onsite=onsite, meta={"saved_meta": str(z["meta"])},
                atom_pair_cutoff=None if np.isnan(z["atom_pair_cutoff"]) else float(z["atom_pair_cutoff"]),
            )


# --- dimer inputs ---------------------------------------------------------------------------------


def dimer_name(element: str, r: float) -> str:
    return f"{element}2_d{r:.6f}"


def dimer_structure(element: str, r: float, box: float = DIMER_BOX):
    """Two atoms of ``element`` at the origin and at ``(0, 0, r)`` in a cubic box (pymatgen)."""
    from pymatgen.core import Lattice, Structure

    if not 0 < r < box / 2:
        raise ValueError(f"dimer distance {r} must be in (0, box / 2 = {box / 2})")
    return Structure(Lattice.cubic(box), [element, element], [[0, 0, 0], [0, 0, r]],
                     coords_are_cartesian=True)


def write_dimer_inputs(element: str, distances: np.ndarray, root: Path | str,
                       aims_kwargs: dict[str, Any] | None = None, box: float = DIMER_BOX) -> list[Path]:
    """Write ``control.in`` / ``geometry.in`` of one Gamma-only overlap-only run per distance,
    with pymatgen's ``StaticSetGenerator`` (as the production flow does), in
    ``root/<dimer_name>``. ``aims_kwargs`` default to `OVL_ONLY_AIMS_KWARGS`."""
    from pymatgen.io.aims.sets.core import StaticSetGenerator

    params = dict(OVL_ONLY_AIMS_KWARGS if aims_kwargs is None else aims_kwargs)
    params["k_grid"] = [1, 1, 1]
    gen = StaticSetGenerator(user_params=params)
    root = Path(root)
    dirs = []
    for r in np.asarray(distances, dtype=np.float64):
        d = root / dimer_name(element, float(r))
        gen.get_input_set(dimer_structure(element, float(r), box)).write_input(d, overwrite=True)
        dirs.append(d)
    return dirs


# --- reading converted data -----------------------------------------------------------------------

def pair_vectors(structure, atom_pairs: np.ndarray) -> np.ndarray:
    """Bond vectors ``tau_j + R . lattice - tau_i`` (n, 3) of ``atom_pairs`` rows."""
    return (structure.positions[atom_pairs[:, 4]] + atom_pairs[:, :3] @ structure.lattice
            - structure.positions[atom_pairs[:, 3]])


def bond_frame_samples(deeph_dir: Path | str, element: str = "C", max_pairs: int | None = None,
                       min_dist: float = 1e-6, seed: int = 0) -> dict[str, np.ndarray]:
    """Channels of the off-site pairs of one converted structure, rotated to the bond frame.

    Returns ``dist`` (n,), ``values`` (n, n_channels), ``resid`` (n,) (deviation from the
    bond-frame form, see `bond_blocks_to_channels`), ``pair_index`` (n,) and the on-site blocks
    ``onsite`` (n_atoms, n_orb, n_orb). ``max_pairs`` draws a random subset of the off-site pairs;
    without it every off-site block is read into memory (n_pairs x n_orb^2 x 8 bytes: ~0.1 GB for
    an 8-atom cell, GBs for 200+ atoms).
    """
    paths = deeph_paths(deeph_dir)
    st = read_poscar(paths["poscar"])
    shells = tuple(int(v) for v in read_info(deeph_dir)["elements_orbital_map"][element])
    ap, _, _ = read_layout(paths["overlap"])
    vecs = pair_vectors(st, ap)
    dist = np.linalg.norm(vecs, axis=1)
    is_el = np.array([st.species[i] == element for i in range(len(st.species))])
    both = is_el[ap[:, 3]] & is_el[ap[:, 4]]
    off = np.flatnonzero(both & (dist > min_dist))
    if max_pairs is not None and len(off) > max_pairs:
        off = np.sort(np.random.default_rng(seed).choice(off, max_pairs, replace=False))
    on = np.flatnonzero(both & (dist <= min_dist))
    blocks = np.array(read_blocks(paths["overlap"], off)) if len(off) else np.zeros((0, 1, 1))
    if len(off):
        M = wigner_blockdiag(shells, rotation_from_z(vecs[off]))
        B0 = np.swapaxes(M, -1, -2) @ blocks @ M
        values, resid = bond_blocks_to_channels(B0, shells, shells)
    else:
        values, resid = np.zeros((0, len(channel_list(shells, shells)))), np.zeros(0)
    return {
        "dist": dist[off],
        "values": values,
        "resid": resid,
        "pair_index": off,
        "onsite": np.array(read_blocks(paths["overlap"], on)),
        "shells": np.array(shells),
    }


def build_table_from_dimers(deeph_dirs: list[Path | str], outer_radius: np.ndarray,
                            atom_pair_cutoff: float, element: str = "C") -> TwoCentreTable:
    """Table from converted dimer runs (one directory per distance; the pair ``(0, 0, 0, 0, 1)``
    is the bond along +z, ``(0, 0, 0, 1, 0)`` the same bond seen from the other end, both are
    averaged).

    ``outer_radius`` (Å, per shell): where each function vanishes, so that channel ``(a, b)`` is
    exactly zero from ``outer_radius[a] + outer_radius[b]`` on; ``atom_pair_cutoff``: FHI-aims'
    pair-storage distance. Both are calibrated by bisection on dimer runs (neither is the "outer
    radius" printed in ``aims.out``). The on-site block is the dimer
    average; replace it with `onsite_from_structures` (the dimer on-site depends on the
    distance by up to ~4e-3 through FHI-aims' grid integration).

    Averaging the two orientations makes ``t_ba = (-1)^(la + lb) t_ab`` hold exactly, so the
    assembled S satisfies ``S_ij(R) = S_ji(-R)^T`` to rounding.
    """
    dists, vals, resid, onsite = [], [], [], []
    shells = None
    for d in deeph_dirs:
        s = bond_frame_samples(d, element=element)
        if len(s["dist"]) != 2 or not np.allclose(s["dist"], s["dist"][0]):
            raise ValueError(f"{d}: expected the two orientations of one bond in a dimer run, "
                             f"found pairs at {s['dist']} Å (box too small?)")
        if shells is None:
            shells = tuple(int(v) for v in s["shells"])
        dists.append(s["dist"][0])
        vals.append(s["values"].mean(axis=0))
        resid.append(max(s["resid"].max(), np.abs(s["values"][0] - s["values"][1]).max()))
        onsite.extend(list(s["onsite"]))
    assert shells is not None
    radii = np.asarray(outer_radius, dtype=np.float64)
    if len(radii) != len(shells):
        raise ValueError(f"{len(radii)} outer radii, but {len(shells)} shells")
    order = np.argsort(dists)
    onsite_arr = np.array(onsite)
    return TwoCentreTable(
        element_a=element, element_b=element, shells_a=shells, shells_b=shells,
        r=np.array(dists)[order], values=np.array(vals)[order],
        outer_radius_a=radii, outer_radius_b=radii,
        onsite={element: symmetrize(onsite_arr.mean(axis=0))},
        meta={
            "n_dimers": len(dists),
            "bond_frame_resid_max": float(np.max(resid)),
            "onsite_source": "dimers",
            "onsite_spread_max": float(np.abs(onsite_arr - onsite_arr.mean(axis=0)).max()),
        },
        atom_pair_cutoff=float(atom_pair_cutoff),
    )


def symmetrize(block: np.ndarray) -> np.ndarray:
    return 0.5 * (block + block.T)


def onsite_from_structures(deeph_dirs: list[Path | str], element: str = "C") -> tuple[np.ndarray, float]:
    """Mean on-site block of ``element`` over all atoms of converted (condensed) structures, and
    the largest deviation of a single atom's block from it (FHI-aims integration noise)."""
    blocks = []
    for d in deeph_dirs:
        paths = deeph_paths(d)
        st = read_poscar(paths["poscar"])
        ap, _, _ = read_layout(paths["overlap"])
        on = np.flatnonzero((ap[:, 3] == ap[:, 4]) & ~ap[:, :3].any(axis=1)
                            & np.array([st.species[i] == element for i in ap[:, 3]]))
        blocks.extend(read_blocks(paths["overlap"], on))
    arr = np.array(blocks)
    mean = symmetrize(arr.mean(axis=0))
    return mean, float(np.abs(arr - mean).max())


def with_onsite(table: TwoCentreTable, element: str, block: np.ndarray, source: str) -> TwoCentreTable:
    """Copy of ``table`` with the on-site block of ``element`` replaced."""
    return TwoCentreTable(
        table.element_a, table.element_b, table.shells_a, table.shells_b, table.r, table.values,
        table.outer_radius_a, table.outer_radius_b,
        onsite={**table.onsite, element: symmetrize(np.asarray(block))},
        meta={**table.meta, "onsite_source": source},
        atom_pair_cutoff=table.atom_pair_cutoff,
    )
