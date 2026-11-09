"""
High-symmetry k-paths for band structures, as the ``k_list_spell`` string of ``BandDataGenerator``
(one line per segment: ``n_points k1x k1y k1z k2x k2y k2z label1 label2``, fractional coordinates).

Every path is expressed in the reciprocal basis of the cell the Hamiltonian is written in. SeeK-path
is called through ``get_path_orig_cell``, which does not standardize the cell: the k-points of
``seekpath.get_path`` refer to the standardized primitive cell, generally rotated or re-chosen with
respect to the input one, and would give wrong bands if used with the input cell's Hamiltonian.

Three sources of a path:

* `seekpath_kpath` (default): the HPKOT path of the structure.
* `labels_kpath`: a user-chosen sequence of SeeK-path labels, e.g. ``"GAMMA-M-K-GAMMA|A-L"``
  (``-`` joins points of one continuous line, ``|`` starts a new one; ``G``/``Γ`` are ``GAMMA``).
* `read_kpath_file`: a ``K_PATH`` file in the spell format above, or a JSON file
  ``{"points": {"GAMMA": [0, 0, 0], "M": [0.5, 0, 0], ...}, "path": "GAMMA-M-K-GAMMA"}``
  (``path`` may also be a list of lines, e.g. ``[["GAMMA", "M", "K"], ["A", "L"]]``).
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Points per 1/Angstrom of path length, with the reciprocal lattice including the 2 pi factor, as
# in `get_band_conf_from_struc`.
DEFAULT_DENSITY = 40
_GAMMA_ALIASES = {"G", "Γ", "GAMMA", "GAM"}


def _gamma_alias(label: str) -> str:
    return "GAMMA" if label.upper() in {a.upper() for a in _GAMMA_ALIASES} else label


def structure_cell(lattice, frac_coords, elements) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """The spglib/SeeK-path cell ``(lattice, frac_coords, atomic numbers)`` of a structure given by
    its lattice (rows are the vectors), fractional coordinates and element symbols or numbers."""
    from pymatgen.core.periodic_table import Element

    numbers = [int(el) if isinstance(el, int | np.integer) else Element(str(el)).Z for el in elements]
    return np.asarray(lattice, dtype=float), np.asarray(frac_coords, dtype=float), numbers


def reciprocal_lattice(lattice) -> np.ndarray:
    """Reciprocal lattice (rows, with the 2 pi factor) of a direct lattice given by rows."""
    return 2 * np.pi * np.linalg.inv(np.asarray(lattice, dtype=float)).T


def _seekpath_orig_cell(cell, symprec: float) -> dict:
    import seekpath

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = seekpath.get_path_orig_cell(cell, with_time_reversal=True, symprec=symprec)
    for w in caught:  # e.g. SupercellWarning: the labels are those of the smaller primitive cell
        if issubclass(w.category, DeprecationWarning):  # spglib's, about its own API
            continue
        logger.warning(f"SeeK-path: {w.message}")
    return result


def seekpath_points(cell, symprec: float = 1e-5) -> dict[str, np.ndarray]:
    """SeeK-path's high-symmetry points of ``cell``, in the reciprocal basis of ``cell`` itself."""
    return {k: np.asarray(v, dtype=float) for k, v in _seekpath_orig_cell(cell, symprec)["point_coords"].items()}


def parse_label_path(spec: str | Sequence) -> list[tuple[str, str]]:
    """Segments ``[(start, end), ...]`` of a label path: a string such as ``"GAMMA-M-K|A-L"``, a list
    of lines ``[["GAMMA", "M", "K"], ["A", "L"]]``, or a list of segments ``[["GAMMA", "M"], ...]``."""
    if isinstance(spec, str):
        lines = [[p.strip() for p in line.split("-")] for line in spec.replace(",", "|").split("|") if line.strip()]
    else:
        lines = [list(line) for line in spec]
    segments = []
    for line in lines:
        line = [_gamma_alias(str(p)) for p in line if str(p)]
        if len(line) < 2:
            raise ValueError(f"A path line needs at least two points, got {line} in {spec!r}")
        segments.extend(pairwise(line))
    return segments


def segments_to_spell(
    segments: Sequence[tuple[str, str]],
    points: Mapping[str, Sequence[float]],
    lattice,
    density: float = DEFAULT_DENSITY,
) -> str:
    """The ``k_list_spell`` of ``segments`` (pairs of labels of ``points``, fractional coordinates in
    the reciprocal basis of ``lattice``), each sampled with ``density`` points per 1/Angstrom."""
    points = {_gamma_alias(k): np.asarray(v, dtype=float) for k, v in points.items()}
    rlv = reciprocal_lattice(lattice)
    lines = []
    for start, end in segments:
        missing = [p for p in (start, end) if p not in points]
        if missing:
            raise KeyError(f"Unknown k-point label(s) {missing}; available: {sorted(points)}")
        k1, k2 = points[start], points[end]
        n = max(2, round(float(np.linalg.norm((k2 - k1) @ rlv)) * density))
        coords = " ".join(f"{c:.8f}" for c in (*k1, *k2))
        lines.append(f"{n} {coords} {start} {end}")
    return "\n".join(lines)


def _in_plane_segments(segments, points, tol: float = 1e-6):
    kept = [s for s in segments if all(abs(points[p][2]) < tol for p in s)]
    if not kept:
        raise ValueError("No segment of the path lies in the k_z = 0 plane")
    return kept


def seekpath_kpath(
    cell,
    density: float = DEFAULT_DENSITY,
    in_plane: bool = False,
    symprec: float = 1e-5,
) -> str:
    """
    The SeeK-path (HPKOT) band path of ``cell`` (see `structure_cell`) as a ``k_list_spell``.

    ``in_plane=True`` keeps only the segments in the k_z = 0 plane of the cell's third reciprocal
    vector, for slabs and 2D materials with vacuum along c: e.g. GAMMA-M-K-GAMMA for a hexagonal
    monolayer, instead of the bulk path that also visits A, L and H.
    """
    result = _seekpath_orig_cell(cell, symprec)
    points = {k: np.asarray(v, dtype=float) for k, v in result["point_coords"].items()}
    segments = [tuple(seg) for seg in result["path"]]
    if in_plane:
        segments = _in_plane_segments(segments, points)
    logger.info(f"SeeK-path ({result.get('bravais_lattice_extended', '?')}): "
                + " ".join(f"{a}-{b}" for a, b in segments))
    return segments_to_spell(segments, points, cell[0], density)


def labels_kpath(
    cell,
    path: str | Sequence,
    density: float = DEFAULT_DENSITY,
    points: Mapping[str, Sequence[float]] | None = None,
    symprec: float = 1e-5,
) -> str:
    """A ``k_list_spell`` along a user-chosen label ``path`` (see `parse_label_path`). The labels are
    looked up in ``points`` (fractional, in the reciprocal basis of ``cell``), then in SeeK-path's
    points of ``cell``."""
    known = {} if points is None else {_gamma_alias(k): v for k, v in points.items()}
    segments = parse_label_path(path)
    if any(p not in known for seg in segments for p in seg):
        known = {**seekpath_points(cell, symprec), **known}
    return segments_to_spell(segments, known, cell[0], density)


def read_kpath_file(path: str | Path, cell, density: float = DEFAULT_DENSITY, symprec: float = 1e-5) -> str:
    """A ``k_list_spell`` from a file: JSON (``{"points": ..., "path": ...}``, see the module
    docstring; ``points`` may be omitted to use SeeK-path's), or a ``K_PATH`` spell file as is."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json" or text.lstrip().startswith("{"):
        spec = json.loads(text)
        if "path" not in spec:
            raise ValueError(f"{path}: a JSON k-path needs a 'path' entry")
        return labels_kpath(cell, spec["path"], density=spec.get("density", density),
                            points=spec.get("points"), symprec=symprec)
    lines = [line.split() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    for line in lines:
        if len(line) != 9:
            raise ValueError(f"{path}: K_PATH lines need 9 fields (n k1x k1y k1z k2x k2y k2z label1 label2), "
                             f"got {' '.join(line)!r}")
        int(line[0])
        [float(x) for x in line[1:7]]
    return "\n".join(" ".join(line) for line in lines)


def resolve_kpath(
    cell,
    kpath: str | Path | None = None,
    density: float = DEFAULT_DENSITY,
    in_plane: bool = False,
    symprec: float = 1e-5,
) -> str:
    """
    The ``k_list_spell`` for ``cell`` from a user specification ``kpath``:

    * ``None`` (default): the SeeK-path path (`seekpath_kpath`, with ``in_plane``);
    * the path of an existing file: `read_kpath_file`;
    * otherwise a label path such as ``"GAMMA-M-K-GAMMA"`` with SeeK-path's coordinates (`labels_kpath`).
    """
    if kpath is None:
        return seekpath_kpath(cell, density=density, in_plane=in_plane, symprec=symprec)
    looks_like_file = isinstance(kpath, Path) or "/" in str(kpath) or Path(str(kpath)).suffix.lower() == ".json"
    if looks_like_file or Path(str(kpath)).is_file():
        if not Path(kpath).is_file():
            raise FileNotFoundError(f"k-path file not found: {kpath}")
        return read_kpath_file(kpath, cell, density=density, symprec=symprec)
    return labels_kpath(cell, str(kpath), density=density, symprec=symprec)
