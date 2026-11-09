"""High-symmetry paths in the reciprocal basis of the input cell, from SeeK-path, labels or files."""

import json

import numpy as np
import pytest

from hamilflow.band_structures.kpath import (
    labels_kpath,
    parse_label_path,
    read_kpath_file,
    reciprocal_lattice,
    resolve_kpath,
    seekpath_kpath,
    structure_cell,
)

A = 3.16
HEX = np.array([[A, 0, 0], [-A / 2, A * np.sqrt(3) / 2, 0], [0, 0, 20.0]])
MOS2 = structure_cell(HEX, [[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.42], [2 / 3, 1 / 3, 0.58]], ["Mo", "S", "S"])


def spell_segments(spell):
    """[(n, k1, k2, label1, label2), ...] of a k_list_spell."""
    out = []
    for line in spell.splitlines():
        f = line.split()
        out.append((int(f[0]), np.array(f[1:4], dtype=float), np.array(f[4:7], dtype=float), f[7], f[8]))
    return out


def test_seekpath_in_plane_hexagonal_monolayer():
    segs = spell_segments(seekpath_kpath(MOS2, in_plane=True))
    assert [(s[3], s[4]) for s in segs] == [("GAMMA", "M"), ("M", "K"), ("K", "GAMMA")]
    rlv = reciprocal_lattice(HEX)
    k_point = segs[1][2]
    np.testing.assert_allclose(np.linalg.norm(k_point @ rlv), 4 * np.pi / (3 * A), rtol=1e-6)  # coordinates written with 8 decimals
    assert all(abs(s[1][2]) < 1e-8 and abs(s[2][2]) < 1e-8 for s in segs)


def test_seekpath_full_path_leaves_the_plane():
    labels = {lab for s in spell_segments(seekpath_kpath(MOS2)) for lab in s[3:]}
    assert {"GAMMA", "M", "K", "A", "L", "H"} <= labels


def test_seekpath_uses_the_input_cell_not_a_standardized_one():
    """Same crystal written with permuted, rotated axes: K must still sit at |k| = 4 pi / 3a in the
    Cartesian frame of *this* cell, and in its k_z = 0 plane (here the plane of a1, a2 is (a2, a3))."""
    perm = [2, 0, 1]
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    lattice = HEX[perm] @ rot.T
    cell = structure_cell(lattice, MOS2[1][:, perm], [42, 16, 16])
    rlv = reciprocal_lattice(lattice)
    segs = spell_segments(seekpath_kpath(cell))
    k_points = [s[2] for s in segs if s[4] == "K"] + [s[1] for s in segs if s[3] == "K"]
    assert k_points
    for k in k_points:
        np.testing.assert_allclose(np.linalg.norm(k @ rlv), 4 * np.pi / (3 * A), rtol=1e-6)  # coordinates written with 8 decimals
        assert abs(k[0]) < 1e-8  # no component along b1, the vacuum direction of this cell


def test_density_sets_the_number_of_points():
    spell = seekpath_kpath(MOS2, in_plane=True, density=100)
    rlv = reciprocal_lattice(HEX)
    for n, k1, k2, *_ in spell_segments(spell):
        assert n == max(2, round(np.linalg.norm((k2 - k1) @ rlv) * 100))


def test_labels_with_gamma_aliases_and_breaks():
    assert parse_label_path("G-M-K-Γ|A-L") == [("GAMMA", "M"), ("M", "K"), ("K", "GAMMA"), ("A", "L")]
    segs = spell_segments(labels_kpath(MOS2, "G-K|M-G"))
    assert [(s[3], s[4]) for s in segs] == [("GAMMA", "K"), ("M", "GAMMA")]


def test_unknown_label_raises():
    with pytest.raises(KeyError, match="X"):
        labels_kpath(MOS2, "GAMMA-X")


def test_json_file_with_own_points(tmp_path):
    f = tmp_path / "path.json"
    f.write_text(json.dumps({"points": {"G": [0, 0, 0], "Y": [0, 0.5, 0]}, "path": [["G", "Y", "M"]]}))
    segs = spell_segments(read_kpath_file(f, MOS2))
    assert [(s[3], s[4]) for s in segs] == [("GAMMA", "Y"), ("Y", "M")]
    np.testing.assert_allclose(segs[0][2], [0, 0.5, 0])
    np.testing.assert_allclose(segs[1][2], [0.5, 0, 0])  # M from SeeK-path


def test_kpath_file_is_used_as_is(tmp_path):
    f = tmp_path / "K_PATH"
    f.write_text("# comment\n10 0 0 0 0.5 0 0 GAMMA M\n\n7 0.5 0 0 0.333 0.333 0 M K\n")
    assert resolve_kpath(MOS2, str(f)).splitlines() == ["10 0 0 0 0.5 0 0 GAMMA M", "7 0.5 0 0 0.333 0.333 0 M K"]


def test_kpath_file_with_wrong_fields_raises(tmp_path):
    f = tmp_path / "K_PATH"
    f.write_text("10 0 0 0 0.5 0 0 GAMMA\n")
    with pytest.raises(ValueError, match="9 fields"):
        read_kpath_file(f, MOS2)


def test_resolve_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_kpath(MOS2, str(tmp_path / "nope.json"))
