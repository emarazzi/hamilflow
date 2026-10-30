"""
Rigid-shift alignment. H - c S has the eigenvalues of H shifted by exactly -c, the property everything rests on.

`write_aligned_hamiltonian` is checked first against the fixture's dense matrices, and is then used to build
Hamiltonians with a known shift for testing the fit.
"""

import itertools

import h5py
import numpy as np
import pytest
import scipy.linalg
from deepx_dock.CONSTANT import DEEPX_HAMILTONIAN_FILENAME, DEEPX_OVERLAP_FILENAME

from hamilflow.sparse_hamiltonian import SparseHamiltonianObj
from hamilflow.uncertainty.alignment import (
    SparseStream,
    fit_rigid_shifts,
    overlap_like,
    pair_batches,
    pair_distances,
    read_layout,
    read_poscar,
    write_aligned_hamiltonian,
)

K = (0.13, 0.37, 0.0)
# The residual subtracts nearly equal sums of squares, so a perfect shift gives about sqrt(eps), not 0
RESIDUAL_FLOOR = 1e-7


def paths(structure):
    return structure.path / DEEPX_HAMILTONIAN_FILENAME, structure.path / DEEPX_OVERLAP_FILENAME


def write_sparse(path, entries, shapes=((1, 2),), atom_pairs=None):
    shapes = np.array(shapes, dtype=np.int64)
    if atom_pairs is None:
        atom_pairs = [(0, 0, 0, i, i) for i in range(len(shapes))]
    with h5py.File(path, "w") as f:
        f["atom_pairs"] = np.array(atom_pairs, dtype=np.int64)
        f["chunk_boundaries"] = np.cumsum([0, *(shapes[:, 0] * shapes[:, 1])], dtype=np.int64)
        f["chunk_shapes"] = shapes
        f["entries"] = np.asarray(entries)
    return path


# ---------------------------------- writing ----------------------------------


def test_aligned_file_holds_h_minus_c_s(synthetic_structure, tmp_path):
    h_path, s_path = paths(synthetic_structure)
    out = write_aligned_hamiltonian(h_path, s_path, 1.5, tmp_path / "a" / "aligned.h5")
    Sk, Hk = SparseHamiltonianObj(synthetic_structure.path, H_file_path=out).Sk_and_Hk(np.array(K))
    S_expected, H_expected = synthetic_structure.Sk_and_Hk(K)
    np.testing.assert_allclose(Sk, S_expected, atol=1e-11)
    np.testing.assert_allclose(Hk, H_expected - 1.5 * S_expected, atol=1e-11)


def test_alignment_shifts_every_eigenvalue_by_exactly_c(synthetic_structure, tmp_path):
    h_path, s_path = paths(synthetic_structure)
    out = write_aligned_hamiltonian(h_path, s_path, 1.5, tmp_path / "aligned.h5")
    Sk, Hk = SparseHamiltonianObj(synthetic_structure.path, H_file_path=out).Sk_and_Hk(np.array(K))
    S0, H0 = synthetic_structure.Sk_and_Hk(K)
    tol = 1e-12 * max(1.0, np.linalg.norm(H0, 2))
    np.testing.assert_allclose(
        scipy.linalg.eigh(Hk, Sk, eigvals_only=True), scipy.linalg.eigh(H0, S0, eigvals_only=True) - 1.5, atol=tol
    )


def test_aligned_file_keeps_the_layout_and_records_its_origin(synthetic_structure, tmp_path):
    h_path, s_path = paths(synthetic_structure)
    out = write_aligned_hamiltonian(h_path, s_path, 1.5, tmp_path / "aligned.h5", aligned_to="dft")
    for original, aligned in zip(read_layout(h_path), read_layout(out), strict=True):
        np.testing.assert_array_equal(original, aligned)
    with h5py.File(out, "r") as f, h5py.File(h_path, "r") as src:
        assert f.attrs["rigid_shift_eV"] == 1.5
        assert f.attrs["aligned_to"] == "dft"
        assert f.attrs["source"] == str(h_path.resolve())
        assert f["entries"].dtype == src["entries"].dtype


@pytest.mark.parametrize("batch_entries", [1, 50, 10**7])
def test_aligned_file_does_not_depend_on_the_batch_size(synthetic_structure, tmp_path, batch_entries):
    h_path, s_path = paths(synthetic_structure)
    reference = write_aligned_hamiltonian(h_path, s_path, 1.5, tmp_path / "reference.h5")
    out = write_aligned_hamiltonian(h_path, s_path, 1.5, tmp_path / "aligned.h5", batch_entries=batch_entries)
    with h5py.File(reference, "r") as a, h5py.File(out, "r") as b:
        np.testing.assert_array_equal(a["entries"][:], b["entries"][:])


# ---------------------------------- fit ----------------------------------


def test_fit_recovers_a_known_shift_with_no_residual(synthetic_structure, tmp_path):
    h_path, s_path = paths(synthetic_structure)
    shifted = write_aligned_hamiltonian(h_path, s_path, -2.5, tmp_path / "shifted.h5")  # H + 2.5 S
    fit = fit_rigid_shifts([shifted], h_path, s_path)[0]
    assert fit["c_eV"] == pytest.approx(2.5)
    assert fit["residual"] < RESIDUAL_FLOOR
    assert fit["rms_after_eV"] < RESIDUAL_FLOOR
    assert fit["rms_before_eV"] > 0


def test_subtracting_the_fitted_shift_recovers_the_reference(synthetic_structure, tmp_path):
    h_path, s_path = paths(synthetic_structure)
    shifted = write_aligned_hamiltonian(h_path, s_path, -2.5, tmp_path / "shifted.h5")
    c = fit_rigid_shifts([shifted], h_path, s_path)[0]["c_eV"]
    back = write_aligned_hamiltonian(shifted, s_path, c, tmp_path / "back.h5")
    with h5py.File(back, "r") as a, h5py.File(h_path, "r") as b:
        np.testing.assert_allclose(a["entries"][:], b["entries"][:], atol=1e-9)


def test_fit_without_reference_uses_the_mean(synthetic_structure, tmp_path):
    h_path, s_path = paths(synthetic_structure)
    members = [write_aligned_hamiltonian(h_path, s_path, -c, tmp_path / f"{c}.h5") for c in (1.0, 2.0, 6.0)]
    fits = fit_rigid_shifts(members, None, s_path)
    np.testing.assert_allclose([f["c_eV"] for f in fits], [-2.0, -1.0, 3.0])  # relative to their mean, 3
    assert all(f["residual"] < RESIDUAL_FLOOR for f in fits)


def test_identical_hamiltonians_have_no_shift_and_no_residual(synthetic_structure):
    h_path, s_path = paths(synthetic_structure)
    fit = fit_rigid_shifts([h_path], h_path, s_path)[0]
    assert fit == {"c_eV": 0.0, "residual": 0.0, "rms_before_eV": 0.0, "rms_after_eV": 0.0}


def test_fit_splits_the_difference_into_shift_and_residual_by_hand(tmp_path):
    # H - H_ref = (2, 3), S = (1, 0): c = 2 leaves (0, 3)
    h = write_sparse(tmp_path / "h.h5", [2.0, 3.0])
    ref = write_sparse(tmp_path / "ref.h5", [0.0, 0.0])
    s = write_sparse(tmp_path / "s.h5", [1.0, 0.0])
    fit = fit_rigid_shifts([h], ref, s)[0]
    assert fit["c_eV"] == pytest.approx(2.0)
    assert fit["residual"] == pytest.approx(3 / np.sqrt(13))
    assert fit["rms_before_eV"] == pytest.approx(np.sqrt(13 / 2))
    assert fit["rms_after_eV"] == pytest.approx(np.sqrt(9 / 2))


def test_difference_orthogonal_to_the_overlap_is_all_residual(tmp_path):
    h = write_sparse(tmp_path / "h.h5", [0.0, 3.0])
    ref = write_sparse(tmp_path / "ref.h5", [0.0, 0.0])
    fit = fit_rigid_shifts([h], ref, write_sparse(tmp_path / "s.h5", [1.0, 0.0]))[0]
    assert fit["c_eV"] == 0.0
    assert fit["residual"] == pytest.approx(1.0)


def test_only_the_real_part_of_a_complex_difference_can_be_a_shift(tmp_path):
    h = write_sparse(tmp_path / "h.h5", np.array([2.0 + 5.0j, 0.0]))
    ref = write_sparse(tmp_path / "ref.h5", np.array([0.0j, 0.0j]))
    fit = fit_rigid_shifts([h], ref, write_sparse(tmp_path / "s.h5", [1.0, 0.0]))[0]
    assert fit["c_eV"] == pytest.approx(2.0)
    assert fit["residual"] == pytest.approx(5 / np.sqrt(29))


def test_zero_overlap_gives_no_shift(tmp_path):
    h = write_sparse(tmp_path / "h.h5", [1.0, 1.0])
    ref = write_sparse(tmp_path / "ref.h5", [0.0, 0.0])
    fit = fit_rigid_shifts([h], ref, write_sparse(tmp_path / "s.h5", [0.0, 0.0]))[0]
    assert fit["c_eV"] == 0.0
    assert fit["residual"] == pytest.approx(1.0)


@pytest.mark.parametrize(("batch_entries", "n_threads"), [(1, 1), (50, 2), (10**7, 1)])
def test_fit_does_not_depend_on_batches_or_threads(synthetic_structure, tmp_path, batch_entries, n_threads):
    h_path, s_path = paths(synthetic_structure)
    shifted = write_aligned_hamiltonian(h_path, s_path, -2.5, tmp_path / "shifted.h5")
    reference = fit_rigid_shifts([shifted], h_path, s_path)[0]
    fit = fit_rigid_shifts([shifted], h_path, s_path, batch_entries=batch_entries, n_threads=n_threads)[0]
    assert fit == pytest.approx(reference, abs=RESIDUAL_FLOOR)


# ---------------------------------- streaming ----------------------------------


def test_hamiltonians_with_another_layout_are_rejected(tmp_path):
    a = write_sparse(tmp_path / "a.h5", [1.0, 2.0])
    b = write_sparse(tmp_path / "b.h5", [1.0, 2.0], atom_pairs=[(1, 0, 0, 0, 0)])
    with pytest.raises(ValueError, match="sparse layout"):
        SparseStream([a, b], a)


def test_overlap_with_other_atom_pairs_is_rejected(tmp_path):
    a = write_sparse(tmp_path / "a.h5", [1.0, 2.0])
    b = write_sparse(tmp_path / "b.h5", [1.0, 2.0], atom_pairs=[(1, 0, 0, 0, 0)])
    with pytest.raises(ValueError, match="atom_pairs"):
        SparseStream([a], b)


def test_stream_yields_all_entries_in_order(synthetic_structure):
    h_path, s_path = paths(synthetic_structure)
    stream = SparseStream([h_path], s_path, batch_entries=50)
    batches = list(stream)
    assert len(batches) > 1
    with h5py.File(h_path, "r") as f:
        np.testing.assert_array_equal(np.concatenate([hs[0] for _, _, hs, _ in batches]), f["entries"][:])
    assert sum(len(s) for _, _, _, s in batches) == stream.n_entries


@pytest.mark.parametrize("batch_entries", [1, 3, 5, 100])
def test_pair_batches_cover_every_pair_once_and_in_order(batch_entries):
    bounds = np.array([0, 4, 4, 9, 10, 16])
    ranges = list(pair_batches(bounds, batch_entries))
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 5
    assert all(a[1] == b[0] for a, b in itertools.pairwise(ranges))
    assert all(p1 > p0 for p0, p1 in ranges)


def test_pair_batches_group_pairs_up_to_the_batch_size():
    assert list(pair_batches(np.array([0, 2, 4, 6, 8]), 4)) == [(0, 2), (2, 4)]


def test_pair_batches_of_nothing_is_empty():
    assert list(pair_batches(np.array([0]), 10)) == []


def test_overlap_like_keeps_matching_blocks():
    s = np.array([1.0, 2.0])
    shapes = np.array([[1, 2]])
    assert overlap_like(s, shapes, shapes) is s


def test_overlap_like_expands_onto_the_diagonal_spin_blocks():
    s = np.array([1.0, 2.0])
    expanded = overlap_like(s, np.array([[1, 2]]), np.array([[2, 4]]))
    np.testing.assert_array_equal(expanded, [1, 2, 0, 0, 0, 0, 1, 2])


def test_overlap_like_rejects_unrelated_shapes():
    with pytest.raises(ValueError, match="neither"):
        overlap_like(np.array([1.0, 2.0]), np.array([[1, 2]]), np.array([[3, 3]]))


# ---------------------------------- POSCAR ----------------------------------


POSCAR = """comment
{scale}
  2.0 0.0 0.0
  0.0 3.0 0.0
  0.0 0.0 4.0
Mo S
1 2
{extra}{mode}
  {a}
  {b}
  {c}
"""


@pytest.mark.parametrize(
    ("scale", "extra", "mode", "coords"),
    [
        ("1.0", "", "Direct", ("0.5 0.0 0.0", "0.0 0.5 0.0", "0.0 0.0 0.5")),
        ("1.0", "", "Cartesian", ("1.0 0.0 0.0", "0.0 1.5 0.0", "0.0 0.0 2.0")),
        ("1.0", "Selective dynamics\n", "Direct", ("0.5 0.0 0.0 T T F", "0.0 0.5 0.0 T T F", "0.0 0.0 0.5 T T F")),
    ],
    ids=["direct", "cartesian", "selective-dynamics"],
)
def test_read_poscar_gives_cartesian_positions(tmp_path, scale, extra, mode, coords):
    a, b, c = coords
    (tmp_path / "POSCAR").write_text(POSCAR.format(scale=scale, extra=extra, mode=mode, a=a, b=b, c=c))
    lattice, species, pos = read_poscar(tmp_path)
    np.testing.assert_allclose(lattice, np.diag([2.0, 3.0, 4.0]))
    assert species == ["Mo", "S", "S"]
    np.testing.assert_allclose(pos, [[1.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 2.0]])


@pytest.mark.parametrize(("mode", "coords"), [("Direct", "0.5 0.0 0.0"), ("Cartesian", "0.5 0.0 0.0")])
def test_read_poscar_applies_the_scale(tmp_path, mode, coords):
    (tmp_path / "POSCAR").write_text(POSCAR.format(scale="2.0", extra="", mode=mode, a=coords, b=coords, c=coords))
    lattice, _, pos = read_poscar(tmp_path)
    np.testing.assert_allclose(lattice, np.diag([4.0, 6.0, 8.0]))
    np.testing.assert_allclose(pos[0], [2.0, 0.0, 0.0] if mode == "Direct" else [1.0, 0.0, 0.0])


def test_pair_distance_includes_the_lattice_translation(tmp_path):
    text = POSCAR.format(scale="1.0", extra="", mode="Cartesian", a="0 0 0", b="1 0 0", c="0 3 0")
    (tmp_path / "POSCAR").write_text(text)
    pairs = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [1, 0, 0, 0, 1], [-1, 0, 0, 0, 1], [0, 0, 1, 1, 2]])
    np.testing.assert_allclose(
        pair_distances(tmp_path, pairs),
        [0.0, 1.0, 3.0, 1.0, np.sqrt(1 + 9 + 16)],
    )
