"""Reduced info.json, the sparse output layout and the h5 writers."""

import json

import h5py
import numpy as np
import pytest

from hamilflow.projection.io import (
    build_reduced_elements_orbital_map,
    build_reduced_output_layout,
    create_zero_dataset,
    dump_reduced_entries_h5,
    dump_reduced_matrix_h5,
    hermitize_real_space_blocks,
    reduce_occupation_value,
    write_reduced_info_json,
    zero_h5_entries,
)

ELEMENTS = ["Mo", "S", "S"]
ORBITAL_MAP = {"Mo": [0, 1], "S": [0, 1]}  # Mo 0-3, S 4-7 and 8-11


# ---------------------------------- orbital map ----------------------------------


def test_removed_shells_disappear_from_the_map():
    assert build_reduced_elements_orbital_map(ELEMENTS, ORBITAL_MAP, [0, 4, 8]) == {"Mo": [1], "S": [1]}
    assert build_reduced_elements_orbital_map(ELEMENTS, ORBITAL_MAP, [1, 2, 3]) == {"Mo": [0], "S": [0, 1]}


def test_nothing_removed_keeps_the_map():
    assert build_reduced_elements_orbital_map(ELEMENTS, ORBITAL_MAP, []) == ORBITAL_MAP


def test_map_keeps_the_order_of_repeated_shells():
    assert build_reduced_elements_orbital_map(["X"], {"X": [0, 1, 0, 2]}, [0]) == {"X": [1, 0, 2]}


def test_part_of_a_shell_cannot_be_removed():
    with pytest.raises(ValueError, match="Partial removal"):
        build_reduced_elements_orbital_map(ELEMENTS, ORBITAL_MAP, [1])


def test_atoms_of_one_element_must_keep_the_same_shells():
    with pytest.raises(ValueError, match="Inconsistent reduced shells"):
        build_reduced_elements_orbital_map(ELEMENTS, ORBITAL_MAP, [4])


# ---------------------------------- occupation ----------------------------------


@pytest.mark.parametrize(
    ("occupation", "removed", "mode", "expected"),
    [
        (1036, list(range(392)), "schur", 252),  # a real structure: 14 Mo and 28 S, without their core
        (10, [0, 1], "schur", 6),
        (10, [0, 0, 1], "schur", 6),  # an index counts once
        (10.5, [0], "schur", 8.5),
        (10, [0, 1], "truncate", 10),
        (None, [0], "schur", None),
    ],
)
def test_occupation_loses_two_electrons_per_orbital_only_with_schur(occupation, removed, mode, expected):
    assert reduce_occupation_value(occupation, removed, mode, 100) == expected


def test_whole_reduced_occupation_is_an_int():
    assert isinstance(reduce_occupation_value(10.0, [0], "schur", 100), int)


def test_occupation_of_another_form_is_rejected():
    with pytest.raises(ValueError, match="Unsupported occupation"):
        reduce_occupation_value([1, 2], [0], "schur", 100)


def test_unknown_reduction_mode_is_rejected():
    with pytest.raises(ValueError, match="Unsupported reduction_mode"):
        reduce_occupation_value(10, [0], "other", 100)


def test_info_json_is_rewritten_and_other_keys_are_kept(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "out").mkdir()
    info = {
        "atoms_quantity": 3,
        "orbits_quantity": 12,
        "orthogonal_basis": False,
        "spinful": False,
        "fermi_energy_eV": -5.3,
        "elements_orbital_map": ORBITAL_MAP,
        "occupation": 20,
    }
    (tmp_path / "in" / "info.json").write_text(json.dumps(info))
    path = write_reduced_info_json(tmp_path / "in", tmp_path / "out", ELEMENTS, [0, 4, 8], "schur")
    written = json.loads(path.read_text())
    assert written == {
        **info,
        "orbits_quantity": 9,
        "elements_orbital_map": {"Mo": [1], "S": [1]},
        "occupation": 14,
    }
    assert json.loads((tmp_path / "in" / "info.json").read_text()) == info


# ---------------------------------- layout ----------------------------------


CUMSUM = np.array([0, 2, 3])  # atom 0: orbitals 0, 1; atom 1: orbital 2
RS = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
PAIRS = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1],
        [1, 0, 0, 0, 1],
        [-1, 0, 0, 1, 0],
    ]
)


def random_mats(n_r, nb, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_r, nb, nb)) + 1j * rng.standard_normal((n_r, nb, nb))


def targets_of(layout, mats):
    """The accumulators the streaming projection would fill from dense real-space matrices."""
    out = np.empty(len(layout.target_flat), dtype=mats.dtype)
    for r in range(len(mats)):
        s, e = layout.r_bounds[r], layout.r_bounds[r + 1]
        out[s:e] = mats[r].reshape(-1)[layout.target_flat[s:e]]
    return out


def entries_of(path):
    with h5py.File(path, "r") as f:
        return f["entries"][:], f["atom_pairs"][:], f["chunk_shapes"][:], f["chunk_boundaries"][:]


@pytest.mark.parametrize("keep", [[0, 1, 2], [0, 2], [1, 2], [0, 1]])
def test_layout_entries_match_the_dense_writer(tmp_path, keep):
    """What the streaming path writes must be what a dense hermitize + dump would write."""
    mats = random_mats(3, len(keep))
    layout = build_reduced_output_layout(RS, PAIRS, CUMSUM, keep)
    entries = layout.hermitize_real(targets_of(layout, mats).real)
    streamed = dump_reduced_entries_h5(tmp_path / "streamed.h5", layout, entries)
    dense = dump_reduced_matrix_h5(
        tmp_path / "dense.h5", hermitize_real_space_blocks(mats, RS), RS, PAIRS, CUMSUM, keep
    )
    for a, b in zip(entries_of(streamed), entries_of(dense), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-14)


def test_layout_imaginary_part_matches_the_dense_hermitization():
    mats = random_mats(3, 3)
    layout = build_reduced_output_layout(RS, PAIRS, CUMSUM, [0, 1, 2])
    imag = layout.hermitize_imag(targets_of(layout, mats).imag)
    hermitized = hermitize_real_space_blocks(mats, RS)
    # the block (R = 1, i = 0, j = 1) holds the orbital pairs (0, 2) and (1, 2)
    start = layout.chunk_boundaries[4]
    np.testing.assert_allclose(imag[start : start + 2], hermitized[1][[0, 1], 2].imag, atol=1e-14)


def test_layout_shapes_count_only_the_kept_orbitals():
    layout = build_reduced_output_layout(RS, PAIRS, CUMSUM, [1, 2])
    np.testing.assert_array_equal(layout.chunk_shapes[:4], [[1, 1], [1, 1], [1, 1], [1, 1]])
    np.testing.assert_array_equal(layout.chunk_boundaries, [0, 1, 2, 3, 4, 5, 6])
    assert layout.nkp == 2


def test_atom_with_no_kept_orbital_gets_empty_blocks():
    layout = build_reduced_output_layout(RS, PAIRS, CUMSUM, [2])
    np.testing.assert_array_equal(layout.chunk_shapes, [[0, 0], [0, 1], [1, 0], [1, 1], [0, 1], [1, 0]])
    assert layout.chunk_boundaries[-1] == 1


def test_targets_are_sorted_within_each_r():
    layout = build_reduced_output_layout(RS, PAIRS, CUMSUM, [0, 1, 2])
    for r in range(len(RS)):
        flat = layout.target_flat[layout.r_bounds[r] : layout.r_bounds[r + 1]]
        assert np.all(np.diff(flat) > 0)


def test_missing_minus_r_is_hermitized_with_r_itself():
    rs = np.array([[1, 0, 0]])
    pairs = np.array([[1, 0, 0, 0, 0]])
    mats = random_mats(1, 2)
    layout = build_reduced_output_layout(rs, pairs, np.array([0, 2]), [0, 1])
    entries = layout.hermitize_real(targets_of(layout, mats).real)
    expected = 0.5 * (mats[0] + mats[0].conj().T).real
    np.testing.assert_allclose(entries, expected.reshape(-1))


def test_partner_of_an_unwritten_entry_is_still_a_target():
    """Only (R=1, 0, 1) is written, but its value needs M(-1)[1, 0], which no block stores."""
    pairs = np.array([[1, 0, 0, 0, 1]])
    mats = random_mats(3, 3)
    layout = build_reduced_output_layout(RS, pairs, CUMSUM, [0, 1, 2])
    entries = layout.hermitize_real(targets_of(layout, mats).real)
    expected = 0.5 * (mats[1][[0, 1], 2] + mats[2][2, [0, 1]].conj()).real
    np.testing.assert_allclose(entries, expected)


# ---------------------------------- hermitization ----------------------------------


def test_hermitization_averages_r_and_minus_r():
    mats = random_mats(3, 3)
    out = hermitize_real_space_blocks(mats, RS)
    np.testing.assert_allclose(out[1], 0.5 * (mats[1] + mats[2].conj().T))
    np.testing.assert_allclose(out[2], out[1].conj().T)
    np.testing.assert_allclose(out[0], 0.5 * (mats[0] + mats[0].conj().T))


def test_hermitization_does_not_change_its_input():
    mats = random_mats(3, 3)
    copy = mats.copy()
    hermitize_real_space_blocks(mats, RS)
    np.testing.assert_array_equal(mats, copy)


def test_hermitization_leaves_a_hermitian_set_alone():
    mats = hermitize_real_space_blocks(random_mats(3, 3), RS)
    np.testing.assert_allclose(hermitize_real_space_blocks(mats, RS), mats)


# ---------------------------------- zero datasets ----------------------------------


def test_zero_dataset_reads_as_zeros_and_takes_no_space(tmp_path):
    path = tmp_path / "zero.h5"
    with h5py.File(path, "w") as f:
        create_zero_dataset(f, "entries", (10**6,), np.float64)
    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(f["entries"][:], np.zeros(10**6))
    assert path.stat().st_size < 10**5


def test_zero_entries_are_rewritten_keeping_everything_else(tmp_path):
    path = tmp_path / "file.h5"
    with h5py.File(path, "w") as f:
        f["atom_pairs"] = np.arange(10)
        entries = f.create_dataset("entries", data=np.ones(10**6, dtype=np.float32))
        entries.attrs["unit"] = "eV"
        f.attrs["origin"] = "test"
    size = path.stat().st_size

    assert zero_h5_entries(path) == path

    assert path.stat().st_size < size / 10
    assert not path.with_name("file.h5.tmp").exists()
    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(f["atom_pairs"][:], np.arange(10))
        np.testing.assert_array_equal(f["entries"][:], np.zeros(10**6))
        assert f["entries"].dtype == np.float32
        assert f["entries"].attrs["unit"] == "eV"
        assert f.attrs["origin"] == "test"


def test_dump_with_no_entries_writes_zeros_of_the_right_size(tmp_path):
    layout = build_reduced_output_layout(RS, PAIRS, CUMSUM, [0, 1, 2])
    path = dump_reduced_entries_h5(tmp_path / "dummy.h5", layout, None)
    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(f["entries"][:], np.zeros(layout.chunk_boundaries[-1]))
        np.testing.assert_array_equal(f["atom_pairs"][:], PAIRS)


def test_dense_writer_with_zero_entries_ignores_the_values(tmp_path):
    path = dump_reduced_matrix_h5(tmp_path / "d.h5", np.empty(0), RS, PAIRS, CUMSUM, [0, 1, 2], zero_entries=True)
    entries, _, shapes, _ = entries_of(path)
    assert np.all(entries == 0)
    assert len(entries) == (shapes[:, 0] * shapes[:, 1]).sum()


def test_dense_writer_needs_the_r_list(tmp_path):
    with pytest.raises(ValueError, match="Rijk_list"):
        dump_reduced_matrix_h5(tmp_path / "d.h5", np.empty(0), None, PAIRS, CUMSUM, [0])
