"""Reading, writing and averaging sparse Hamiltonian files."""

import h5py
import numpy as np
import pytest

from hamilflow.uncertainty.hamiltonian_io import (
    average_predicted_hamiltonians,
    read_deeph_hamiltonian,
    write_deeph_hamiltonian,
)

PAIRS = np.array([[0, 0, 0, 0, 0]])
BOUNDS = np.array([0, 2])
SHAPES = np.array([[1, 2]])


def write(path, entries, pairs=PAIRS, bounds=BOUNDS, shapes=SHAPES):
    write_deeph_hamiltonian(path, pairs, bounds, shapes, np.asarray(entries))
    return path


@pytest.mark.parametrize("entries", [[1.5, 2.5], [1.5 + 1j, 2.5 - 1j]], ids=["real", "complex"])
def test_write_then_read_round_trip(tmp_path, entries):
    path = write(tmp_path / "new" / "folder" / "h.h5", entries)
    for read, expected in zip(read_deeph_hamiltonian(path), (PAIRS, BOUNDS, SHAPES, entries), strict=True):
        np.testing.assert_array_equal(read, expected)


def test_single_precision_entries_are_read_as_double(tmp_path):
    path = tmp_path / "h.h5"
    with h5py.File(path, "w") as f:
        f["atom_pairs"], f["chunk_boundaries"], f["chunk_shapes"] = PAIRS, BOUNDS, SHAPES
        f["entries"] = np.array([1.0, 2.0], dtype=np.float32)
    assert read_deeph_hamiltonian(path)[3].dtype == np.float64


def test_average_is_entrywise(tmp_path):
    paths = [write(tmp_path / f"{i}.h5", [float(i), 2.0 * i]) for i in (1, 2, 6)]
    average_predicted_hamiltonians(paths, tmp_path / "avg" / "h.h5")
    pairs, _, _, entries = read_deeph_hamiltonian(tmp_path / "avg" / "h.h5")
    np.testing.assert_allclose(entries, [3.0, 6.0])
    np.testing.assert_array_equal(pairs, PAIRS)


def test_average_does_not_need_more_than_one_file(tmp_path):
    average_predicted_hamiltonians([write(tmp_path / "a.h5", [1.0, 2.0])], tmp_path / "avg.h5")
    np.testing.assert_allclose(read_deeph_hamiltonian(tmp_path / "avg.h5")[3], [1.0, 2.0])


def test_average_of_nothing_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="must not be empty"):
        average_predicted_hamiltonians([], tmp_path / "avg.h5")


@pytest.mark.parametrize(
    ("other", "message"),
    [
        ({"pairs": np.array([[1, 0, 0, 0, 0]])}, "atom_pairs"),
        ({"bounds": np.array([0, 1])}, "chunk_boundaries"),
        ({"shapes": np.array([[2, 1]])}, "chunk_shapes"),
    ],
)
def test_average_of_other_layouts_is_rejected(tmp_path, other, message):
    a = write(tmp_path / "a.h5", [1.0, 2.0])
    b = write(tmp_path / "b.h5", [1.0, 2.0], **other)
    with pytest.raises(ValueError, match=message):
        average_predicted_hamiltonians([a, b], tmp_path / "avg.h5")


def test_average_of_another_number_of_entries_is_rejected(tmp_path):
    a = write(tmp_path / "a.h5", [1.0, 2.0])
    b = write(tmp_path / "b.h5", [1.0])
    with pytest.raises(ValueError, match="entries shape"):
        average_predicted_hamiltonians([a, b], tmp_path / "avg.h5")
