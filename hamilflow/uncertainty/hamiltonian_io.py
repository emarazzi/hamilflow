from pathlib import Path
import h5py
import numpy as np

def read_deeph_hamiltonian(h5_path: str | Path):
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        atom_pairs = np.array(f["atom_pairs"][:], dtype=np.int64)
        chunk_boundaries = np.array(f["chunk_boundaries"][:], dtype=np.int64)
        chunk_shapes = np.array(f["chunk_shapes"][:], dtype=np.int64)

        entries_dset = f["entries"]
        entries_dtype = np.complex128 if entries_dset.dtype.kind == "c" else np.float64
        entries = np.array(entries_dset[:], dtype=entries_dtype)

    return atom_pairs, chunk_boundaries, chunk_shapes, entries


def write_deeph_hamiltonian(
    h5_path: str | Path,
    atom_pairs: np.ndarray,
    chunk_boundaries: np.ndarray,
    chunk_shapes: np.ndarray,
    entries: np.ndarray,
):
    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("atom_pairs", data=atom_pairs)
        f.create_dataset("chunk_boundaries", data=chunk_boundaries)
        f.create_dataset("chunk_shapes", data=chunk_shapes)
        f.create_dataset("entries", data=entries)


def average_predicted_hamiltonians(pred_h5_paths, output_h5_path):
    """Average a list of deeph-style Hamiltonian HDF5 prediction files.

    Parameters
    - pred_h5_paths: iterable of paths to predicted hamiltonian.h5 files
    - output_h5_path: destination path to write the averaged Hamiltonian
    """
    pred_h5_paths = [Path(p) for p in pred_h5_paths]
    if not pred_h5_paths:
        raise ValueError("pred_h5_paths must not be empty")

    atom_pairs0, chunk_boundaries0, chunk_shapes0, entries0 = read_deeph_hamiltonian(pred_h5_paths[0])
    entries_sum = np.array(entries0, copy=True)

    for h5_path in pred_h5_paths[1:]:
        atom_pairs, chunk_boundaries, chunk_shapes, entries = read_deeph_hamiltonian(h5_path)

        if not np.array_equal(atom_pairs0, atom_pairs):
            raise ValueError(f"atom_pairs mismatch in {h5_path}")
        if not np.array_equal(chunk_boundaries0, chunk_boundaries):
            raise ValueError(f"chunk_boundaries mismatch in {h5_path}")
        if not np.array_equal(chunk_shapes0, chunk_shapes):
            raise ValueError(f"chunk_shapes mismatch in {h5_path}")
        if entries.shape != entries_sum.shape:
            raise ValueError(f"entries shape mismatch in {h5_path}")

        entries_sum += entries

    entries_avg = entries_sum / len(pred_h5_paths)

    write_deeph_hamiltonian(
        output_h5_path,
        atom_pairs0,
        chunk_boundaries0,
        chunk_shapes0,
        entries_avg,
    )


__all__ = ["read_deeph_hamiltonian", "write_deeph_hamiltonian", "average_predicted_hamiltonians"]
