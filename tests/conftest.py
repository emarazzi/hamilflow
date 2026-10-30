import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pytest
from deepx_dock.CONSTANT import (
    DEEPX_HAMILTONIAN_FILENAME,
    DEEPX_INFO_FILENAME,
    DEEPX_OVERLAP_FILENAME,
    DEEPX_POSCAR_FILENAME,
)

from hamilflow.sparse_hamiltonian.dense_backend import gpu_backend

ELEMENTS_ORBITAL_MAP = {"Mo": [0, 1], "S": [0]}
ATOM_ELEMENTS = ["Mo", "S", "S"]
ATOM_N_ORBITALS = [4, 1, 1]
HALF_RS = [(1, 0, 0), (0, 1, 0), (1, 1, 0)]
CORE_ORBITAL = 0  # the Mo s orbital
CORE_LEVEL = -20000.0  # eV, the scale of the deepest states in FHI-aims all-electron Hamiltonians


def pytest_collection_modifyitems(items):
    gpu_items = [item for item in items if item.get_closest_marker("gpu")]
    if gpu_items and gpu_backend() is None:
        skip_gpu = pytest.mark.skip(reason="no GPU backend available")
        for item in gpu_items:
            item.add_marker(skip_gpu)


@dataclass(frozen=True)
class SyntheticStructure:
    """A DeepH-format structure on disk, with the dense matrices it was written from as a reference."""

    path: Path
    spinful: bool
    Rs: np.ndarray  # (N_R, 3)
    S_R: np.ndarray  # (N_R, dim, dim), spin-expanded when spinful
    H_R: np.ndarray  # (N_R, dim, dim)
    core_level: float | None = None

    def Sk_and_Hk(self, k) -> tuple[np.ndarray, np.ndarray]:
        """Direct sum M(k) = sum_R exp(2 pi i k.R) M(R), independent of any file parsing."""
        phase = np.exp(2j * np.pi * (self.Rs @ np.asarray(k, dtype=np.float64)))
        return np.tensordot(phase, self.S_R, axes=1), np.tensordot(phase, self.H_R, axes=1)


def _write_blocks(path: Path, blocks: dict[tuple, np.ndarray], keys: list[tuple]) -> None:
    sizes = [blocks[key].size for key in keys]
    with h5py.File(path, "w") as f:
        f["atom_pairs"] = np.array(keys, dtype=np.int32)
        f["chunk_boundaries"] = np.cumsum([0, *sizes], dtype=np.int32)
        f["chunk_shapes"] = np.array([blocks[key].shape for key in keys], dtype=np.int32)
        f["entries"] = np.concatenate([blocks[key].ravel() for key in keys])


def write_synthetic_structure(
    path: Path, spinful: bool = False, seed: int = 0, core_level: float | None = None
) -> SyntheticStructure:
    """
    Write POSCAR, info.json, overlap.h5 and hamiltonian.h5 for a Mo1S2-like cell of 6 orbitals (12 when spinful).

    Atoms have different orbital counts and some inter-cell atom pairs are absent, as in real files. M(-R) is
    the conjugate transpose of M(R), so H(k) and S(k) are Hermitian, and the off-site overlap is scaled so
    that every eigenvalue of S(k) lies in [0.5, 1.5] at every k (Gershgorin).

    ``core_level`` mimics an all-electron Hamiltonian: the Mo s orbital gets this on-site energy and is fully
    decoupled, in H and S, from everything else, so it is an exact flat eigenvalue at every k. Real core states
    couple weakly to the valence, which this does not reproduce.
    """
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_atoms = len(ATOM_N_ORBITALS)
    n_orbitals = sum(ATOM_N_ORBITALS)
    offsets = np.cumsum([0, *ATOM_N_ORBITALS])
    orbital_atom = np.repeat(np.arange(n_atoms), ATOM_N_ORBITALS)
    dim = 2 * n_orbitals if spinful else n_orbitals

    (path / DEEPX_POSCAR_FILENAME).write_text(
        "synthetic\n1.0\n3.2 0.0 0.0\n-1.6 2.8 0.0\n0.0 0.0 15.0\nMo S\n1 2\nDirect\n"
        "0.33 0.67 0.50\n0.67 0.33 0.40\n0.67 0.33 0.60\n"
    )
    (path / DEEPX_INFO_FILENAME).write_text(
        json.dumps(
            {
                "atoms_quantity": n_atoms,
                "orbits_quantity": n_orbitals,
                "orthogonal_basis": False,
                "spinful": spinful,
                "fermi_energy_eV": -5.0,
                "elements_orbital_map": ELEMENTS_ORBITAL_MAP,
                "occupation": n_orbitals,
            }
        )
    )

    zero = (0, 0, 0)
    pairs = {zero: np.ones((n_atoms, n_atoms), dtype=bool)}
    for R in HALF_RS:
        pairs[R] = rng.random((n_atoms, n_atoms)) < 0.7
        pairs[tuple(-c for c in R)] = pairs[R].T
    Rs = list(pairs)

    # Overlap: real, identity on site, scaled by its largest absolute row sum
    S_R = {}
    for R in [zero, *HALF_RS]:
        block = rng.uniform(-1, 1, (n_orbitals, n_orbitals)) * pairs[R][np.ix_(orbital_atom, orbital_atom)]
        S_R[R] = block
        S_R[tuple(-c for c in R)] = block.T
    S_R[zero] = (S_R[zero] + S_R[zero].T) / 2
    np.fill_diagonal(S_R[zero], 0.0)
    if core_level is not None:
        for block in S_R.values():
            block[CORE_ORBITAL, :] = 0.0
            block[:, CORE_ORBITAL] = 0.0
    scale = 0.5 / np.abs(np.stack(list(S_R.values()))).sum(axis=0).sum(axis=1).max()
    S_R = {R: scale * block for R, block in S_R.items()}
    S_R[zero] += np.eye(n_orbitals)

    # Hamiltonian: complex with spin blocks when spinful, real otherwise
    H_R = {}
    for R in [zero, *HALF_RS]:
        block = rng.uniform(-1, 1, (dim, dim))
        if spinful:
            block = block + 1j * rng.uniform(-1, 1, (dim, dim))
        spin_atom = np.tile(orbital_atom, 2) if spinful else orbital_atom
        block = block * pairs[R][np.ix_(spin_atom, spin_atom)]
        H_R[R] = block
        H_R[tuple(-c for c in R)] = block.conj().T
    H_R[zero] = (H_R[zero] + H_R[zero].conj().T) / 2
    if core_level is not None:
        core = [CORE_ORBITAL, CORE_ORBITAL + n_orbitals] if spinful else [CORE_ORBITAL]
        for block in H_R.values():
            block[core, :] = 0.0
            block[:, core] = 0.0
        H_R[zero][core, core] = core_level

    S_blocks = {}
    H_blocks = {}
    for R, atom_mask in pairs.items():
        for i, j in zip(*np.nonzero(atom_mask)):
            key = (*R, int(i), int(j))
            i_slice = np.arange(offsets[i], offsets[i + 1])
            j_slice = np.arange(offsets[j], offsets[j + 1])
            S_blocks[key] = S_R[R][np.ix_(i_slice, j_slice)]
            if spinful:
                i_slice = np.r_[i_slice, i_slice + n_orbitals]
                j_slice = np.r_[j_slice, j_slice + n_orbitals]
            H_blocks[key] = H_R[R][np.ix_(i_slice, j_slice)]
    # Real files do not group atom pairs by R (R and -R alternate), so no order may be relied on
    keys = list(S_blocks)
    keys = [keys[i] for i in rng.permutation(len(keys))]
    _write_blocks(path / DEEPX_OVERLAP_FILENAME, S_blocks, keys)
    _write_blocks(path / DEEPX_HAMILTONIAN_FILENAME, H_blocks, keys)

    S_dense = np.stack([S_R[R] for R in Rs])
    if spinful:
        zeros = np.zeros_like(S_dense)
        S_dense = np.block([[S_dense, zeros], [zeros, S_dense]])
    return SyntheticStructure(path, spinful, np.array(Rs), S_dense, np.stack([H_R[R] for R in Rs]), core_level)


@pytest.fixture(
    scope="session",
    params=[(False, None), (True, None), (False, CORE_LEVEL), (True, CORE_LEVEL)],
    ids=["spinless", "spinful", "spinless-core", "spinful-core"],
)
def synthetic_structure(request, tmp_path_factory) -> SyntheticStructure:
    """A read-only structure shared by the session, with and without spin and core. Do not write to its directory."""
    spinful, core_level = request.param
    return write_synthetic_structure(tmp_path_factory.mktemp("synthetic"), spinful=spinful, core_level=core_level)


@pytest.fixture
def make_synthetic_structure(tmp_path):
    """Factory for tests that need their own structure, e.g. another seed: make(name, spinful=..., seed=...)."""

    def make(
        name: str = "structure", spinful: bool = False, seed: int = 0, core_level: float | None = None
    ) -> SyntheticStructure:
        return write_synthetic_structure(tmp_path / name, spinful=spinful, seed=seed, core_level=core_level)

    return make
