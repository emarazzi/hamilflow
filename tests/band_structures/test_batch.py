"""Folder-wide band structures: every DeepH subfolder, any solver, against a direct diagonalization."""

import h5py
import numpy as np
import pytest
import scipy.linalg

from hamilflow.band_structures.batch import BandOptions, compute_band_structure, find_structure_dirs, main, run_folder
from hamilflow.sparse_hamiltonian import SparseHamiltonianObj
from hamilflow.sparse_hamiltonian.eigensolver import eigh_generalized, eigh_generalized_sparse

LABELS = [["GAMMA", "X"], ["Y", "GAMMA"]]


@pytest.fixture
def kpath_file(tmp_path_factory):
    """A path with its own points: the synthetic cell's SeeK-path labels depend on its exact shape."""
    import json

    f = tmp_path_factory.mktemp("kpath") / "path.json"
    f.write_text(json.dumps({"points": {"GAMMA": [0, 0, 0], "X": [0.5, 0, 0], "Y": [0, 0.5, 0.2]},
                             "path": "GAMMA-X|Y-GAMMA", "density": 10}))
    return str(f)


@pytest.fixture
def parent(tmp_path, make_synthetic_structure):
    for i, spinful in enumerate([False, True]):
        make_synthetic_structure(tmp_path / f"structure_{i}", spinful=spinful, seed=i)
    (tmp_path / "structure_junk").mkdir()  # no inputs: skipped
    (tmp_path / "notes.txt").write_text("not a folder")
    return tmp_path


def reference_bands(case_dir, kpoints):
    obj = SparseHamiltonianObj(case_dir)
    return np.stack([scipy.linalg.eigh(*obj.Sk_and_Hk(k)[::-1], eigvals_only=True) for k in kpoints], axis=1)


def read(path):
    with h5py.File(path, "r") as f:
        return {k: f[k][()] for k in ("band_data", "kpoints_frac_list", "fermi_energy_before_shift_eV",
                                      "hsk_symbol_list")}


def test_eigh_generalized_sparse_matches_dense(synthetic_structure):
    obj = SparseHamiltonianObj(synthetic_structure.path)
    for k in [(0, 0, 0), (0.13, 0.37, 0.0), (0.21, -0.44, 0.3)]:
        H, S = obj._sparse_k(np.array(k)), obj._s_obj._sparse_k(np.array(k))
        np.testing.assert_allclose(eigh_generalized_sparse(H, S, eigvals_only=True),
                                   eigh_generalized(H.toarray(), S.toarray(), eigvals_only=True), atol=1e-11)


def test_diag_sparse_assembly_matches_default(synthetic_structure):
    obj = SparseHamiltonianObj(synthetic_structure.path)
    ks = np.array([(0, 0, 0), (0.5, 0, 0), (0.21, -0.44, 0.3)])
    np.testing.assert_allclose(obj.diag(ks, n_jobs=2, sparse_assembly=True), obj.diag(ks, n_jobs=2), atol=1e-11)
    with pytest.raises(ValueError, match="sparse_assembly"):
        obj.diag(ks, sparse_assembly=True, sparse_calc=True)


def test_find_structure_dirs_skips_incomplete(parent):
    assert [d.name for d in find_structure_dirs(parent)] == ["structure_0", "structure_1"]


def test_run_folder_full_matches_direct_diagonalization(parent, kpath_file):
    results = run_folder(parent, BandOptions(kpath=kpath_file, solver="full", n_jobs=1))
    assert set(results) == {parent / "structure_0", parent / "structure_1"}
    assert all(err is None for err in results.values())
    for d in results:
        out = read(d / "band_data.h5")
        ref = reference_bands(d, out["kpoints_frac_list"])
        np.testing.assert_allclose(out["band_data"] + out["fermi_energy_before_shift_eV"], ref, atol=1e-10)
        assert [list(map(bytes.decode, s)) for s in out["hsk_symbol_list"]] == LABELS


def test_default_path_is_seekpath(parent):
    compute_band_structure(parent / "structure_0", BandOptions(solver="full", n_jobs=1))
    labels = {s.decode() for seg in read(parent / "structure_0" / "band_data.h5")["hsk_symbol_list"] for s in seg}
    assert "GAMMA" in labels and len(labels) > 2


def test_band_window_keeps_same_indices_at_every_k(parent, kpath_file):
    d = parent / "structure_0"
    full = compute_band_structure(d, BandOptions(kpath=kpath_file, solver="full", n_jobs=1)).band_data
    window = compute_band_structure(d, BandOptions(kpath=kpath_file, solver="full", n_jobs=1, num_band=3, emin=-1.0))
    start = int(np.nonzero(full.max(axis=1) >= -1.0)[0][0])
    start = min(start, full.shape[0] - 3)
    np.testing.assert_allclose(window.band_data + window.fermi_energy_before_shift,
                               full[start:start + 3] + window.fermi_energy_before_shift, atol=1e-12)


def test_sparse_solver_matches_full(parent, kpath_file):
    d = parent / "structure_1"
    full = compute_band_structure(d, BandOptions(kpath=kpath_file, solver="full", n_jobs=1))
    sparse = compute_band_structure(d, BandOptions(kpath=kpath_file, solver="sparse", n_jobs=1, num_band=4,
                                                   emin=-1e3, output_filename="sparse.h5"))
    ref = full.band_data + full.fermi_energy_before_shift
    got = sparse.band_data + sparse.fermi_energy_before_shift
    np.testing.assert_allclose(got, ref[:4], atol=1e-6)


def test_failure_in_one_structure_does_not_stop_the_others(parent, kpath_file):
    (parent / "structure_1" / "overlap.h5").write_bytes(b"corrupt")
    results = run_folder(parent, BandOptions(kpath=kpath_file, solver="full", n_jobs=1))
    assert results[parent / "structure_0"] is None
    assert results[parent / "structure_1"] is not None


def test_gpu_solver_without_gpu_raises(parent, monkeypatch, kpath_file):
    monkeypatch.setenv("HAMILFLOW_GPU", "0")
    from hamilflow.sparse_hamiltonian import dense_backend

    monkeypatch.setattr(dense_backend, "_gpu_cache", {})
    results = run_folder(parent, BandOptions(kpath=kpath_file, solver="gpu", n_jobs=1), pattern="structure_0")
    assert "no GPU backend" in results[parent / "structure_0"]


def test_cli_workers_and_skip_existing(parent, kpath_file):
    main([str(parent), "--kpath", kpath_file, "--solver", "full", "--workers", "2", "--n-jobs", "1"])
    out = parent / "structure_0" / "band_data.h5"
    mtime = out.stat().st_mtime_ns
    main([str(parent), "--kpath", kpath_file, "--solver", "full", "--skip-existing"])
    assert out.stat().st_mtime_ns == mtime
    assert (parent / "structure_1" / "band_data.h5").is_file()


@pytest.mark.gpu
def test_gpu_sparse_upload_matches_cpu(synthetic_structure, monkeypatch):
    """With a GPU and matrices above GPU_MIN_DIM the sparse matrices are densified on the device."""
    from hamilflow.sparse_hamiltonian import dense_backend

    monkeypatch.setattr("hamilflow.sparse_hamiltonian.eigensolver.get_backend",
                        lambda n: dense_backend.gpu_backend())
    obj = SparseHamiltonianObj(synthetic_structure.path)
    k = np.array([0.21, -0.44, 0.3])
    H, S = obj._sparse_k(k), obj._s_obj._sparse_k(k)
    ref = scipy.linalg.eigh(H.toarray(), S.toarray(), eigvals_only=True)
    np.testing.assert_allclose(eigh_generalized_sparse(H, S, eigvals_only=True), ref, atol=1e-10)
    vals, vecs = eigh_generalized_sparse(H, S)
    np.testing.assert_allclose(H.toarray() @ vecs, S.toarray() @ vecs * vals, atol=1e-10)
