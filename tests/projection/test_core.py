"""
The whole projection, from a structure on disk to a reduced structure on disk.

The synthetic cell has Mo (s 0, p 1-3) and two S (s 4 and s 5), and hoppings up to one cell away, so a k-mesh
of 3 x 3 x 1 represents its real-space matrices exactly.
"""

import json

import h5py
import numpy as np
import pytest
import scipy.linalg
from deepx_dock.compute.eigen.hamiltonian import HamiltonianObj
from deepx_dock.CONSTANT import DEEPX_HAMILTONIAN_FILENAME, DEEPX_OVERLAP_FILENAME

from hamilflow.dft_workflows import kpoints
from hamilflow.projection import ProjectionConfig, run_projection
from hamilflow.projection.core import _resolve_projection_kgrid
from hamilflow.projection.io import dump_reduced_matrix_h5, hermitize_real_space_blocks
from hamilflow.projection.kspace import (
    apply_custom_kspace_transform,
    apply_truncation_kspace_transform,
    build_uniform_kmesh,
    k_to_r_operator,
)
from hamilflow.sparse_hamiltonian import SparseHamiltonianObj

KGRID = (3, 3, 1)
KPOINTS = [(0.0, 0.0, 0.0), (0.13, 0.37, 0.0), (0.21, -0.44, 0.3)]
CORE_LEVEL = -20000.0
REMOVE_P = [{"target_elements": ["Mo"], "remove_orbitals": ["p"]}]  # orbitals 1, 2, 3
REMOVE_CORE = [{"target_elements": ["Mo"], "remove_shells": ["1s"]}]  # orbital 0


@pytest.fixture
def structure(make_synthetic_structure):
    return make_synthetic_structure("input")


def project(structure, tmp_path, plan=REMOVE_P, mode="schur", **kwargs):
    config = ProjectionConfig(
        structure.path, tmp_path / "output", kgrid=KGRID, reduction_mode=mode, n_workers=1, **kwargs
    )
    return run_projection(config, plan)


def read_h5(path):
    with h5py.File(path, "r") as f:
        return {name: f[name][:] for name in f}


# ---------------------------------- exact results ----------------------------------


@pytest.mark.parametrize("k", KPOINTS)
def test_truncation_gives_the_submatrices_of_the_input(structure, tmp_path, k):
    result = project(structure, tmp_path, mode="truncate")
    Sk, Hk = SparseHamiltonianObj(result.output_dir).Sk_and_Hk(np.array(k))
    S0, H0 = structure.Sk_and_Hk(k)
    keep = np.ix_([0, 4, 5], [0, 4, 5])
    np.testing.assert_allclose(Hk, H0[keep], atol=1e-10)
    np.testing.assert_allclose(Sk, S0[keep], atol=1e-10)


@pytest.mark.parametrize("mode", ["schur", "truncate"])
@pytest.mark.parametrize("k", KPOINTS)
def test_removing_a_decoupled_core_level_keeps_the_valence_bands(make_synthetic_structure, tmp_path, mode, k):
    """The all-electron case: the -20000 eV level goes, every other eigenvalue stays."""
    structure = make_synthetic_structure("input", core_level=CORE_LEVEL)
    result = project(structure, tmp_path, REMOVE_CORE, mode)
    Sk, Hk = SparseHamiltonianObj(result.output_dir).Sk_and_Hk(np.array(k))
    S0, H0 = structure.Sk_and_Hk(k)
    full = scipy.linalg.eigh(H0, S0, eigvals_only=True)
    assert full[0] == pytest.approx(CORE_LEVEL)
    np.testing.assert_allclose(scipy.linalg.eigh(Hk, Sk, eigvals_only=True), full[1:], atol=1e-9)


@pytest.mark.parametrize("mode", ["schur", "truncate"])
def test_projection_matches_the_dense_pipeline(structure, tmp_path, mode):
    """Written entries against the straightforward dense route: transform, inverse FT, hermitize, select."""
    result = project(structure, tmp_path, mode=mode)
    obj = SparseHamiltonianObj(structure.path)
    ks = build_uniform_kmesh(KGRID)
    Sk, Hk = obj.Sk_and_Hk(ks)
    transform = apply_custom_kspace_transform if mode == "schur" else apply_truncation_kspace_transform
    H_new, S_new = transform(Hk, Sk, [1, 2, 3])
    for dense_k, written in ((H_new, result.hamiltonian_path), (S_new, result.overlap_path)):
        mats = hermitize_real_space_blocks(k_to_r_operator(ks, obj.Rijk_list, dense_k), obj.Rijk_list)
        reference = dump_reduced_matrix_h5(
            tmp_path / "reference.h5", mats, obj.Rijk_list, obj.atom_pairs, obj.atom_num_orbits_cumsum, [0, 4, 5]
        )
        expected = read_h5(reference)
        actual = read_h5(written)
        np.testing.assert_allclose(actual["entries"], expected["entries"], atol=1e-10)
        for name in ("atom_pairs", "chunk_shapes", "chunk_boundaries"):
            np.testing.assert_array_equal(actual[name], expected[name])


def test_schur_differs_from_truncation_when_orbitals_couple(structure, tmp_path):
    """Otherwise the two modes could silently be the same code."""
    schur = project(structure, tmp_path, mode="schur")
    entries_schur = read_h5(schur.hamiltonian_path)["entries"]
    truncate = run_projection(
        ProjectionConfig(structure.path, tmp_path / "t", kgrid=KGRID, reduction_mode="truncate", n_workers=1), REMOVE_P
    )
    assert np.abs(entries_schur - read_h5(truncate.hamiltonian_path)["entries"]).max() > 1e-3


@pytest.mark.parametrize("n_workers", [1, 2])
def test_result_does_not_depend_on_the_workers(structure, tmp_path, n_workers):
    reference = project(structure, tmp_path)
    config = ProjectionConfig(structure.path, tmp_path / "w", kgrid=KGRID, n_workers=n_workers)
    other = run_projection(config, REMOVE_P)
    np.testing.assert_allclose(
        read_h5(other.hamiltonian_path)["entries"], read_h5(reference.hamiltonian_path)["entries"], atol=1e-12
    )


def test_automatic_worker_count_gives_the_same_result(structure, tmp_path):
    reference = project(structure, tmp_path)
    other = run_projection(ProjectionConfig(structure.path, tmp_path / "w", kgrid=KGRID), REMOVE_P)
    np.testing.assert_allclose(
        read_h5(other.hamiltonian_path)["entries"], read_h5(reference.hamiltonian_path)["entries"], atol=1e-12
    )


def test_two_truncations_in_sequence_equal_one_of_both(structure, tmp_path):
    both = run_projection(
        ProjectionConfig(structure.path, tmp_path / "both", kgrid=KGRID, reduction_mode="truncate", n_workers=1),
        [*REMOVE_P, *REMOVE_CORE],
    )
    first = run_projection(
        ProjectionConfig(structure.path, tmp_path / "first", kgrid=KGRID, reduction_mode="truncate", n_workers=1),
        REMOVE_P,
    )
    second = run_projection(
        ProjectionConfig(first.output_dir, tmp_path / "second", kgrid=KGRID, reduction_mode="truncate", n_workers=1),
        REMOVE_CORE,
    )
    np.testing.assert_allclose(
        read_h5(second.hamiltonian_path)["entries"], read_h5(both.hamiltonian_path)["entries"], atol=1e-10
    )


# ---------------------------------- the output ----------------------------------


def test_output_is_a_valid_structure_for_upstream_too(structure, tmp_path):
    result = project(structure, tmp_path)
    Sk, Hk = HamiltonianObj(result.output_dir).Sk_and_Hk(np.array(KPOINTS[1]))
    assert Hk.shape == (3, 3)
    np.testing.assert_allclose(Hk, Hk.conj().T, atol=1e-10)
    np.testing.assert_allclose(Sk, Sk.conj().T, atol=1e-10)


def test_output_is_hermitian_at_general_k(structure, tmp_path):
    result = project(structure, tmp_path)
    Sk, Hk = SparseHamiltonianObj(result.output_dir).Sk_and_Hk(np.array(KPOINTS[2]))
    np.testing.assert_allclose(Hk, Hk.conj().T, atol=1e-10)
    np.testing.assert_allclose(Sk, Sk.conj().T, atol=1e-10)


def test_output_is_hermitian_even_when_the_input_is_not(structure, tmp_path):
    """Predicted Hamiltonians need not satisfy H(-R) = H(R)^T: the written blocks must, on a kept atom pair."""
    with h5py.File(structure.path / DEEPX_HAMILTONIAN_FILENAME, "r+") as f:
        pairs = f["atom_pairs"][:]
        i_pair = next(i for i, p in enumerate(pairs) if p[:3].any() and p[3] in (1, 2) and p[4] in (1, 2))
        entries = f["entries"][:]
        entries[f["chunk_boundaries"][i_pair]] += 1.0
        f["entries"][:] = entries
    result = project(structure, tmp_path, mode="truncate")
    _, Hk = SparseHamiltonianObj(result.output_dir).Sk_and_Hk(np.array(KPOINTS[1]))
    np.testing.assert_allclose(Hk, Hk.conj().T, atol=1e-10)


def test_info_json_describes_the_reduced_basis(structure, tmp_path):
    info = json.loads(project(structure, tmp_path).info_path.read_text())
    assert info["orbits_quantity"] == 3
    assert info["elements_orbital_map"] == {"Mo": [0], "S": [0]}
    assert info["occupation"] == 6 - 2 * 3  # the input has 6 electrons, schur removes 3 orbitals
    assert info["fermi_energy_eV"] == -5.0


def test_truncation_keeps_the_occupation(structure, tmp_path):
    assert json.loads(project(structure, tmp_path, mode="truncate").info_path.read_text())["occupation"] == 6


def test_poscar_is_copied(structure, tmp_path):
    result = project(structure, tmp_path)
    assert (result.output_dir / "POSCAR").read_text() == (structure.path / "POSCAR").read_text()


def test_input_files_are_not_changed(structure, tmp_path):
    before = {p.name: p.read_bytes() for p in structure.path.iterdir()}
    project(structure, tmp_path)
    assert {p.name: p.read_bytes() for p in structure.path.iterdir()} == before


def test_metadata_records_the_reduction(structure, tmp_path):
    result = project(structure, tmp_path)
    meta = result.metadata
    assert meta["removed_global_indices"] == [1, 2, 3]
    assert meta["kept_global_indices"] == [0, 4, 5]
    assert (meta["original_orbits_quantity"], meta["reduced_orbits_quantity"]) == (6, 3)
    assert meta["reduced_orbitals_per_atom"] == [1, 1, 1]
    assert meta["resolved_kgrid"] == list(KGRID)
    assert meta["reduction_mode"] == "schur"
    assert meta["rule_plan_resolution"][0]["resolved"]["selected_atoms"] == [0]
    assert meta["overlap_imag_max_before_real_cast"] < 1e-10
    assert json.loads(result.meta_path.read_text()) == meta


def test_result_dict_is_serializable(structure, tmp_path):
    result = project(structure, tmp_path).to_dict()
    assert json.loads(json.dumps(result))["hamiltonian_path"] == str(tmp_path / "output" / DEEPX_HAMILTONIAN_FILENAME)


def test_plan_can_be_a_file(structure, tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"rules": REMOVE_P}))
    assert project(structure, tmp_path, plan=path).metadata["removed_global_indices"] == [1, 2, 3]


def test_empty_plan_keeps_the_structure(structure, tmp_path):
    result = project(structure, tmp_path, plan=[])
    Sk, Hk = SparseHamiltonianObj(result.output_dir).Sk_and_Hk(np.array(KPOINTS[1]))
    S0, H0 = structure.Sk_and_Hk(KPOINTS[1])
    np.testing.assert_allclose(Hk, H0, atol=1e-10)
    np.testing.assert_allclose(Sk, S0, atol=1e-10)


# ---------------------------------- overlap only ----------------------------------


def test_overlap_only_writes_no_hamiltonian_and_the_same_overlap(structure, tmp_path):
    full = project(structure, tmp_path)
    only = run_projection(ProjectionConfig(structure.path, tmp_path / "o", kgrid=KGRID, overlap_only=True), REMOVE_P)
    assert only.hamiltonian_path is None
    assert not (tmp_path / "o" / DEEPX_HAMILTONIAN_FILENAME).exists()
    np.testing.assert_allclose(read_h5(only.overlap_path)["entries"], read_h5(full.overlap_path)["entries"], atol=1e-12)


def test_dummy_hamiltonian_is_zero_with_the_right_layout(structure, tmp_path):
    full = project(structure, tmp_path)
    result = run_projection(
        ProjectionConfig(structure.path, tmp_path / "o", kgrid=KGRID, overlap_only=True, write_dummy_hamiltonian=True),
        REMOVE_P,
    )
    dummy, reference = read_h5(result.hamiltonian_path), read_h5(full.hamiltonian_path)
    assert np.all(dummy["entries"] == 0)
    assert dummy["entries"].shape == reference["entries"].shape
    np.testing.assert_array_equal(dummy["atom_pairs"], reference["atom_pairs"])


def test_overlap_only_runs_can_be_chained_through_the_dummy(structure, tmp_path):
    first = run_projection(
        ProjectionConfig(
            structure.path,
            tmp_path / "first",
            kgrid=KGRID,
            reduction_mode="truncate",
            overlap_only=True,
            write_dummy_hamiltonian=True,
        ),
        REMOVE_P,
    )
    second = run_projection(
        ProjectionConfig(
            first.output_dir, tmp_path / "second", kgrid=KGRID, reduction_mode="truncate", overlap_only=True
        ),
        REMOVE_CORE,
    )
    assert read_h5(second.overlap_path)["chunk_boundaries"][-1] > 0
    assert not (tmp_path / "second" / DEEPX_HAMILTONIAN_FILENAME).exists()


# ---------------------------------- errors ----------------------------------


def test_missing_hamiltonian_is_explained(structure, tmp_path):
    (structure.path / DEEPX_HAMILTONIAN_FILENAME).unlink()
    with pytest.raises(FileNotFoundError, match="overlap_only mode still requires"):
        project(structure, tmp_path)
    assert (structure.path / DEEPX_OVERLAP_FILENAME).exists()


def test_removal_that_differs_between_atoms_of_an_element_is_rejected(structure, tmp_path):
    with pytest.raises(ValueError, match="Inconsistent reduced shells"):
        project(structure, tmp_path, plan=[{"target_atom_indices": [1], "remove_orbitals": ["s"]}])


def test_unknown_reduction_mode_is_rejected(structure, tmp_path):
    with pytest.raises(ValueError, match="Unsupported reduction_mode"):
        project(structure, tmp_path, mode="other")


def test_removing_everything_is_rejected(structure, tmp_path):
    with pytest.raises(ValueError, match="Cannot remove all"):
        project(structure, tmp_path, plan=[{"remove_orbitals": ["s", "p"]}])


def test_force_2d_and_1d_are_exclusive(tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        ProjectionConfig(tmp_path, tmp_path, force_2d=True, force_1d=True)


def test_spinful_input_is_rejected_before_writing_anything(make_synthetic_structure, tmp_path):
    structure = make_synthetic_structure("input", spinful=True)
    with pytest.raises(NotImplementedError, match="spinful"):
        project(structure, tmp_path, REMOVE_CORE)
    assert not (tmp_path / "output").exists()


# ---------------------------------- k-grid ----------------------------------


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, (4, 5, 6)),
        ({"force_2d": True}, (4, 5, 1)),
        ({"force_1d": True}, (1, 1, 6)),
        ({"user_kpoints_settings": {}}, (4, 5, 6)),
    ],
)
def test_kgrid_follows_the_config(tmp_path, options, expected):
    config = ProjectionConfig(tmp_path, tmp_path, kgrid=(4, 5, 6), **options)
    assert _resolve_projection_kgrid(config) == expected


def run_with_kgrid(structure, tmp_path, kgrid):
    config = ProjectionConfig(structure.path, tmp_path / "output", kgrid=kgrid, reduction_mode="truncate", n_workers=1)
    return run_projection(config, REMOVE_P)


@pytest.mark.parametrize("kgrid", [(2, 3, 1), (3, 2, 1), (2, 2, 1)])
def test_grid_too_coarse_for_the_r_vectors_warns_but_still_runs(structure, tmp_path, caplog, kgrid):
    """The fixture's R go from -1 to 1 in-plane, which needs 3 points per direction."""
    with caplog.at_level("WARNING", logger="hamilflow.projection.core"):
        result = run_with_kgrid(structure, tmp_path, kgrid)
    assert "too coarse for the stored R vectors, which need at least (3, 3, 1)" in caplog.text
    assert result.overlap_path.exists()


@pytest.mark.parametrize("kgrid", [(3, 3, 1), (4, 5, 2)])
def test_grid_that_separates_the_r_vectors_does_not_warn(structure, tmp_path, caplog, kgrid):
    with caplog.at_level("WARNING", logger="hamilflow.projection.core"):
        run_with_kgrid(structure, tmp_path, kgrid)
    assert "too coarse" not in caplog.text


def test_warning_is_right_about_what_it_warns_of(structure, tmp_path):
    """Below the limit the output really is wrong, and at the limit it is exact."""
    k = np.array(KPOINTS[1])
    keep = np.ix_([0, 4, 5], [0, 4, 5])
    H0 = structure.Sk_and_Hk(k)[1][keep]
    exact = run_with_kgrid(structure, tmp_path / "a", (3, 3, 1))
    np.testing.assert_allclose(SparseHamiltonianObj(exact.output_dir).Sk_and_Hk(k)[1], H0, atol=1e-10)
    coarse = run_with_kgrid(structure, tmp_path / "b", (2, 2, 1))
    assert np.abs(SparseHamiltonianObj(coarse.output_dir).Sk_and_Hk(k)[1] - H0).max() > 1e-3


def test_user_kpoint_settings_go_through_the_real_sampling_function(structure, tmp_path):
    config = ProjectionConfig(structure.path, tmp_path, kgrid=(9, 9, 9), user_kpoints_settings={"k_grid": [2, 3, 5]})
    assert _resolve_projection_kgrid(config) == (2, 3, 5)
    config = ProjectionConfig(structure.path, tmp_path, user_kpoints_settings={"k_grid": [2, 3, 5]}, force_2d=True)
    assert _resolve_projection_kgrid(config) == (2, 3, 1)


def test_user_kpoint_settings_take_over_the_kgrid(structure, tmp_path, monkeypatch):
    calls = {}

    def fake(structure, user_kpoints_settings, force_2d, force_1d):
        calls.update(settings=user_kpoints_settings, force_2d=force_2d, n_sites=len(structure))
        return {"k_grid": [2, 3, 1]}

    monkeypatch.setattr(kpoints, "get_ksampling", fake)
    settings = {"reciprocal_density": 100}
    config = ProjectionConfig(structure.path, tmp_path, kgrid=(9, 9, 9), user_kpoints_settings=settings, force_2d=True)
    assert _resolve_projection_kgrid(config) == (2, 3, 1)
    assert calls == {"settings": settings, "force_2d": True, "n_sites": 3}


@pytest.mark.parametrize(
    ("sampling", "message"),
    [({"k_path": []}, "uniform k_grid"), (None, "uniform k_grid"), ({"k_grid": [1, 2]}, "three")],
)
def test_user_settings_that_give_no_uniform_grid_are_rejected(structure, tmp_path, monkeypatch, sampling, message):
    monkeypatch.setattr(kpoints, "get_ksampling", lambda **kwargs: sampling)
    config = ProjectionConfig(structure.path, tmp_path, user_kpoints_settings={"line_density": 10})
    with pytest.raises(ValueError, match=message):
        _resolve_projection_kgrid(config)
