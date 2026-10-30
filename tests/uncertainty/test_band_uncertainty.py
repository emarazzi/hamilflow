"""
Band uncertainty of an ensemble: the expected values are recomputed in the tests from dense matrices with scipy.

The ensemble members are the fixture's Hamiltonian plus a small random Hermitian perturbation, so the same block
layout and overlap. With mesh (2, 2, 1) the bands are computed at a few k-points, and every structure has 6
electrons: 3 occupied bands of 6 when spinless, 6 of 12 when spinful.
"""

import dataclasses
import json
import logging
import shutil
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import scipy.linalg
from deepx_dock.CONSTANT import DEEPX_HAMILTONIAN_FILENAME, DEEPX_INFO_FILENAME, DEEPX_OVERLAP_FILENAME

from hamilflow.uncertainty import band_uncertainty
from hamilflow.uncertainty.band_postprocess import BandUncertaintyPostProcessor, UncertaintyCalibrator
from hamilflow.uncertainty.band_uncertainty import BandUncertaintyCalculator

NAME = "tMoS2-001"
MESH = (2, 2, 1)
N_MODELS = 3
SCALE = 0.02


def make_ensemble(structure, root, deltas):
    """Model folders holding a copy of `structure` with H(R) + delta(R) for each of `deltas`, and those H(R)."""
    model_dirs, members = [], []
    for i, delta in enumerate(deltas):
        folder = root / f"model_{i}" / NAME
        folder.mkdir(parents=True)
        for path in structure.path.iterdir():
            if path.name != DEEPX_HAMILTONIAN_FILENAME:
                shutil.copy(path, folder / path.name)
        members.append(structure.H_R + delta)
        structure.write_hamiltonian(members[-1], folder / DEEPX_HAMILTONIAN_FILENAME)
        model_dirs.append(folder.parent)
    return model_dirs, members


def random_deltas(structure, n=N_MODELS, scale=SCALE):
    return [scale * structure.perturbation(seed=i) for i in range(n)]


def eigvals_at(structure, H_R, k):
    phase = np.exp(2j * np.pi * (structure.Rs @ np.asarray(k)))
    Sk, Hk = np.tensordot(phase, structure.S_R, axes=1), np.tensordot(phase, H_R, axes=1)
    return scipy.linalg.eigh(Hk, Sk, eigvals_only=True)


def spectra(structure, H_R, kpoints):
    """(n_bands, n_k) eigenvalues at the k-points of an output."""
    return np.stack([eigvals_at(structure, H_R, e["k_frac"]) for e in kpoints.values()], axis=1)


def rigid_shift(structure, H_R, ref_R):
    return np.sum(np.real(H_R - ref_R) * structure.S_R) / np.sum(structure.S_R**2)


def tol(structure):
    return 1e-10 * max(1.0, np.abs(structure.H_R).max())


def per_k(result, key):
    return np.stack([np.array(e[key]) for e in result["kpoints"].values()], axis=1)


def n_occupied(structure):
    return 6 if structure.spinful else 3


@pytest.fixture
def ensemble(synthetic_structure, tmp_path):
    return make_ensemble(synthetic_structure, tmp_path, random_deltas(synthetic_structure))


@pytest.fixture
def small_structure(make_synthetic_structure):
    return make_synthetic_structure("input")


def compute(model_dirs, tmp_path, **options):
    calc = BandUncertaintyCalculator(grid_mesh=MESH, **options)
    return calc.compute(model_dirs, tmp_path / "average", n_jobs=1)


# ---------------------------------- exact method ----------------------------------


def test_uncertainty_with_matrix_alignment_matches_the_dense_reference(synthetic_structure, ensemble, tmp_path):
    model_dirs, members = ensemble
    result = compute(model_dirs, tmp_path)[NAME]
    H_avg = np.mean(members, axis=0)
    avg = spectra(synthetic_structure, H_avg, result["kpoints"])
    aligned = np.stack(
        [
            spectra(synthetic_structure, H, result["kpoints"]) - rigid_shift(synthetic_structure, H, H_avg)
            for H in members
        ]
    )
    atol = tol(synthetic_structure)
    np.testing.assert_allclose(per_k(result, "sigma_eV"), np.std(aligned, axis=0, ddof=1), atol=atol)
    np.testing.assert_allclose(
        per_k(result, "sigma_eV_avg_ham"), np.sqrt(np.mean((aligned - avg) ** 2, axis=0)), atol=atol
    )


def test_stored_eigenvalues_are_those_of_the_average_with_midgap_at_zero(synthetic_structure, ensemble, tmp_path):
    model_dirs, members = ensemble
    result = compute(model_dirs, tmp_path)[NAME]
    avg = spectra(synthetic_structure, np.mean(members, axis=0), result["kpoints"])
    n_occ = n_occupied(synthetic_structure)
    vbm, cbm = avg[n_occ - 1].max(), avg[n_occ].min()
    atol = tol(synthetic_structure)
    np.testing.assert_allclose(per_k(result, "eigvals_avg_ham_eV"), avg - (vbm + cbm) / 2, atol=atol)
    assert result["avg_hamiltonian_shift_eV"] == pytest.approx(-(vbm + cbm) / 2, abs=atol)
    assert result["avg_hamiltonian_gap_eV"] == pytest.approx(cbm - vbm, abs=atol)
    assert result["vbm_index"] == n_occ - 1


def test_result_records_its_settings_and_shape(synthetic_structure, ensemble, tmp_path):
    model_dirs, _ = ensemble
    result = compute(model_dirs, tmp_path)[NAME]
    assert result["align_mode"] == "matrix"
    assert result["ill_method"] is None
    assert result["ill_threshold"] is None
    assert result["grid_mesh"] == list(MESH)
    assert result["n_bands"] == (12 if synthetic_structure.spinful else 6)
    assert result["occupation"] == 6
    assert result["n_kept_orbitals"] is None
    assert result["n_irreducible_kpoints"] == len(result["kpoints"])
    assert sum(e["weight"] for e in result["kpoints"].values()) == np.prod(MESH)
    assert len(result["per_model_shift_eV"]) == N_MODELS
    assert "method" not in result


@pytest.mark.parametrize("method", ["exact", "projected"])
def test_each_member_shift_is_the_common_shift_minus_its_rigid_shift(small_structure, tmp_path, method):
    model_dirs, _ = make_ensemble(small_structure, tmp_path, random_deltas(small_structure))
    result = compute(model_dirs, tmp_path, method=method)[NAME]
    expected = result["avg_hamiltonian_shift_eV"] - np.array(result["per_model_rigid_shift_eV"])
    assert np.abs(np.array(result["per_model_rigid_shift_eV"])).max() > 0
    np.testing.assert_allclose(result["per_model_shift_eV"], expected)


def test_identical_members_have_no_uncertainty(synthetic_structure, tmp_path):
    zero = np.zeros_like(synthetic_structure.H_R)
    model_dirs, _ = make_ensemble(synthetic_structure, tmp_path, [zero, zero, zero])
    result = compute(model_dirs, tmp_path)[NAME]
    atol = tol(synthetic_structure)
    np.testing.assert_allclose(per_k(result, "sigma_eV"), 0.0, atol=atol)
    np.testing.assert_allclose(per_k(result, "sigma_eV_avg_ham"), 0.0, atol=atol)
    np.testing.assert_allclose(result["per_model_rigid_shift_eV"], 0.0, atol=atol)


def test_members_differing_by_a_rigid_shift_have_no_uncertainty(synthetic_structure, tmp_path):
    """The whole point of the alignment: a member's energy offset is not an uncertainty."""
    shifts = [-1.0, 0.5, 0.5]
    deltas = [c * synthetic_structure.S_R for c in shifts]
    model_dirs, _ = make_ensemble(synthetic_structure, tmp_path, deltas)
    result = compute(model_dirs, tmp_path)[NAME]
    atol = 1e-6 * max(1.0, tol(synthetic_structure) * 1e10)
    np.testing.assert_allclose(per_k(result, "sigma_eV"), 0.0, atol=atol)
    np.testing.assert_allclose(result["per_model_rigid_shift_eV"], shifts, atol=atol)
    assert max(result["per_model_rigid_shift_residual"]) < 1e-6


def test_uncertainty_is_not_changed_by_adding_a_rigid_shift_to_a_member(small_structure, tmp_path):
    deltas = random_deltas(small_structure)
    reference = compute(make_ensemble(small_structure, tmp_path / "a", deltas)[0], tmp_path / "a")[NAME]
    # adding c S to one member moves the mean by c S / 3, and every member's fitted shift with it
    deltas[0] = deltas[0] + 2.0 * small_structure.S_R
    shifted = compute(make_ensemble(small_structure, tmp_path / "b", deltas)[0], tmp_path / "b")[NAME]
    np.testing.assert_allclose(per_k(shifted, "sigma_eV"), per_k(reference, "sigma_eV"), atol=1e-8)


def test_doubling_a_small_perturbation_about_doubles_the_uncertainty(small_structure, tmp_path):
    small = compute(
        make_ensemble(small_structure, tmp_path / "a", random_deltas(small_structure, scale=1e-4))[0], tmp_path / "a"
    )
    large = compute(
        make_ensemble(small_structure, tmp_path / "b", random_deltas(small_structure, scale=2e-4))[0], tmp_path / "b"
    )
    ratio = per_k(large[NAME], "sigma_eV") / per_k(small[NAME], "sigma_eV")
    np.testing.assert_allclose(ratio, 2.0, atol=0.05)


def test_gaps_are_those_of_each_member(small_structure, tmp_path):
    model_dirs, members = make_ensemble(small_structure, tmp_path, random_deltas(small_structure))
    result = compute(model_dirs, tmp_path)[NAME]
    for gap, H in zip(result["per_model_gap_eV"], members, strict=True):
        bands = spectra(small_structure, H, result["kpoints"])
        assert gap == pytest.approx(bands[3].min() - bands[2].max(), abs=1e-10)


def test_grid_alignment_shifts_each_member_by_its_own_midgap(small_structure, tmp_path):
    model_dirs, members = make_ensemble(small_structure, tmp_path, random_deltas(small_structure))
    result = compute(model_dirs, tmp_path, align_mode="grid")[NAME]
    aligned = []
    for H in members:
        bands = spectra(small_structure, H, result["kpoints"])
        aligned.append(bands - (bands[2].max() + bands[3].min()) / 2)
    np.testing.assert_allclose(per_k(result, "sigma_eV"), np.std(aligned, axis=0, ddof=1), atol=1e-10)
    assert "per_model_rigid_shift_eV" not in result


def test_gamma_alignment_uses_the_anchor_k_point_only(small_structure, tmp_path):
    model_dirs, members = make_ensemble(small_structure, tmp_path, random_deltas(small_structure))
    result = compute(model_dirs, tmp_path, align_mode="gamma")[NAME]
    gamma = next(i for i, e in enumerate(result["kpoints"].values()) if np.allclose(e["k_frac"], 0.0))
    aligned = []
    for H in members:
        bands = spectra(small_structure, H, result["kpoints"])
        aligned.append(bands - (bands[2, gamma] + bands[3, gamma]) / 2)
    np.testing.assert_allclose(per_k(result, "sigma_eV"), np.std(aligned, axis=0, ddof=1), atol=1e-10)


def test_aligned_hamiltonians_are_written_on_request(small_structure, tmp_path):
    model_dirs, _ = make_ensemble(small_structure, tmp_path, random_deltas(small_structure))
    aligned_dir = tmp_path / "aligned"
    result = compute(model_dirs, tmp_path, aligned_hamiltonian_dir=aligned_dir)[NAME]
    for i, c in enumerate(result["per_model_rigid_shift_eV"], start=1):
        with h5py.File(aligned_dir / f"model_{i}" / NAME / DEEPX_HAMILTONIAN_FILENAME, "r") as f:
            assert f.attrs["rigid_shift_eV"] == pytest.approx(c)
            assert f.attrs["aligned_to"] == "ensemble average"


def test_aligned_hamiltonians_are_off_by_default(small_structure, tmp_path):
    model_dirs, _ = make_ensemble(small_structure, tmp_path, random_deltas(small_structure))
    compute(model_dirs, tmp_path)
    assert not list(tmp_path.glob("aligned*"))


def test_average_is_written_and_then_reused_as_is(small_structure, tmp_path):
    model_dirs, members = make_ensemble(small_structure, tmp_path, random_deltas(small_structure))
    avg_path = tmp_path / "average" / NAME / DEEPX_HAMILTONIAN_FILENAME
    first = compute(model_dirs, tmp_path)[NAME]
    assert avg_path.exists()

    small_structure.write_hamiltonian(np.mean(members, axis=0) + 1.0 * small_structure.S_R, avg_path)
    second = compute(model_dirs, tmp_path)[NAME]
    # the stored average moved by S: each member's fitted shift moves with it
    np.testing.assert_allclose(
        second["per_model_rigid_shift_eV"], np.array(first["per_model_rigid_shift_eV"]) - 1.0, atol=1e-8
    )


# ---------------------------------- orbital removal ----------------------------------


def ill_threshold_for(structure):
    """Just above the lowest S(k) eigenvalue of the mesh, so that removal really happens."""
    ks = [(a / 2, b / 2, 0.0) for a in range(2) for b in range(2)]
    phases = [np.exp(2j * np.pi * (structure.Rs @ np.array(k))) for k in ks]
    return min(np.linalg.eigvalsh(np.tensordot(p, structure.S_R, axes=1))[0] for p in phases) + 0.01


def test_orbital_removal_computes_all_spectra_in_the_same_reduced_basis(small_structure, tmp_path):
    model_dirs, _ = make_ensemble(small_structure, tmp_path, random_deltas(small_structure))
    threshold = ill_threshold_for(small_structure)
    result = compute(model_dirs, tmp_path, ill_method="orbital_removal", ill_threshold=threshold)[NAME]
    n_kept = result["n_kept_orbitals"]
    assert 3 < n_kept < 6
    assert result["n_bands"] == n_kept
    assert result["ill_threshold"] == threshold
    assert np.array(result["kpoints"]["k0"]["eigvals_avg_ham_eV"]).max() < 1000  # no fill value left


def test_ill_method_none_given_as_text_means_no_treatment():
    assert BandUncertaintyCalculator(ill_method="none").ill_method is None


def test_common_bands_cuts_to_the_bands_real_in_all_spectra():
    calc = BandUncertaintyCalculator()
    fill = calc._ILL_FILL_VALUE
    a = np.array([[1.0, 2.0], [3.0, 4.0], [fill, fill]])
    b = np.array([[1.0, 2.0], [fill, 4.0], [fill, fill]])
    cut_a, cut_b = calc.common_bands(a, b)
    np.testing.assert_array_equal(cut_a, a[:1])
    np.testing.assert_array_equal(cut_b, b[:1])


# ---------------------------------- projected method ----------------------------------


@pytest.mark.parametrize("downfold", [True, False])
def test_projected_with_the_whole_spectrum_reproduces_exact(small_structure, tmp_path, downfold):
    model_dirs, _ = make_ensemble(small_structure, tmp_path, random_deltas(small_structure))
    exact = compute(model_dirs, tmp_path)[NAME]
    projected = compute(model_dirs, tmp_path, method="projected", downfold=downfold)[NAME]
    for key in ("sigma_eV", "sigma_eV_avg_ham", "eigvals_avg_ham_eV"):
        np.testing.assert_allclose(per_k(projected, key), per_k(exact, key), atol=1e-9)
    np.testing.assert_allclose(projected["per_model_gap_eV"], exact["per_model_gap_eV"], atol=1e-9)


def test_projected_output_is_restricted_to_the_window(small_structure, tmp_path):
    model_dirs, _ = make_ensemble(small_structure, tmp_path, random_deltas(small_structure))
    exact = compute(model_dirs, tmp_path)[NAME]
    result = compute(model_dirs, tmp_path, method="projected", n_bands_below=2, n_bands_above=1)[NAME]
    assert (result["first_band_index"], result["n_bands"], result["n_bands_total"]) == (1, 3, 6)
    assert result["vbm_index"] == 1
    assert result["method"] == "projected"
    assert (result["n_bands_below"], result["n_bands_above"], result["downfold"]) == (2, 1, True)
    np.testing.assert_allclose(per_k(result, "eigvals_avg_ham_eV"), per_k(exact, "eigvals_avg_ham_eV")[1:4], atol=1e-9)
    assert result["avg_hamiltonian_gap_eV"] == pytest.approx(exact["avg_hamiltonian_gap_eV"], abs=1e-9)


def test_downfolding_brings_a_window_closer_to_exact(small_structure, tmp_path):
    model_dirs, _ = make_ensemble(small_structure, tmp_path, random_deltas(small_structure, scale=1e-3))
    exact = compute(model_dirs, tmp_path)[NAME]
    window = {"method": "projected", "n_bands_below": 1, "n_bands_above": 1}
    with_downfold = compute(model_dirs, tmp_path, downfold=True, **window)[NAME]
    without = compute(model_dirs, tmp_path, downfold=False, **window)[NAME]
    reference = per_k(exact, "sigma_eV")[2:4]
    error_with = np.abs(per_k(with_downfold, "sigma_eV") - reference).max()
    error_without = np.abs(per_k(without, "sigma_eV") - reference).max()
    assert error_with < 0.2 * error_without


@pytest.mark.parametrize("max_fill", [0.0, 1.0], ids=["dense-product", "sparse-product"])
def test_sparse_and_dense_products_agree(small_structure, tmp_path, monkeypatch, max_fill):
    model_dirs, _ = make_ensemble(small_structure, tmp_path, random_deltas(small_structure))
    exact = compute(model_dirs, tmp_path)[NAME]
    monkeypatch.setattr(band_uncertainty, "SPARSE_PRODUCT_MAX_FILL", max_fill)
    projected = compute(model_dirs, tmp_path, method="projected")[NAME]
    np.testing.assert_allclose(per_k(projected, "sigma_eV"), per_k(exact, "sigma_eV"), atol=1e-9)


def test_projected_with_orbital_removal_reproduces_exact(small_structure, tmp_path):
    model_dirs, _ = make_ensemble(small_structure, tmp_path, random_deltas(small_structure))
    options = {"ill_method": "orbital_removal", "ill_threshold": ill_threshold_for(small_structure)}
    exact = compute(model_dirs, tmp_path, **options)[NAME]
    projected = compute(model_dirs, tmp_path, method="projected", **options)[NAME]
    assert projected["n_bands"] == exact["n_bands"] == exact["n_kept_orbitals"]
    np.testing.assert_allclose(per_k(projected, "sigma_eV"), per_k(exact, "sigma_eV"), atol=1e-9)


def test_downfold_correction_is_the_second_order_formula_by_hand():
    e_w = np.array([0.0, 1.0])
    e_o = np.array([3.0])
    d_wo = np.array([[2.0], [1.0j]])
    sigma = BandUncertaintyCalculator._downfold_correction(e_w, d_wo, e_o)
    # sigma_ij = D_io conj(D_jo) (1/(E_i - E_o) + 1/(E_j - E_o)) / 2
    assert sigma[0, 0] == pytest.approx(4 * (-1 / 3))
    assert sigma[1, 1] == pytest.approx(1 * (-1 / 2))
    assert sigma[0, 1] == pytest.approx(2 * (-1j) * 0.5 * (-1 / 3 - 1 / 2))
    np.testing.assert_allclose(sigma, sigma.conj().T)


def test_downfold_correction_of_no_outside_bands_is_zero():
    sigma = BandUncertaintyCalculator._downfold_correction(np.array([0.0, 1.0]), np.zeros((2, 0)), np.zeros(0))
    np.testing.assert_array_equal(sigma, np.zeros((2, 2)))


def test_downfold_correction_matches_exact_diagonalization_for_a_small_coupling():
    e_w, e_o = np.array([0.0, 1.0]), np.array([5.0, 7.0])
    d_wo = 1e-3 * np.array([[1.0, 2.0], [-1.0, 0.5]])
    block = np.block([[np.diag(e_w), d_wo], [d_wo.T, np.diag(e_o)]])
    exact = np.linalg.eigvalsh(block)[:2]
    sigma = BandUncertaintyCalculator._downfold_correction(e_w, d_wo, e_o)
    np.testing.assert_allclose(np.linalg.eigvalsh(np.diag(e_w) + sigma), exact, atol=1e-10)


def test_window_does_not_split_a_degenerate_group_of_bands(small_structure, tmp_path, monkeypatch):
    """Band 5 is degenerate with band 4 at the edge of a window ending at band 4: the correction would blow up."""
    S_R = np.where(np.all(small_structure.Rs == 0, axis=1)[:, None, None], np.eye(6), 0.0)
    H_R = np.where(
        np.all(small_structure.Rs == 0, axis=1)[:, None, None], np.diag([-3.0, -2.0, -1.0, 1.0, 2.0, 2.0 + 1e-13]), 0.0
    )
    structure = dataclasses.replace(small_structure, S_R=S_R, H_R=H_R)
    # opposite perturbations: the average keeps the degeneracy
    delta = 0.01 * small_structure.perturbation()
    model_dirs, _ = make_ensemble(structure, tmp_path, [delta, -delta])
    for model in model_dirs:
        structure.write_hamiltonian(S_R, model / NAME / DEEPX_OVERLAP_FILENAME)
    window = {"method": "projected", "n_bands_below": 1, "n_bands_above": 2}

    exact = compute(model_dirs, tmp_path)[NAME]
    result = compute(model_dirs, tmp_path, **window)[NAME]
    assert (result["first_band_index"], result["n_bands"]) == (2, 4)
    assert per_k(result, "sigma_eV_avg_ham").max() < 1.0
    np.testing.assert_allclose(per_k(result, "sigma_eV_avg_ham"), per_k(exact, "sigma_eV_avg_ham")[2:], atol=1e-3)

    # the same run without the protection: what this test guards against
    monkeypatch.setattr(band_uncertainty, "WINDOW_EDGE_MIN_GAP", 0.0)
    unprotected = compute(model_dirs, tmp_path, **window)[NAME]
    assert unprotected["n_bands"] == 3
    # sigma_eV would not show it: the correction goes with D^2, the same for both members, so their spread cancels
    assert per_k(unprotected, "sigma_eV_avg_ham").max() > 1e3


def fake_structure(occupation=6, spinful=False):
    return SimpleNamespace(occupation=occupation, spinful=spinful, info_dir_path="here")


# bands in the columns of two k-points; homo is band 2
WINDOW_BANDS = np.array(
    [
        [-5.0, -5.0],
        [-3.0, -2.0],
        [-1.0, -1.5],
        [1.0, 1.5],
        [2.0, 4.0],
        [6.0, 7.0],
    ]
)


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, (0, 6)),
        ({"n_bands_below": 1}, (2, 6)),
        ({"n_bands_above": 1}, (0, 4)),
        ({"n_bands_below": 2, "n_bands_above": 2}, (1, 5)),
        ({"n_bands_below": 99, "n_bands_above": 99}, (0, 6)),
        ({"window_e": 0.0}, (2, 4)),
        # VBM -1, CBM 1: band 1 reaches -2 at the second k-point, band 4 reaches 2 at the first
        ({"window_e": 1.0}, (1, 5)),
        ({"window_e": 0.99}, (2, 4)),
        ({"window_e": 100.0}, (0, 6)),
    ],
)
def test_band_window(options, expected):
    calc = BandUncertaintyCalculator(**options)
    assert calc._band_window(WINDOW_BANDS, fake_structure()) == expected


@pytest.mark.parametrize(
    "options", [{"n_bands_below": 1, "n_bands_above": 1}, {"window_e": 0.001}], ids=["count", "energy"]
)
@pytest.mark.parametrize(
    ("gap", "expected"),
    [
        (0.0, (1, 5)),  # exactly degenerate
        (1e-13, (1, 5)),
        (0.009, (1, 5)),  # just inside the limit
        (0.011, (2, 4)),  # just outside
    ],
)
def test_window_grows_past_bands_too_close_to_its_edges(options, gap, expected):
    # at the second k-point only, band 1 comes within `gap` of band 2 and band 4 of band 3
    bands = np.array([[-5.0, -5.0], [-3.0, -1.0 - gap], [-1.0, -1.0], [1.0, 1.0], [2.0, 1.0 + gap], [6.0, 7.0]])
    assert BandUncertaintyCalculator(**options)._band_window(bands, fake_structure()) == expected


def test_window_grows_through_a_whole_group_of_close_bands():
    bands = np.array([[-1.002], [-1.001], [-1.0], [1.0], [1.001], [1.002], [5.0]])
    calc = BandUncertaintyCalculator(n_bands_below=1, n_bands_above=1)
    assert calc._band_window(bands, fake_structure()) == (0, 6)


def test_window_growth_stops_at_the_ends_of_the_spectrum():
    bands = np.array([[-1.0], [-1.0], [-1.0], [1.0], [1.0], [1.0]])
    assert BandUncertaintyCalculator(n_bands_below=1, n_bands_above=1)._band_window(bands, fake_structure()) == (0, 6)


# ---------------------------------- comparison to DFT ----------------------------------


def compare(model_dirs, dft_root, tmp_path, **options):
    calc = BandUncertaintyCalculator(grid_mesh=MESH, **options)
    return calc.compare_averaged_to_dft(model_dirs, tmp_path / "average", dft_root, n_jobs=1)


@pytest.fixture
def dft_root(synthetic_structure, tmp_path):
    root = tmp_path / "dft"
    root.mkdir()
    shutil.copytree(synthetic_structure.path, root / NAME)
    return root


def test_error_is_measured_after_the_rigid_alignment(synthetic_structure, ensemble, dft_root, tmp_path):
    model_dirs, members = ensemble
    result = compare(model_dirs, dft_root, tmp_path)[NAME]
    H_avg = np.mean(members, axis=0)
    c = rigid_shift(synthetic_structure, H_avg, synthetic_structure.H_R)
    avg = spectra(synthetic_structure, H_avg, result["kpoints"])
    dft = spectra(synthetic_structure, synthetic_structure.H_R, result["kpoints"])
    atol = tol(synthetic_structure)
    np.testing.assert_allclose(per_k(result, "abs_err_eV"), np.abs(avg - c - dft), atol=atol)
    assert result["avg_vs_dft_rigid_shift_eV"] == pytest.approx(c, abs=atol)


def test_average_that_is_dft_plus_a_rigid_shift_has_no_error(synthetic_structure, dft_root, tmp_path):
    deltas = [c * synthetic_structure.S_R for c in (1.0, 2.0, 3.0)]
    model_dirs, _ = make_ensemble(synthetic_structure, tmp_path, deltas)
    result = compare(model_dirs, dft_root, tmp_path)[NAME]
    atol = 1e-6 * max(1.0, tol(synthetic_structure) * 1e10)
    np.testing.assert_allclose(per_k(result, "abs_err_eV"), 0.0, atol=atol)
    assert result["avg_vs_dft_rigid_shift_eV"] == pytest.approx(2.0, abs=atol)
    assert result["overall_mae_eV"] == pytest.approx(0.0, abs=atol)


def test_gaps_of_the_average_and_of_dft_are_reported(small_structure, tmp_path):
    root = tmp_path / "dft"
    root.mkdir()
    shutil.copytree(small_structure.path, root / NAME)
    model_dirs, members = make_ensemble(small_structure, tmp_path, random_deltas(small_structure))
    result = compare(model_dirs, root, tmp_path)[NAME]
    avg = spectra(small_structure, np.mean(members, axis=0), result["kpoints"])
    dft = spectra(small_structure, small_structure.H_R, result["kpoints"])
    assert result["gap_avg_ham_eV"] == pytest.approx(avg[3].min() - avg[2].max(), abs=1e-10)
    assert result["gap_dft_eV"] == pytest.approx(dft[3].min() - dft[2].max(), abs=1e-10)


def test_overall_mae_of_one_k_point_is_its_mean_error(small_structure, tmp_path):
    root = tmp_path / "dft"
    root.mkdir()
    shutil.copytree(small_structure.path, root / NAME)
    model_dirs, _ = make_ensemble(small_structure, tmp_path, random_deltas(small_structure))
    calc = BandUncertaintyCalculator(grid_mesh=(1, 1, 1))
    result = calc.compare_averaged_to_dft(model_dirs, tmp_path / "average", root, n_jobs=1)[NAME]
    assert result["overall_mae_eV"] == pytest.approx(np.mean(result["kpoints"]["k0"]["abs_err_eV"]))


def test_overall_mae_weights_each_k_point_by_its_symmetry_weight(small_structure, tmp_path, monkeypatch):
    root = tmp_path / "dft"
    root.mkdir()
    shutil.copytree(small_structure.path, root / NAME)
    model_dirs, _ = make_ensemble(small_structure, tmp_path, random_deltas(small_structure))
    original = BandUncertaintyCalculator.build_irreducible_kpoints

    def unequal_weights(self, h_obj, mesh, symprec):
        ks, _, anchor = original(self, h_obj, mesh, symprec)
        return ks, np.arange(1, len(ks) + 1), anchor

    monkeypatch.setattr(BandUncertaintyCalculator, "build_irreducible_kpoints", unequal_weights)
    result = compare(model_dirs, root, tmp_path)[NAME]
    means = [np.mean(e["abs_err_eV"]) for e in result["kpoints"].values()]
    weights = [e["weight"] for e in result["kpoints"].values()]
    assert len(set(weights)) > 1
    assert not np.isclose(np.mean(means), np.average(means, weights=weights))
    assert result["overall_mae_eV"] == pytest.approx(np.average(means, weights=weights))


def test_grid_alignment_of_the_comparison_uses_each_own_midgap(small_structure, tmp_path):
    root = tmp_path / "dft"
    root.mkdir()
    shutil.copytree(small_structure.path, root / NAME)
    model_dirs, members = make_ensemble(small_structure, tmp_path, random_deltas(small_structure))
    result = compare(model_dirs, root, tmp_path, align_mode="grid")[NAME]
    avg = spectra(small_structure, np.mean(members, axis=0), result["kpoints"])
    dft = spectra(small_structure, small_structure.H_R, result["kpoints"])
    expected = np.abs((avg - (avg[2].max() + avg[3].min()) / 2) - (dft - (dft[2].max() + dft[3].min()) / 2))
    np.testing.assert_allclose(per_k(result, "abs_err_eV"), expected, atol=1e-10)
    assert "avg_vs_dft_rigid_shift_eV" not in result


@pytest.mark.integration
def test_outputs_feed_the_calibrator(small_structure, tmp_path):
    """The two kinds of output must stay consistent with what the postprocessing reads."""
    root = tmp_path / "dft"
    root.mkdir()
    shutil.copytree(small_structure.path, root / NAME)
    model_dirs, _ = make_ensemble(small_structure, tmp_path, random_deltas(small_structure))
    calc = BandUncertaintyCalculator(grid_mesh=MESH)
    u_path, e_path = tmp_path / "u.json", tmp_path / "e.json"
    calc.compute(model_dirs, tmp_path / "average", n_jobs=1, output_path=u_path)
    calc.compare_averaged_to_dft(model_dirs, tmp_path / "average", root, n_jobs=1, output_path=e_path)

    cal = UncertaintyCalibrator.from_json(u_path, e_path, e_cut=1.0, tau=0.1)
    names, u, e = cal.paired()
    assert names == [NAME]
    assert u[0] > 0
    assert e[0] > 0
    stats = BandUncertaintyPostProcessor.from_json(u_path).weighted_stats(NAME)
    assert stats["gap_eV"] == pytest.approx(json.loads(u_path.read_text())[NAME]["avg_hamiltonian_gap_eV"])


# ---------------------------------- k-points and edges ----------------------------------


def cubic(a=3.0, species=("Mo",)):
    return SimpleNamespace(lattice=a * np.eye(3), frac_coords=np.zeros((len(species), 3)), elements=list(species))


def test_irreducible_mesh_of_a_cubic_cell_by_hand():
    # the 8 points of a 2 x 2 x 2 mesh: Gamma, 3 X, 3 M, R
    calc = BandUncertaintyCalculator()
    ks, weights, anchor = calc.build_irreducible_kpoints(cubic(), (2, 2, 2), 1e-5)
    assert sorted(weights.tolist()) == [1, 1, 3, 3]
    assert weights.sum() == 8
    np.testing.assert_allclose(ks[anchor], 0.0)
    assert ks.shape == (4, 3)


def test_irreducible_points_represent_the_full_mesh_for_a_symmetric_function():
    """A cubic-invariant function averaged with the weights must equal its average over the full mesh."""
    calc = BandUncertaintyCalculator()
    ks, weights, _ = calc.build_irreducible_kpoints(cubic(), (4, 4, 4), 1e-5)
    f = lambda k: np.cos(2 * np.pi * k).sum(axis=-1) ** 2
    grid = np.stack(np.meshgrid(*[np.arange(4) / 4] * 3, indexing="ij"), axis=-1)
    assert np.average(f(ks), weights=weights) == pytest.approx(f(grid).mean())


def test_anchor_k_point_must_be_on_the_mesh():
    calc = BandUncertaintyCalculator(anchor_k=(0.3, 0.0, 0.0))
    with pytest.raises(AssertionError):
        calc.build_irreducible_kpoints(cubic(), (2, 2, 2), 1e-5)


def test_anchor_k_point_is_found_on_the_mesh():
    calc = BandUncertaintyCalculator(anchor_k=(0.5, 0.0, 0.0))
    ks, _, anchor = calc.build_irreducible_kpoints(
        SimpleNamespace(lattice=np.diag([3.0, 4.0, 5.0]), frac_coords=np.zeros((1, 3)), elements=["Mo"]),
        (2, 2, 2),
        1e-5,
    )
    np.testing.assert_allclose(ks[anchor], [0.5, 0.0, 0.0])


def test_species_can_be_tagged_for_unknown_elements():
    with pytest.raises(ValueError, match="No species_number"):
        BandUncertaintyCalculator().build_irreducible_kpoints(cubic(species=("Xx",)), (1, 1, 1), 1e-5)
    calc = BandUncertaintyCalculator(species_number={"Xx": 42})
    assert len(calc.build_irreducible_kpoints(cubic(species=("Xx",)), (1, 1, 1), 1e-5)[0]) == 1


def test_mesh_comes_from_the_real_sampling_function(small_structure):
    from hamilflow.sparse_hamiltonian import SparseHamiltonianObj

    obj = SparseHamiltonianObj(small_structure.path)
    assert BandUncertaintyCalculator(grid_mesh=[3, 3, 1]).resolve_mesh(obj) == (3, 3, 1)
    assert BandUncertaintyCalculator(kpoints_settings={"k_grid": [2, 3, 5]}).resolve_mesh(obj) == (2, 3, 5)
    calc = BandUncertaintyCalculator(kpoints_settings={"k_grid": [2, 3, 5]}, kpoints_force_2d=True)
    assert calc.resolve_mesh(obj) == (2, 3, 1)
    calc = BandUncertaintyCalculator(kpoints_settings={"k_grid": [2, 3, 5]}, kpoints_force_1d=True)
    assert calc.resolve_mesh(obj) == (1, 1, 5)
    assert all(n >= 1 for n in BandUncertaintyCalculator(kpoints_settings={"grid_density": 100}).resolve_mesh(obj))


def test_homo_and_lumo_count_two_electrons_per_band_when_spinless():
    calc = BandUncertaintyCalculator()
    assert calc.homo_lumo_indices(fake_structure(1036)) == (517, 518)
    assert calc.homo_lumo_indices(fake_structure(6, spinful=True)) == (5, 6)
    assert calc.homo_lumo_indices(fake_structure(7, spinful=True)) == (6, 7)


def test_missing_or_odd_occupation_is_rejected():
    calc = BandUncertaintyCalculator()
    with pytest.raises(ValueError, match="not set"):
        calc.homo_lumo_indices(fake_structure(None))
    with pytest.raises(ValueError, match="Odd electron count"):
        calc.homo_lumo_indices(fake_structure(7))


def test_band_edges_and_gap_take_each_edge_over_the_grid():
    calc = BandUncertaintyCalculator()
    assert calc.band_edges(WINDOW_BANDS, fake_structure()) == (-1.0, 1.0)
    assert calc.gap(WINDOW_BANDS, fake_structure()) == 2.0
    aligned, shift = calc.align_to_midgap(WINDOW_BANDS + 3.0, fake_structure())
    assert shift == -3.0
    np.testing.assert_allclose(aligned, WINDOW_BANDS)


def test_gamma_band_edges_use_the_anchor_but_the_gap_stays_on_the_grid():
    calc = BandUncertaintyCalculator(align_mode="gamma")
    assert calc.band_edges(WINDOW_BANDS, fake_structure(), anchor_k_idx=1) == (-1.5, 1.5)
    assert calc.gap(WINDOW_BANDS, fake_structure()) == 2.0


# ---------------------------------- settings ----------------------------------


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"align_mode": "other"}, "align_mode"),
        ({"ill_method": "window_regularization"}, "ill_method"),
        ({"method": "other"}, "method must be"),
        ({"method": "projected", "align_mode": "grid"}, "needs align_mode='matrix'"),
        ({"window_e": 1.0, "n_bands_below": 2}, "not both"),
        ({"window_e": 1.0, "n_bands_above": 2}, "not both"),
        ({"window_e": -1.0}, "window_e must be"),
    ],
)
def test_invalid_settings_are_rejected(options, message):
    with pytest.raises(ValueError, match=message):
        BandUncertaintyCalculator(**options)


def test_settings_of_the_exact_method_stay_minimal():
    settings = BandUncertaintyCalculator().settings()
    assert settings == {"align_mode": "matrix", "ill_method": None, "ill_threshold": None}


def test_settings_of_the_projected_method_include_the_window():
    settings = BandUncertaintyCalculator(method="projected", window_e=2.5, downfold=False).settings()
    assert settings["window_e"] == 2.5
    assert settings["downfold"] is False
    assert settings["method"] == "projected"


# ---------------------------------- running over structures ----------------------------------


def test_structures_are_listed_with_pattern_and_exclusions(tmp_path):
    for name in ("str1", "str2", "other"):
        (tmp_path / name).mkdir()
    (tmp_path / "a_file").write_text("")
    calc = BandUncertaintyCalculator
    assert sorted(calc._list_structures(tmp_path)) == ["other", "str1", "str2"]
    assert sorted(calc._list_structures(tmp_path, "str*")) == ["str1", "str2"]
    assert sorted(calc._list_structures(tmp_path, exclude_structures=["other"])) == ["str1", "str2"]
    assert calc._list_structures(tmp_path, "str*", ["str1"]) == ["str2"]


def test_no_structures_left_is_a_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        assert BandUncertaintyCalculator._list_structures(tmp_path, "nothing*") == []
    assert "No structures left" in caplog.text


def test_output_is_written_and_finished_structures_are_kept(small_structure, tmp_path):
    model_dirs, _ = make_ensemble(small_structure, tmp_path, random_deltas(small_structure))
    output = tmp_path / "sub" / "out.json"
    calc = BandUncertaintyCalculator(grid_mesh=MESH)
    first = calc.compute(model_dirs, tmp_path / "average", n_jobs=1, output_path=output)
    assert json.loads(output.read_text()) == json.loads(json.dumps(first))
    assert not output.with_suffix(".json.tmp").exists()

    marked = json.loads(output.read_text())
    marked[NAME]["marker"] = True
    output.write_text(json.dumps(marked))
    assert calc.compute(model_dirs, tmp_path / "average", n_jobs=1, output_path=output)[NAME]["marker"] is True


def test_not_resuming_recomputes_and_clears_the_stale_file(small_structure, tmp_path):
    model_dirs, _ = make_ensemble(small_structure, tmp_path, random_deltas(small_structure))
    output = tmp_path / "out.json"
    output.write_text(json.dumps({"stale": {"align_mode": "matrix"}, NAME: {"marker": True}}))
    calc = BandUncertaintyCalculator(grid_mesh=MESH)
    result = calc.compute(model_dirs, tmp_path / "average", n_jobs=1, output_path=output, skip_existing=False)
    assert set(result) == {NAME}
    assert "marker" not in result[NAME]
    assert set(json.loads(output.read_text())) == {NAME}


def test_not_resuming_clears_the_file_before_any_structure_finishes(small_structure, tmp_path):
    """An interrupted rerun must not leave the old results looking like new ones."""
    model_dirs, _ = make_ensemble(small_structure, tmp_path, random_deltas(small_structure))
    output = tmp_path / "out.json"
    output.write_text(json.dumps({NAME: {"marker": True}}))
    calc = BandUncertaintyCalculator(grid_mesh=MESH)
    calc.compute(model_dirs, tmp_path / "average", "nothing*", output_path=output, skip_existing=False)
    assert json.loads(output.read_text()) == {}


def test_results_with_other_settings_are_kept_with_a_warning(caplog):
    calc = BandUncertaintyCalculator()
    output = {"a": calc.settings(), "b": {**calc.settings(), "align_mode": "grid"}}
    with caplog.at_level(logging.WARNING):
        assert calc._drop_done(["a", "b", "c"], output) == ["c"]
    assert "1 stored structures were computed with settings other" in caplog.text


def test_interrupted_run_keeps_the_structures_already_finished(small_structure, tmp_path, monkeypatch):
    model_dirs, _ = make_ensemble(small_structure, tmp_path, random_deltas(small_structure))
    second = model_dirs[0] / "broken"
    second.mkdir()
    output = tmp_path / "out.json"
    calc = BandUncertaintyCalculator(grid_mesh=MESH)
    monkeypatch.setattr(BandUncertaintyCalculator, "_list_structures", staticmethod(lambda *a, **k: [NAME, "broken"]))
    with pytest.raises(FileNotFoundError):
        calc.compute(model_dirs, tmp_path / "average", n_jobs=1, output_path=output)
    assert set(json.loads(output.read_text())) == {NAME}


def test_kpoint_threads_do_not_change_the_result(small_structure, tmp_path):
    model_dirs, _ = make_ensemble(small_structure, tmp_path, random_deltas(small_structure))
    calc = BandUncertaintyCalculator(grid_mesh=MESH)
    serial = calc.compute(model_dirs, tmp_path / "average", n_jobs=1, parallel_k=False)[NAME]
    threaded = calc.compute(model_dirs, tmp_path / "average", n_jobs=2, parallel_k=True)[NAME]
    np.testing.assert_allclose(per_k(threaded, "sigma_eV"), per_k(serial, "sigma_eV"), atol=1e-10)


def test_missing_info_files_are_reported(small_structure, tmp_path):
    model_dirs, _ = make_ensemble(small_structure, tmp_path, random_deltas(small_structure))
    (model_dirs[0] / NAME / DEEPX_INFO_FILENAME).unlink()
    with pytest.raises(FileNotFoundError):
        compute(model_dirs, tmp_path)


def test_member_with_another_layout_is_rejected(small_structure, make_synthetic_structure, tmp_path):
    model_dirs, _ = make_ensemble(small_structure, tmp_path, random_deltas(small_structure))
    other = make_synthetic_structure("other", seed=5)
    shutil.copy(other.path / DEEPX_HAMILTONIAN_FILENAME, model_dirs[1] / NAME / DEEPX_HAMILTONIAN_FILENAME)
    shutil.copy(other.path / DEEPX_OVERLAP_FILENAME, model_dirs[1] / NAME / DEEPX_OVERLAP_FILENAME)
    with pytest.raises(ValueError, match="mismatch"):
        compute(model_dirs, tmp_path)
