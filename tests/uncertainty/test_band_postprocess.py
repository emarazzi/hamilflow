"""
Band uncertainty statistics and conformal calibration, with expected values worked out by hand.

The files mimic real `compute` and `compare_averaged_to_dft` outputs, where the stored bands are a window
of the spectrum and `vbm_index` is relative to it.
"""

import json

import numpy as np
import pytest

from hamilflow.uncertainty.band_postprocess import BandUncertaintyPostProcessor, UncertaintyCalibrator

SIGMA = "sigma_eV_avg_ham"
ERROR = "abs_err_eV"
SETTINGS = {"align_mode": "matrix", "ill_method": "orbital_removal", "ill_threshold": 0.001}

# Two k-points of weights 1 and 3, four bands, the VBM is band 1:
#   VBM = -0.5 (at k1), CBM = 0.5 (at k1), gap 1
#   edge distance: k0 [1.5, 0.5, 0.5, 1.5], k1 [2.5, 0, 0, 2.5]
EIGVALS = [[-2.0, -1.0, 1.0, 2.0], [-3.0, -0.5, 0.5, 3.0]]
VALUES = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
WEIGHTS = [1, 3]


def structure(values=VALUES, eigvals=EIGVALS, weights=WEIGHTS, key=SIGMA, vbm_index=1, per_k="kpoints", **top):
    entries = {
        f"k{i}": {"k_frac": [0.1 * i, 0.0, 0.0], "weight": w, key: v, "eigvals_avg_ham_eV": e}
        for i, (v, e, w) in enumerate(zip(values, eigvals, weights, strict=True))
    }
    result = {**SETTINGS, "occupation": 1116, per_k: entries, **top}
    if vbm_index is not None:
        result["vbm_index"] = vbm_index
    return result


@pytest.fixture
def pp():
    return BandUncertaintyPostProcessor({"a": structure()})


# ---------------------------------- reading ----------------------------------


def test_values_and_weights_follow_the_file(pp):
    values, weights = pp.values("a")
    np.testing.assert_array_equal(values, VALUES)
    np.testing.assert_array_equal(weights, WEIGHTS)


def test_from_json_reads_the_requested_key(tmp_path):
    path = tmp_path / "file.json"
    path.write_text(json.dumps({"a": structure(key=ERROR)}))
    values, _ = BandUncertaintyPostProcessor.from_json(path, key=ERROR).values("a")
    np.testing.assert_array_equal(values, VALUES)


def test_older_files_with_per_k_are_read():
    pp = BandUncertaintyPostProcessor({"a": structure(per_k="per_k")})
    np.testing.assert_array_equal(pp.values("a")[0], VALUES)


def test_vbm_index_of_the_file_is_relative_to_the_stored_window():
    """Real file: occupation 1116 with a window starting at band 365 stores vbm_index 192, not 557."""
    assert BandUncertaintyPostProcessor.vbm_index(structure(vbm_index=192)) == 192


def test_older_files_without_vbm_index_assume_two_electrons_per_band():
    assert BandUncertaintyPostProcessor.vbm_index(structure(vbm_index=None)) == 557


def test_wrong_key_says_so(pp):
    wrong = BandUncertaintyPostProcessor(pp.results, key=ERROR)
    with pytest.raises(KeyError, match="wrong `key`"):
        wrong.values("a")


def test_file_without_eigenvalues_says_to_recompute():
    result = structure()
    for entry in result["kpoints"].values():
        del entry["eigvals_avg_ham_eV"]
    with pytest.raises(KeyError, match="recompute"):
        BandUncertaintyPostProcessor({"a": result}).eigenvalues("a")


# ---------------------------------- band edges ----------------------------------


def test_band_edges_are_the_extrema_over_k(pp):
    assert pp.band_edges("a") == (-0.5, 0.5)


def test_indirect_gap_takes_each_edge_from_its_own_k_point():
    eigvals = [[-1.0, 2.0], [-0.5, 3.0]]
    pp = BandUncertaintyPostProcessor({"a": structure(values=[[0, 0], [0, 0]], eigvals=eigvals, vbm_index=0)})
    assert pp.band_edges("a") == (-0.5, 2.0)


def test_window_without_a_conduction_band_is_rejected():
    pp = BandUncertaintyPostProcessor({"a": structure(vbm_index=3)})
    with pytest.raises(ValueError, match="no conduction band"):
        pp.band_edges("a")


def test_edge_distance_measures_each_state_from_its_own_edge(pp):
    np.testing.assert_allclose(pp.edge_distance("a"), [[1.5, 0.5, 0.5, 1.5], [2.5, 0.0, 0.0, 2.5]])


# ---------------------------------- soft window ----------------------------------


def test_soft_window_is_one_at_the_edge():
    assert BandUncertaintyPostProcessor.soft_window(np.array(0.0), 1.0, 0.1) == pytest.approx(1.0)


def test_soft_window_is_the_rescaled_fermi_function():
    d, e_cut, tau = 0.7, 1.0, 0.2
    expected = (1 + np.exp(-e_cut / tau)) / (1 + np.exp((d - e_cut) / tau))
    assert BandUncertaintyPostProcessor.soft_window(np.array(d), e_cut, tau) == pytest.approx(expected)


def test_soft_window_small_tau_is_a_hard_window():
    w = BandUncertaintyPostProcessor.soft_window(np.array([0.0, 0.5, 0.99, 1.01, 2.0]), 1.0, 1e-4)
    np.testing.assert_allclose(w, [1, 1, 1, 0, 0], atol=1e-12)


def test_soft_window_does_not_overflow_far_from_the_edge():
    with np.errstate(all="raise"):
        assert BandUncertaintyPostProcessor.soft_window(np.array(1e6), 1.0, 1e-3) == 0.0


def test_soft_window_decreases():
    w = BandUncertaintyPostProcessor.soft_window(np.linspace(0, 3, 50), 1.0, 0.1)
    assert np.all(np.diff(w) < 0)


@pytest.mark.parametrize("tau", [0.0, -1.0])
def test_soft_window_rejects_non_positive_tau(tau):
    with pytest.raises(ValueError, match="`tau` must be"):
        BandUncertaintyPostProcessor.soft_window(np.array(0.0), 1.0, tau)


# ---------------------------------- band selection ----------------------------------


def test_all_bands_by_default(pp):
    np.testing.assert_array_equal(pp.band_indices("a"), [0, 1, 2, 3])


@pytest.mark.parametrize(
    ("bands", "expected"), [([0, 2], [0, 2]), ([-1], [3]), (slice(1, 3), [1, 2]), (slice(None, None, 2), [0, 2])]
)
def test_explicit_bands(pp, bands, expected):
    np.testing.assert_array_equal(pp.band_indices("a", bands=bands), expected)


@pytest.mark.parametrize(("fraction", "expected"), [(0.5, [1, 2]), (1.0, [0, 1, 2, 3]), (0.01, [1, 2])])
def test_vbm_fraction_takes_the_same_number_of_bands_on_each_side_of_the_gap(pp, fraction, expected):
    np.testing.assert_array_equal(pp.band_indices("a", vbm_fraction=fraction), expected)


def test_vbm_fraction_is_clipped_to_the_stored_bands():
    # 3 occupied, 1 conduction: fraction 1 asks for 3 per side
    values = [[0.0] * 4]
    pp = BandUncertaintyPostProcessor({"a": structure(values=values, eigvals=values, weights=[1], vbm_index=2)})
    np.testing.assert_array_equal(pp.band_indices("a", vbm_fraction=1.0), [0, 1, 2, 3])


def test_bands_and_vbm_fraction_are_exclusive(pp):
    with pytest.raises(ValueError, match="not both"):
        pp.band_indices("a", bands=[0], vbm_fraction=0.5)


@pytest.mark.parametrize("fraction", [0.0, -0.5, 1.5])
def test_vbm_fraction_out_of_range_is_rejected(pp, fraction):
    with pytest.raises(ValueError, match="`vbm_fraction` must be"):
        pp.band_indices("a", vbm_fraction=fraction)


# ---------------------------------- statistics ----------------------------------


def test_structure_stats_weight_each_k_point_by_its_symmetry_weight(pp):
    stats = pp.structure_stats("a")
    assert stats["mean_eV"] == pytest.approx((1 * 2.5 + 3 * 6.5) / 4)
    assert stats["max_eV"] == 8.0
    assert (stats["argmax_k"], stats["argmax_band"]) == (1, 3)
    assert (stats["n_valence"], stats["n_conduction"]) == (2, 2)


def test_structure_stats_report_the_band_index_of_the_full_list(pp):
    stats = pp.structure_stats("a", bands=[0, 1])
    assert stats["max_eV"] == 6.0
    assert stats["argmax_band"] == 1
    assert (stats["n_valence"], stats["n_conduction"]) == (2, 0)


def test_argmax_band_is_an_index_of_the_full_list_not_of_the_selection(pp):
    assert pp.structure_stats("a", bands=[2, 3])["argmax_band"] == 3
    assert pp.weighted_stats("a", e_cut=5.0, tau=1e-3, bands=[2, 3])["weighted_argmax_band"] == 3


def test_summary_covers_every_structure():
    pp = BandUncertaintyPostProcessor({"a": structure(), "b": structure()})
    assert set(pp.summary()) == {"a", "b"}


def test_hard_window_keeps_only_the_states_near_the_edges(pp):
    """With e_cut 1 and a tiny tau the window keeps bands 1 and 2 at both k-points, values 2, 3, 6, 7."""
    stats = pp.weighted_stats("a", e_cut=1.0, tau=1e-3)
    assert stats["weighted_mean_eV"] == pytest.approx((1 * (2 + 3) + 3 * (6 + 7)) / 8)
    assert stats["weighted_max_eV"] == pytest.approx(7.0)
    assert (stats["weighted_argmax_k"], stats["weighted_argmax_band"]) == (1, 2)
    assert stats["n_eff_bands"] == pytest.approx(2.0)


def test_weighted_stats_include_edges_and_unweighted_stats(pp):
    stats = pp.weighted_stats("a", e_cut=1.0, tau=1e-3)
    assert (stats["vbm_eV"], stats["cbm_eV"], stats["gap_eV"]) == (-0.5, 0.5, 1.0)
    assert stats["max_eV"] == 8.0


def test_large_value_far_from_the_edges_does_not_dominate_the_weighted_max():
    values = [[1.0, 2.0, 3.0, 1000.0], [5.0, 6.0, 7.0, 1000.0]]
    stats = BandUncertaintyPostProcessor({"a": structure(values=values)}).weighted_stats("a", e_cut=1.0, tau=1e-3)
    assert stats["max_eV"] == 1000.0
    assert stats["weighted_max_eV"] == pytest.approx(7.0)


def test_weighted_max_is_taken_after_weighting():
    # band 0 of k1 (d = 2.5) is weak at e_cut 2.5, tau 0.1: about half of its 8 stays below the 6 at the edge
    values = [[0.0, 0.0, 0.0, 0.0], [8.0, 6.0, 0.0, 0.0]]
    stats = BandUncertaintyPostProcessor({"a": structure(values=values)}).weighted_stats("a", e_cut=2.5, tau=0.1)
    assert stats["weighted_max_eV"] == pytest.approx(6.0)
    assert stats["weighted_argmax_band"] == 1


def test_bands_are_a_hard_selection_before_the_window(pp):
    stats = pp.weighted_stats("a", e_cut=1.0, tau=1e-3, bands=[1])
    assert stats["weighted_mean_eV"] == pytest.approx((1 * 2 + 3 * 6) / 4)
    assert stats["n_eff_bands"] == pytest.approx(1.0)


def test_quantile_is_only_reported_when_asked(pp):
    assert "weighted_quantile_eV" not in pp.weighted_stats("a")
    # states 2 (w 1), 3 (w 1), 6 (w 3), 7 (w 3): half of the total weight 8 is reached at 6
    assert pp.weighted_stats("a", e_cut=1.0, tau=1e-3, quantile=0.5)["weighted_quantile_eV"] == pytest.approx(6.0)


def test_weighted_summary_passes_its_options_on():
    pp = BandUncertaintyPostProcessor({"a": structure()})
    assert pp.weighted_summary(e_cut=1.0, tau=1e-3, bands=[1], quantile=1.0)["a"]["weighted_quantile_eV"] == 6.0


@pytest.mark.parametrize(
    ("q", "expected"),
    [(0.0, 1.0), (0.125, 1.0), (0.25, 2.0), (0.26, 3.0), (0.5, 3.0), (0.51, 4.0), (1.0, 4.0)],
)
def test_weighted_quantile_is_the_smallest_value_reaching_the_weight(q, expected):
    # cumulative weights 1, 2, 4, 8 of a total of 8
    x = np.array([4.0, 1.0, 3.0, 2.0])
    w = np.array([4.0, 1.0, 2.0, 1.0])
    assert BandUncertaintyPostProcessor.weighted_quantile(x, w, q) == expected


def test_weighted_quantile_of_one_ignores_states_of_zero_weight():
    x = np.array([1.0, 2.0, 100.0])
    assert BandUncertaintyPostProcessor.weighted_quantile(x, np.array([1.0, 1.0, 0.0]), 1.0) == 2.0


def test_weighted_quantile_is_not_skipped_by_rounding():
    x = np.arange(10.0)
    assert BandUncertaintyPostProcessor.weighted_quantile(x, np.full(10, 0.1), 0.3) == 2.0


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 7])
def test_weighted_quantile_is_not_pushed_up_a_state_by_rounding(m):
    """With nine equal weights, the cumulative sum at m/9 falls just short of m/9 of the rounded total."""
    x, w = np.arange(9.0), np.full(9, 1 / 9)
    assert BandUncertaintyPostProcessor.weighted_quantile(x, w, m / 9) == m - 1


def test_weighted_quantile_flattens_k_and_bands():
    x, w = np.array([[1.0, 2.0], [3.0, 4.0]]), np.ones((2, 2))
    assert BandUncertaintyPostProcessor.weighted_quantile(x, w, 0.75) == 3.0


@pytest.mark.parametrize("q", [-0.1, 1.1])
def test_weighted_quantile_rejects_q_out_of_range(q):
    with pytest.raises(ValueError, match="`quantile` must be"):
        BandUncertaintyPostProcessor.weighted_quantile(np.ones(2), np.ones(2), q)


# ---------------------------------- conformal factor ----------------------------------


@pytest.mark.parametrize(("n", "q", "rank"), [(9, 0.9, 9), (19, 0.9, 18), (10, 0.9, 10), (99, 0.95, 95), (3, 0.5, 2)])
def test_conformal_factor_is_the_ceil_n_plus_one_q_th_smallest_ratio(n, q, rank):
    ratios = np.arange(1.0, n + 1)[::-1]  # the value of a ratio is its rank
    assert UncertaintyCalibrator.conformal_factor(ratios, q) == rank


@pytest.mark.parametrize(("n", "q", "rank"), [(99, 0.07, 7), (99, 0.55, 55)])
def test_conformal_rank_is_not_pushed_up_by_rounding(n, q, rank):
    """(n + 1) q is a whole number here, but evaluates to just above it in floating point."""
    assert UncertaintyCalibrator.conformal_factor(np.arange(1.0, n + 1), q) == rank


def test_too_few_ratios_are_rejected_with_the_number_needed():
    with pytest.raises(ValueError, match="need at least 9"):
        UncertaintyCalibrator.conformal_factor(np.arange(8.0), 0.9)


@pytest.mark.parametrize("q", [0.0, 1.0, -0.1])
def test_conformal_q_out_of_range_is_rejected(q):
    with pytest.raises(ValueError, match="`q` must be"):
        UncertaintyCalibrator.conformal_factor(np.arange(20.0), q)


def test_conformal_factor_delivers_its_coverage_on_new_exchangeable_ratios():
    """The guarantee itself: for n = 19, q = 0.9 the probability that a new ratio is covered is 18/20."""
    rng = np.random.default_rng(0)
    trials = rng.lognormal(size=(20000, 20))
    covered = [ratios[-1] <= UncertaintyCalibrator.conformal_factor(ratios[:-1], 0.9) for ratios in trials]
    assert np.mean(covered) == pytest.approx(0.9, abs=0.01)


def test_zero_uncertainty_gives_an_infinite_ratio():
    ratios = UncertaintyCalibrator._ratios(np.array([0.0, 2.0]), np.array([1.0, 1.0]))
    np.testing.assert_array_equal(ratios, [np.inf, 0.5])


# ---------------------------------- calibrator ----------------------------------


def flat(value, key, n_k=1):
    """A structure where every state has this value, so that every statistic equals it."""
    eigvals = [[-1.0, 1.0]] * n_k
    return structure(values=[[value, value]] * n_k, eigvals=eigvals, weights=[1] * n_k, key=key, vbm_index=0)


def calibrator(u, e, **kwargs):
    """Structures s0, s1, ... with uncertainty u[i] and error e[i]."""
    return UncertaintyCalibrator(
        BandUncertaintyPostProcessor({f"s{i}": flat(v, SIGMA) for i, v in enumerate(u)}, key=SIGMA),
        BandUncertaintyPostProcessor({f"s{i}": flat(v, ERROR) for i, v in enumerate(e)}, key=ERROR),
        **kwargs,
    )


def two_k_calibrator(stat, **kwargs):
    return UncertaintyCalibrator(
        BandUncertaintyPostProcessor({"a": structure()}, key=SIGMA),
        BandUncertaintyPostProcessor({"a": structure(key=ERROR)}, key=ERROR),
        stat=stat,
        e_cut=1.0,
        tau=1e-3,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("stat", "expected"),
    [
        ("max_eV", 8.0),
        ("mean_eV", 5.5),
        ("weighted_max_eV", 7.0),
        ("weighted_mean_eV", 5.5),
        ("weighted_quantile_eV", 7.0),
    ],
)
def test_score_uses_the_chosen_statistic(stat, expected):
    cal = two_k_calibrator(stat, quantile=0.9)
    assert cal.score(cal.uncertainty, "a") == pytest.approx(expected)


def test_unknown_stat_and_quantile_are_rejected():
    with pytest.raises(ValueError, match="`stat` must be"):
        calibrator([1.0], [1.0], stat="nonsense")
    with pytest.raises(ValueError, match="`quantile` must be"):
        calibrator([1.0], [1.0], quantile=2.0)


def test_from_json_reads_each_file_with_its_key(tmp_path):
    (tmp_path / "u.json").write_text(json.dumps({"a": flat(2.0, SIGMA)}))
    (tmp_path / "e.json").write_text(json.dumps({"a": flat(1.0, ERROR)}))
    _, u, e = UncertaintyCalibrator.from_json(tmp_path / "u.json", tmp_path / "e.json").paired()
    np.testing.assert_allclose(u, [2.0])
    np.testing.assert_allclose(e, [1.0])


@pytest.mark.parametrize("setting", ["align_mode", "ill_method", "ill_threshold"])
def test_structures_computed_with_other_settings_are_rejected(setting):
    other = flat(1.0, ERROR)
    other[setting] = "other"
    with pytest.raises(ValueError, match="differ in settings"):
        UncertaintyCalibrator(
            BandUncertaintyPostProcessor({"a": flat(1.0, SIGMA)}, key=SIGMA),
            BandUncertaintyPostProcessor({"a": other}, key=ERROR),
        )


def test_structures_computed_on_other_k_points_are_rejected():
    other = flat(1.0, ERROR)
    other["kpoints"]["k0"]["k_frac"] = [0.5, 0.0, 0.0]
    with pytest.raises(ValueError, match="k-points"):
        UncertaintyCalibrator(
            BandUncertaintyPostProcessor({"a": flat(1.0, SIGMA)}, key=SIGMA),
            BandUncertaintyPostProcessor({"a": other}, key=ERROR),
        )


def test_structures_with_another_number_of_k_points_are_rejected():
    with pytest.raises(ValueError):
        UncertaintyCalibrator(
            BandUncertaintyPostProcessor({"a": flat(1.0, SIGMA, n_k=3)}, key=SIGMA),
            BandUncertaintyPostProcessor({"a": flat(1.0, ERROR, n_k=2)}, key=ERROR),
        )


def test_structures_in_only_one_file_are_not_compared_and_not_paired():
    other = flat(1.0, ERROR)
    other["align_mode"] = "other"
    cal = UncertaintyCalibrator(
        BandUncertaintyPostProcessor({"a": flat(2.0, SIGMA), "b": flat(3.0, SIGMA)}, key=SIGMA),
        BandUncertaintyPostProcessor({"a": flat(1.0, ERROR), "c": other}, key=ERROR),
    )
    names, u, e = cal.paired()
    assert names == ["a"]
    np.testing.assert_allclose(u, [2.0])
    np.testing.assert_allclose(e, [1.0])


def test_paired_restricts_to_the_requested_structures_and_rejects_missing_ones():
    cal = calibrator([1.0, 2.0], [3.0, 4.0])
    names, u, e = cal.paired(["s1"])
    assert names == ["s1"]
    np.testing.assert_allclose(u, [2.0])
    np.testing.assert_allclose(e, [4.0])
    with pytest.raises(KeyError, match="not in both files"):
        cal.paired(["s9"])


def test_fit_takes_the_conformal_factor_of_the_ratios():
    u = np.ones(9)
    e = np.arange(1.0, 10.0)
    fit = calibrator(u, e).fit(q=0.9)
    assert fit["s"] == pytest.approx(9.0)
    assert fit["n"] == 9
    assert fit["ratio_median"] == pytest.approx(5.0)
    assert fit["ratio_max"] == pytest.approx(9.0)


def test_fit_ratios_divide_the_error_by_the_uncertainty():
    fit = calibrator(np.full(9, 2.0), np.arange(1.0, 10.0)).fit(q=0.9)
    assert fit["s"] == pytest.approx(4.5)


def test_fit_reports_the_rank_correlation():
    u = np.arange(1.0, 10.0)
    assert calibrator(u, 2 * u).fit()["rank_correlation"] == pytest.approx(1.0)
    assert calibrator(u, 1 / u).fit()["rank_correlation"] == pytest.approx(-1.0)


def test_fit_on_too_few_structures_is_rejected():
    with pytest.raises(ValueError, match="too few"):
        calibrator(np.ones(3), np.ones(3)).fit(q=0.9)


def test_fit_per_group_falls_back_to_the_global_factor_on_small_groups():
    # s0..s8 have ratios 1..9, s9..s17 have 11..19, s18 has 100
    ratios = np.r_[np.arange(1.0, 10.0), np.arange(11.0, 20.0), 100.0]
    group = lambda name: "a" if int(name[1:]) < 9 else ("b" if int(name[1:]) < 18 else "c")
    fit = calibrator(np.ones(19), ratios).fit(q=0.9, groups=group, min_group_size=9)
    assert fit["s"] == pytest.approx(19.0)  # 18th of 19
    assert fit["s_per_group"] == pytest.approx({"a": 9.0, "b": 19.0, "c": 19.0})
    assert fit["fallback_groups"] == ["c"]
    assert fit["group_sizes"] == {"a": 9, "b": 9, "c": 1}


def test_fit_per_group_falls_back_when_a_group_is_too_small_for_q():
    group = lambda name: "a" if int(name[1:]) < 3 else "b"
    fit = calibrator(np.ones(12), np.arange(1.0, 13.0)).fit(q=0.9, groups=group, min_group_size=1)
    assert fit["fallback_groups"] == ["a"]
    assert fit["s_per_group"]["a"] == fit["s"]


def test_fit_per_group_falls_back_below_min_group_size_even_when_q_could_be_met():
    groups = {f"s{i}": "a" if i < 9 else "b" for i in range(19)}
    ratios = np.r_[np.arange(1.0, 10.0), np.arange(11.0, 21.0)]
    fit = calibrator(np.ones(19), ratios).fit(q=0.9, groups=groups, min_group_size=10)
    assert fit["fallback_groups"] == ["a"]
    assert fit["s_per_group"]["a"] == fit["s"]


def test_groups_may_be_a_mapping():
    groups = {f"s{i}": "a" for i in range(9)}
    fit = calibrator(np.ones(9), np.arange(1.0, 10.0)).fit(groups=groups, min_group_size=9)
    assert fit["s_per_group"] == pytest.approx({"a": 9.0})


def test_coverage_counts_the_structures_inside_the_bound_and_lists_the_others():
    result = calibrator([1.0, 1.0, 2.0, 1.0], [1.0, 2.0, 5.0, 3.0]).coverage(2.0)
    assert result["coverage"] == pytest.approx(0.5)  # e <= 2 u holds for s0, s1, with the bound itself included
    assert set(result["violations"]) == {"s2", "s3"}
    assert result["violations"]["s2"] == pytest.approx({"u": 2.0, "e": 5.0, "bound": 4.0})


def test_coverage_with_per_group_factors():
    groups = {"s0": "a", "s1": "b"}
    cal = calibrator([1.0, 1.0], [2.0, 2.0])
    assert cal.coverage({"a": 3.0, "b": 1.0}, groups=groups)["coverage"] == pytest.approx(0.5)


def test_per_group_factors_need_groups():
    with pytest.raises(ValueError, match="needs `groups`"):
        calibrator([1.0], [1.0]).coverage({"a": 1.0})


def test_coverage_can_be_restricted_to_some_structures():
    result = calibrator([1.0, 1.0], [1.0, 9.0]).coverage(2.0, structures=["s0"])
    assert result["n"] == 1
    assert result["coverage"] == 1.0


def test_coverage_of_no_structures_is_nan():
    assert np.isnan(calibrator([1.0], [1.0]).coverage(2.0, structures=[])["coverage"])


def pool_of(u):
    return BandUncertaintyPostProcessor({f"p{i}": flat(v, SIGMA) for i, v in enumerate(u)}, key=SIGMA)


def test_pool_criterion_compares_the_percentile_of_the_bounds_with_the_tolerance():
    # bounds 2, 4, 6, 8, 10: their median is 6
    cal = calibrator([1.0], [1.0])
    pool = pool_of([1.0, 2.0, 3.0, 4.0, 5.0])
    result = cal.pool_criterion(pool, 2.0, eta_tol=6.0, percentile=50.0)
    assert result["stop"] is True
    assert result["bound_percentile_eV"] == pytest.approx(6.0)
    assert result["bound_max_eV"] == pytest.approx(10.0)
    assert result["n_above_eta"] == 2
    assert result["n_pool"] == 5
    assert list(result["largest_bounds"]) == ["p4", "p3", "p2", "p1", "p0"]


def test_pool_criterion_does_not_stop_above_the_tolerance():
    assert calibrator([1.0], [1.0]).pool_criterion(pool_of([1.0, 2.0, 3.0]), 2.0, 3.9, percentile=50.0)["stop"] is False


def test_pool_criterion_ignores_outliers_beyond_its_percentile():
    pool = pool_of([*[1.0] * 99, 1000.0])
    result = calibrator([1.0], [1.0]).pool_criterion(pool, 1.0, eta_tol=1.0, percentile=95.0)
    assert result["stop"] is True
    assert result["n_above_eta"] == 1


def test_pool_criterion_with_per_group_factors():
    groups = {"p0": "a", "p1": "b"}
    result = calibrator([1.0], [1.0]).pool_criterion(
        pool_of([1.0, 1.0]), {"a": 2.0, "b": 4.0}, eta_tol=9.0, percentile=100.0, groups=groups
    )
    assert result["bound_max_eV"] == pytest.approx(4.0)
