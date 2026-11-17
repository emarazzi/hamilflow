import warnings

import numpy as np
import pytest

from hamilflow.candidate_selection import (
    CandidateSelector,
    GroupQuotaSelection,
    allocate_quotas,
    select_by_group_quota,
)

# -- allocate_quotas ------------------------------------------------------------


def test_equal_weights_split_evenly():
    assert allocate_quotas(9, ["a", "b", "c"]) == {"a": 3, "b": 3, "c": 3}


def test_proportional_to_weights():
    assert allocate_quotas(12, ["a", "b", "c"], weights={"a": 3, "b": 2, "c": 1}) == {"a": 6, "b": 4, "c": 2}


def test_largest_remainder():
    # raw shares 10 * (3, 2, 1) / 6 = 5, 3.33, 1.67 -> floors 5, 3, 1; the extra unit goes to "c" (0.67 > 0.33)
    assert allocate_quotas(10, ["a", "b", "c"], weights={"a": 3, "b": 2, "c": 1}) == {"a": 5, "b": 3, "c": 2}


def test_ties_go_to_the_earlier_group():
    assert allocate_quotas(4, ["a", "b", "c"]) == {"a": 2, "b": 1, "c": 1}
    assert allocate_quotas(4, ["c", "b", "a"]) == {"c": 2, "b": 1, "a": 1}
    # output order follows the given group order
    assert list(allocate_quotas(4, ["c", "b", "a"])) == ["c", "b", "a"]


def test_callable_weights():
    q = allocate_quotas(6, [1, 2], weights=lambda g: g)
    assert q == {1: 2, 2: 4}


def test_min_per_group_lifts_small_groups_and_takes_from_the_rest():
    # proportional 10 * (8, 1, 1) / 10 = 8, 1, 1; min 2 -> the big group pays for the small ones
    q = allocate_quotas(10, ["a", "b", "c"], weights={"a": 8, "b": 1, "c": 1}, min_per_group=2)
    assert q == {"a": 6, "b": 2, "c": 2}
    assert sum(q.values()) == 10


def test_fixed_overrides():
    q = allocate_quotas(10, ["a", "b", "c"], fixed={"b": 6})
    assert q == {"a": 2, "b": 6, "c": 2}
    # a fixed quota is exempt from min_per_group
    q = allocate_quotas(10, ["a", "b", "c"], fixed={"b": 0}, min_per_group=3)
    assert q == {"a": 5, "b": 0, "c": 5}


def test_cap_at_available_redistributes_the_excess():
    q = allocate_quotas(12, ["a", "b", "c"], available={"a": 1, "b": 100, "c": 100})
    assert q == {"a": 1, "b": 6, "c": 5}
    # min_per_group cannot exceed the capacity either
    q = allocate_quotas(6, ["a", "b"], available={"a": 1, "b": 10}, min_per_group=2)
    assert q == {"a": 1, "b": 5}


def test_cap_shortfall_warns():
    with pytest.warns(UserWarning, match="instead of 10"):
        q = allocate_quotas(10, ["a", "b"], available={"a": 2, "b": 3})
    assert q == {"a": 2, "b": 3}


def test_fixed_capped_at_available_warns():
    with pytest.warns(UserWarning, match="capped"):
        q = allocate_quotas(5, ["a", "b"], available={"a": 2, "b": 10}, fixed={"a": 4})
    assert q == {"a": 2, "b": 3}


def test_zero_weight_gets_only_its_minimum():
    q = allocate_quotas(6, ["a", "b", "c"], weights={"a": 1, "b": 0, "c": 1}, min_per_group=1)
    assert q == {"a": 3, "b": 1, "c": 2}  # 2.5 / 1 / 2.5, the tie goes to "a"


def test_quotas_always_sum_to_n_total():
    rng = np.random.default_rng(0)
    for _ in range(200):
        ng = int(rng.integers(1, 8))
        groups = list(range(ng))
        w = dict(zip(groups, rng.uniform(0.1, 5, ng)))
        avail = dict(zip(groups, rng.integers(0, 15, ng).tolist()))
        n = int(rng.integers(0, sum(avail.values()) + 1))
        mn = int(rng.integers(0, 3))
        if sum(min(mn, a) for a in avail.values()) > n:
            continue
        q = allocate_quotas(n, groups, weights=w, available=avail, min_per_group=mn)
        assert sum(q.values()) == n
        assert all(min(mn, avail[g]) <= q[g] <= avail[g] for g in groups)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"n_total": 5, "groups": ["a", "a"]}, "distinct"),
        ({"n_total": -1, "groups": ["a"]}, ">= 0"),
        ({"n_total": 2.5, "groups": ["a"]}, "integers"),
        ({"n_total": 5, "groups": ["a"], "fixed": {"z": 1}}, "unknown"),
        ({"n_total": 5, "groups": ["a", "b"], "fixed": {"a": -1}}, "non-negative"),
        ({"n_total": 3, "groups": ["a", "b"], "fixed": {"a": 4}}, "exceed n_total"),
        ({"n_total": 3, "groups": ["a", "b"], "min_per_group": 2}, "too small"),
        ({"n_total": 3, "groups": ["a", "b"], "weights": {"a": 1}}, "no entry"),
        ({"n_total": 3, "groups": ["a", "b"], "weights": {"a": 1, "b": -1}}, "finite"),
        ({"n_total": 3, "groups": ["a", "b"], "weights": {"a": 1, "b": np.nan}}, "finite"),
        ({"n_total": 3, "groups": ["a", "b"], "available": {"a": 1}}, "no entry"),
    ],
)
def test_allocate_quotas_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        allocate_quotas(**kwargs)


# -- select_by_group_quota ------------------------------------------------------


def _clustered_groups():
    """Two groups; group "a" has 3 tight blobs, group "b" has 2. Candidate i
    sits in blob i % n_blobs of its group (5 points per blob)."""
    rng = np.random.default_rng(1)
    X, labels, blob = [], [], []
    for g, centers in (("a", [(0, 0), (10, 0), (0, 10)]), ("b", [(50, 50), (60, 50)])):
        for j in range(5 * len(centers)):
            c = j % len(centers)
            X.append(np.array(centers[c]) + rng.normal(scale=0.1, size=2))
            labels.append(g)
            blob.append((g, c))
    return np.array(X), labels, blob


def test_one_pick_per_blob_with_centroid_rule():
    X, labels, blob = _clustered_groups()
    res = select_by_group_quota(X, labels, 5, weights={"a": 3, "b": 2})
    assert isinstance(res, GroupQuotaSelection)
    assert res.quotas == {"a": 3, "b": 2}
    assert res.n_selected == {"a": 3, "b": 2}
    assert sorted(blob[i] for i in res.selected_idx) == [("a", 0), ("a", 1), ("a", 2), ("b", 0), ("b", 1)]
    assert list(res.cluster_size) == [5, 5, 5, 5, 5]
    # grouped in group order, ascending within each group
    assert [labels[i] for i in res.selected_idx] == ["a", "a", "a", "b", "b"]
    assert np.all(np.diff(res.selected_by_group["a"]) > 0)
    rows = res.summary()
    assert rows[0] == {"group": "a", "weight": 3.0, "quota": 3, "selected": 3, "available": 15, "candidates": 15}


def test_centroid_rule_picks_the_member_closest_to_the_centroid():
    X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.5]])
    res = select_by_group_quota(X, ["g"] * 6, 2)
    assert sorted(res.selected_idx) == [1, 4]


def test_max_score_rule_picks_the_highest_score_in_each_cluster():
    X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
    scores = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.1])
    res = select_by_group_quota(X, ["g"] * 6, 2, scores=scores, pick="max_score")
    assert sorted(res.selected_idx) == [2, 3]


def test_max_score_ties_go_to_the_lowest_index():
    X = np.array([[0.0], [0.5], [10.0], [10.5]])
    res = select_by_group_quota(X, ["g"] * 4, 2, scores=np.ones(4), pick="max_score")
    assert sorted(res.selected_idx) == [0, 2]


def test_top_fraction_filter_within_each_group():
    X = np.arange(20, dtype=float)[:, None]
    labels = ["a"] * 10 + ["b"] * 10
    scores = np.r_[np.arange(10), np.arange(10)[::-1]].astype(float)  # a: high at the end, b: high at the start
    res = select_by_group_quota(X, labels, 4, scores=scores, top_fraction=0.3)
    assert res.n_candidates == {"a": 3, "b": 3}
    assert list(res.candidate_idx) == [7, 8, 9, 10, 11, 12]
    assert set(res.selected_idx) <= set(res.candidate_idx)
    assert res.n_available == {"a": 10, "b": 10}


def test_top_n_filter_and_never_below_the_quota():
    X = np.arange(20, dtype=float)[:, None]
    labels = ["a"] * 10 + ["b"] * 10
    scores = np.arange(20, dtype=float)
    res = select_by_group_quota(X, labels, 8, scores=scores, top_n=2, fixed={"a": 5})
    assert res.quotas == {"a": 5, "b": 3}
    # "a" keeps max(top_n, quota) = 5, "b" keeps 3
    assert res.n_candidates == {"a": 5, "b": 3}
    assert list(res.candidate_idx) == [5, 6, 7, 8, 9, 17, 18, 19]
    # quota == number of candidates -> taken whole
    assert list(res.selected_idx) == [5, 6, 7, 8, 9, 17, 18, 19]


def test_score_filter_ties_keep_the_lower_index():
    X = np.arange(6, dtype=float)[:, None]
    res = select_by_group_quota(X, ["g"] * 6, 1, scores=np.zeros(6), top_n=2)
    assert list(res.candidate_idx) == [0, 1]


def test_deterministic_and_seed_dependent_only_through_kmeans():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 4))
    labels = [i % 3 for i in range(60)]
    r1 = select_by_group_quota(X, labels, 10, seed=7)
    r2 = select_by_group_quota(X, labels, 10, seed=7)
    np.testing.assert_array_equal(r1.selected_idx, r2.selected_idx)
    np.testing.assert_array_equal(r1.cluster_size, r2.cluster_size)
    assert r1.quotas == {0: 4, 1: 3, 2: 3}


def test_group_order_controls_ties_and_output_order():
    X = np.arange(12, dtype=float)[:, None]
    labels = ["a"] * 4 + ["b"] * 4 + ["c"] * 4
    res = select_by_group_quota(X, labels, 4, group_order=["c", "b", "a"])
    assert res.groups == ["c", "b", "a"]
    assert res.quotas == {"c": 2, "b": 1, "a": 1}
    assert [labels[i] for i in res.selected_idx] == ["c", "c", "b", "a"]


def test_candidate_idx_restricts_the_pool():
    X = np.arange(12, dtype=float)[:, None]
    labels = ["a"] * 6 + ["b"] * 6
    res = select_by_group_quota(X, labels, 4, candidate_idx=[0, 1, 6, 7, 8, 9, 10, 11])
    assert res.n_available == {"a": 2, "b": 6}
    assert set(res.selected_idx) <= {0, 1, 6, 7, 8, 9, 10, 11}
    assert len(res.selected_idx) == 4


def test_sample_weight_is_used():
    # with zero weight on the far point, k=1 centroid sits among the near points
    X = np.array([[0.0], [0.1], [0.2], [100.0]])
    res = select_by_group_quota(X, ["g"] * 4, 1, sample_weight=[1, 1, 1, 0])
    assert list(res.selected_idx) == [1]


def test_degenerate_descriptors_warn_about_empty_clusters():
    X = np.zeros((5, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn ConvergenceWarning
        res = select_by_group_quota(X, ["g"] * 5, 3)
    assert len(res.selected_idx) <= 3
    assert res.quotas == {"g": 3}


def test_amd_like_list_of_vectors_and_hashable_groups():
    amds = [np.full(3, float(i)) for i in range(8)]
    groups = [(i % 2, "x") for i in range(8)]
    res = select_by_group_quota(amds, groups, 4)
    assert res.quotas == {(0, "x"): 2, (1, "x"): 2}


def test_candidate_selector_method_uses_its_seed():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, 3))
    labels = [i % 2 for i in range(40)]
    res = CandidateSelector(seed=11).quota_selection(X, labels, 6, min_per_group=1)
    ref = select_by_group_quota(X, labels, 6, min_per_group=1, seed=11)
    np.testing.assert_array_equal(res.selected_idx, ref.selected_idx)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"pick": "random"}, "Unknown pick"),
        ({"pick": "max_score"}, "need scores"),
        ({"top_fraction": 0.5}, "need scores"),
        ({"scores": np.ones(4), "top_fraction": 0.5, "top_n": 1}, "not both"),
        ({"scores": np.ones(4), "top_fraction": 0.0}, r"\(0, 1\]"),
        ({"scores": np.ones(4), "top_fraction": 1.5}, r"\(0, 1\]"),
        ({"scores": np.ones(4), "top_n": 0}, ">= 1"),
        ({"scores": np.ones(3)}, "scores must have shape"),
        ({"scores": np.array([1, 2, np.nan, 3])}, "non-finite"),
        ({"sample_weight": [1, 1, -1, 1]}, ">= 0"),
        ({"sample_weight": [1, 1]}, "sample_weight must have shape"),
        ({"candidate_idx": [0, 9]}, "out of range"),
        ({"candidate_idx": []}, "no candidates"),
        ({"group_order": ["a"]}, "misses"),
        ({"group_order": ["a", "b", "a"]}, "duplicates"),
    ],
)
def test_select_by_group_quota_validation(kwargs, match):
    X = np.arange(4, dtype=float)[:, None]
    with pytest.raises(ValueError, match=match):
        select_by_group_quota(X, ["a", "a", "b", "b"], 2, **kwargs)


def test_descriptor_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        select_by_group_quota(np.zeros((3, 2)), ["a"] * 4, 2)
