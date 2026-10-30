"""K-fold splits: the partition invariants, family spreading, and frozen assignments across AL cycles."""

import json
from collections import Counter

import pytest

from hamilflow.training.dataset_split import (
    make_kfold_splits,
    parent_of,
    read_kfold_splits,
    write_kfold_splits,
)

K_VALUES = [2, 3, 5]


def family_names(n_families=6, n_variants=5):
    return [f"tMoS2Bilayer_n{f}m1Ang{f}.5-{v:03d}" for f in range(n_families) for v in range(n_variants)]


def assert_is_kfold_partition(splits, names):
    validates = [set(s["validate"]) for s in splits]
    assert sum(len(s["validate"]) for s in splits) == len(set(names))
    assert set().union(*validates) == set(names)
    for split, validate in zip(splits, validates, strict=True):
        assert set(split["train"]) == set(names) - validate
        assert len(split["train"]) == len(set(split["train"]))


@pytest.mark.parametrize(
    ("name", "parent"),
    [
        ("tMoS2Bilayer_n1m5Ang6-003", "tMoS2Bilayer_n1m5Ang6"),
        ("tMoS2Bilayer_n1m5Ang6-init", "tMoS2Bilayer_n1m5Ang6"),
        ("a-b-c", "a-b"),
        ("no_suffix", "no_suffix"),
    ],
)
def test_parent_of_cuts_at_the_last_dash(name, parent):
    assert parent_of(name) == parent


@pytest.mark.parametrize("k", [1, 0, -1])
def test_k_below_two_is_rejected(k):
    with pytest.raises(ValueError, match="k must be"):
        make_kfold_splits(["a", "b"], k)


@pytest.mark.parametrize("k", K_VALUES)
@pytest.mark.parametrize("group_key", [parent_of, None], ids=["families", "random"])
def test_every_structure_is_validated_exactly_once(k, group_key):
    names = family_names()
    splits = make_kfold_splits(names, k, group_key=group_key)
    assert len(splits) == k
    assert_is_kfold_partition(splits, names)
    assert all(s["test"] == [] for s in splits)


def test_duplicate_names_are_counted_once():
    splits = make_kfold_splits(["a", "b", "a", "c", "b"], 2, group_key=None)
    assert_is_kfold_partition(splits, ["a", "b", "c"])


@pytest.mark.parametrize("k", K_VALUES)
@pytest.mark.parametrize("seed", range(5))
def test_every_member_trains_on_every_family(k, seed):
    """The point of the stratification: no fold may hold a whole family out of training."""
    names = family_names()
    for split in make_kfold_splits(names, k, seed=seed):
        assert {parent_of(n) for n in split["train"]} == {parent_of(n) for n in names}


@pytest.mark.parametrize("seed", range(5))
def test_family_of_at_most_k_members_has_one_member_per_validate_set(seed):
    names = family_names(n_families=6, n_variants=5)
    for split in make_kfold_splits(names, 5, seed=seed):
        assert max(Counter(parent_of(n) for n in split["validate"]).values()) == 1


@pytest.mark.parametrize("k", K_VALUES)
def test_family_larger_than_k_is_spread_evenly(k):
    names = family_names(n_families=1, n_variants=2 * k + 1)
    counts = [len(s["validate"]) for s in make_kfold_splits(names, k)]
    assert max(counts) - min(counts) == 1


@pytest.mark.parametrize("k", K_VALUES)
@pytest.mark.parametrize("group_key", [parent_of, None], ids=["families", "random"])
def test_validate_sets_are_balanced(k, group_key):
    names = family_names(n_families=7, n_variants=3) + [f"single_{i}" for i in range(4)]
    sizes = [len(s["validate"]) for s in make_kfold_splits(names, k, group_key=group_key)]
    assert max(sizes) - min(sizes) <= 1


def test_custom_group_key_defines_the_families():
    names = [f"Si64_rattle{i:02d}" for i in range(4)]
    key = lambda name: name.rsplit("_rattle", 1)[0]
    for split in make_kfold_splits(names, 4, group_key=key):
        assert len(split["validate"]) == 1


def test_same_seed_is_reproducible_and_other_seeds_differ():
    names = family_names()
    assert make_kfold_splits(names, 5, seed=1) == make_kfold_splits(names, 5, seed=1)
    assert make_kfold_splits(names, 5, seed=1) != make_kfold_splits(names, 5, seed=2)


def test_next_cycle_freezes_previous_assignments_and_order():
    old = family_names(n_families=4)
    previous = make_kfold_splits(old, 5)
    new = family_names(n_families=6)[len(old) :]
    splits = make_kfold_splits(old + new, 5, previous=previous)

    assert_is_kfold_partition(splits, old + new)
    for split, before in zip(splits, previous, strict=True):
        assert split["train"][: len(before["train"])] == before["train"]
        assert split["validate"][: len(before["validate"])] == before["validate"]


def test_next_cycle_without_new_structures_changes_nothing():
    names = family_names()
    previous = make_kfold_splits(names, 5)
    assert make_kfold_splits(names, 5, previous=previous) == previous


def test_new_variant_avoids_the_fold_already_validating_its_family():
    previous = [
        {"train": ["g-0", "h-0"], "validate": ["f-0"], "test": []},
        {"train": ["f-0", "h-0"], "validate": ["g-0"], "test": []},
        {"train": ["f-0", "g-0"], "validate": ["h-0"], "test": []},
    ]
    splits = make_kfold_splits(["f-0", "g-0", "h-0", "f-1"], 3, previous=previous)
    assert "f-1" not in splits[0]["validate"]
    assert "f-1" in splits[0]["train"]


def test_spreading_a_family_outranks_keeping_folds_balanced():
    previous = [
        {"train": ["x-0", "y-0"], "validate": ["f-0"], "test": []},
        {"train": ["f-0"], "validate": ["x-0", "y-0"], "test": []},
    ]
    splits = make_kfold_splits(["f-0", "x-0", "y-0", "f-1"], 2, previous=previous)
    assert "f-1" in splits[1]["validate"]


def test_new_variant_goes_to_the_smaller_fold_when_family_counts_tie():
    previous = [
        {"train": ["f-1", "f-2"], "validate": ["f-0", "x-0"], "test": []},
        {"train": ["f-0", "x-0", "f-2"], "validate": ["f-1"], "test": []},
        {"train": ["f-0", "x-0", "f-1"], "validate": ["f-2"], "test": []},
    ]
    splits = make_kfold_splits(["f-0", "f-1", "f-2", "x-0", "f-3"], 3, previous=previous)
    assert "f-3" not in splits[0]["validate"]


def test_next_cycle_keeps_the_test_set_out_of_the_new_structures():
    previous = [
        {"train": ["b"], "validate": ["a"], "test": ["t"]},
        {"train": ["a"], "validate": ["b"], "test": ["t"]},
    ]
    splits = make_kfold_splits(["a", "b", "t", "c"], 2, previous=previous)
    for split in splits:
        assert split["test"] == ["t"]
        assert "t" not in split["train"] + split["validate"]
    assert_is_kfold_partition(splits, ["a", "b", "c"])


def test_previous_with_wrong_number_of_folds_is_rejected():
    with pytest.raises(ValueError, match="expected k=3"):
        make_kfold_splits(["a", "b"], 3, previous=make_kfold_splits(["a", "b"], 2, group_key=None))


def test_previous_with_overlapping_validate_sets_is_rejected():
    previous = [{"train": ["b"], "validate": ["a"]}, {"train": ["b"], "validate": ["a"]}]
    with pytest.raises(ValueError, match="overlapping"):
        make_kfold_splits(["a", "b"], 2, previous=previous)


@pytest.mark.parametrize(
    "train", [[], ["a", "b"]], ids=["structure-missing-from-train", "structure-in-both-train-and-validate"]
)
def test_previous_that_is_not_a_partition_is_rejected(train):
    previous = [{"train": train, "validate": ["a"]}, {"train": ["a"], "validate": ["b"]}]
    with pytest.raises(ValueError, match="not a k-fold partition"):
        make_kfold_splits(["a", "b"], 2, previous=previous)


def test_structures_dropped_since_the_previous_cycle_are_rejected():
    previous = make_kfold_splits(["a", "b"], 2, group_key=None)
    with pytest.raises(ValueError, match="not in names"):
        make_kfold_splits(["a"], 2, previous=previous)


def test_write_then_read_round_trip(tmp_path):
    names = family_names()
    paths = write_kfold_splits(names, 3, tmp_path / "cycle1")
    assert [p.name for p in paths] == ["dataset_split_1.json", "dataset_split_2.json", "dataset_split_3.json"]
    assert read_kfold_splits(tmp_path / "cycle1", 3) == make_kfold_splits(names, 3)
    assert json.loads(paths[0].read_text()) == make_kfold_splits(names, 3)[0]


def test_write_extends_the_splits_of_previous_dir(tmp_path):
    old = family_names(n_families=2)
    write_kfold_splits(old, 3, tmp_path / "cycle1")
    new = old + ["new_structure"]
    write_kfold_splits(new, 3, tmp_path / "cycle2", previous_dir=tmp_path / "cycle1")

    before = read_kfold_splits(tmp_path / "cycle1", 3)
    after = read_kfold_splits(tmp_path / "cycle2", 3)
    assert_is_kfold_partition(after, new)
    for split, previous in zip(after, before, strict=True):
        assert split["validate"][: len(previous["validate"])] == previous["validate"]


def test_prefix_names_the_files(tmp_path):
    write_kfold_splits(["a", "b"], 2, tmp_path, group_key=None, prefix="other")
    assert len(read_kfold_splits(tmp_path, 2, prefix="other")) == 2
