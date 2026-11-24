"""The wiring of one AL cycle: splits -> per-fold train.toml -> trained run -> per-fold infer.toml."""

import copy
import json
import tomllib

import pytest

from hamilflow.training.dataset_split import write_kfold_splits
from hamilflow.training.ensemble import (
    deep_merge,
    find_run_dir,
    fold_dir,
    train_outputs_dir,
    write_ensemble_infer_tomls,
    write_ensemble_train_tomls,
)

K = 3
NAMES = [f"family{f}-{v:03d}" for f in range(4) for v in range(3)]
OVERRIDES = {"data": {"inputs_dir": "/inputs", "outputs_dir": "./outputs"}, "system": {"random_seed": 100}}


def read_toml(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def make_run(outputs_dir, name="2026-10-22_10-30-00"):
    run = outputs_dir / name
    (run / "model").mkdir(parents=True)
    return run


@pytest.fixture
def cycle(tmp_path):
    cycle_dir = tmp_path / "cycle1"
    write_kfold_splits(NAMES, K, cycle_dir)
    return cycle_dir


def test_find_run_dir_takes_the_most_recent_run_and_warns(tmp_path):
    make_run(tmp_path, "2026-10-22_10-30-00")
    newest = make_run(tmp_path, "2026-11-01_08-00-00")
    make_run(tmp_path, "2026-01-01_00-00-00")
    with pytest.warns(UserWarning, match="3 training runs"):
        assert find_run_dir(tmp_path) == newest


def test_find_run_dir_ignores_folders_that_are_not_finished_runs(tmp_path):
    run = make_run(tmp_path, "2026-10-22_10-30-00")
    (tmp_path / "2026-12-31_00-00-00").mkdir()  # a newer run with no model/
    make_run(tmp_path, "not_a_timestamp")
    assert find_run_dir(tmp_path) == run


def test_find_run_dir_without_a_run_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No yyyy-mm-dd_hh-mm-ss"):
        find_run_dir(tmp_path)


def test_outputs_dir_is_resolved_against_the_fold_unless_absolute(cycle, tmp_path):
    write_ensemble_train_tomls(OVERRIDES, cycle, K)
    assert train_outputs_dir(cycle, 2) == fold_dir(cycle, 2) / "outputs"

    absolute = copy.deepcopy(OVERRIDES)
    absolute["data"]["outputs_dir"] = str(tmp_path / "elsewhere")
    write_ensemble_train_tomls(absolute, cycle, K)
    assert train_outputs_dir(cycle, 2) == tmp_path / "elsewhere"


def test_each_fold_gets_its_own_split_sizes_and_seed(cycle):
    paths = write_ensemble_train_tomls(OVERRIDES, cycle, K)
    assert paths == [fold_dir(cycle, j) / "train.toml" for j in range(1, K + 1)]
    for j, path in enumerate(paths, start=1):
        written = read_toml(path)
        split_path = cycle / f"dataset_split_{j}.json"
        split = json.loads(split_path.read_text())
        dataloader = written["process"]["train"]["dataloader"]
        assert dataloader["dataset_split_json"] == str(split_path.resolve())
        assert dataloader["train_size"] == len(split["train"])
        assert dataloader["validate_size"] == len(split["validate"])
        assert dataloader["test_size"] == 0
        assert written["system"]["random_seed"] == 100 + j
        assert written["process"]["train"]["continued"]["enable"] is False


def test_seed_defaults_to_the_config_default_plus_fold(cycle):
    overrides = {"data": OVERRIDES["data"]}
    path = write_ensemble_train_tomls(overrides, cycle, K)[0]
    assert read_toml(path)["system"]["random_seed"] == 137 + 1


def test_overrides_are_not_changed(cycle):
    overrides = copy.deepcopy(OVERRIDES)
    write_ensemble_train_tomls(overrides, cycle, K)
    assert overrides == OVERRIDES


@pytest.mark.parametrize("key", ["dataset_split_json", "train_size", "validate_size", "test_size"])
def test_split_keys_in_overrides_are_rejected(cycle, key):
    overrides = {"process": {"train": {"dataloader": {key: 1}}}}
    with pytest.raises(ValueError, match=key):
        write_ensemble_train_tomls(overrides, cycle, K)


def test_previous_cycle_and_continue_from_are_exclusive(cycle, tmp_path):
    with pytest.raises(ValueError, match="not both"):
        write_ensemble_train_tomls(OVERRIDES, cycle, K, previous_cycle_dir=tmp_path, continue_from="outputs_3")


def test_missing_split_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        write_ensemble_train_tomls(OVERRIDES, tmp_path, K)


def test_previous_cycle_warm_starts_each_fold_from_its_own_run_on_new_data(cycle, tmp_path):
    write_ensemble_train_tomls(OVERRIDES, cycle, K)
    runs = [make_run(train_outputs_dir(cycle, j)) for j in range(1, K + 1)]

    next_cycle = tmp_path / "cycle2"
    write_kfold_splits([*NAMES, "family9-000"], K, next_cycle, previous_dir=cycle)
    paths = write_ensemble_train_tomls(OVERRIDES, next_cycle, K, previous_cycle_dir=cycle)

    for path, run in zip(paths, runs, strict=True):
        continued = read_toml(path)["process"]["train"]["continued"]
        assert continued["enable"] is True
        assert continued["new_training_data"] is True
        assert continued["previous_output_dir"] == str(run.resolve())


def test_continue_from_resumes_each_fold_on_the_same_data(cycle):
    runs = [make_run(fold_dir(cycle, j) / "outputs_3") for j in range(1, K + 1)]
    paths = write_ensemble_train_tomls(OVERRIDES, cycle, K, continue_from="outputs_3")
    for path, run in zip(paths, runs, strict=True):
        continued = read_toml(path)["process"]["train"]["continued"]
        assert continued["enable"] is True
        assert continued["new_training_data"] is False
        assert continued["previous_output_dir"] == str(run.resolve())


def test_other_continued_keys_come_from_the_overrides(cycle):
    for j in range(1, K + 1):
        make_run(fold_dir(cycle, j) / "outputs_3")
    overrides = copy.deepcopy(OVERRIDES)
    overrides["process"] = {"train": {"continued": {"new_optimizer": True, "load_model_type": "best"}}}
    path = write_ensemble_train_tomls(overrides, cycle, K, continue_from="outputs_3")[0]
    continued = read_toml(path)["process"]["train"]["continued"]
    assert continued["new_optimizer"] is True
    assert continued["load_model_type"] == "best"


def test_infer_tomls_point_at_the_model_of_each_fold(cycle):
    write_ensemble_train_tomls(OVERRIDES, cycle, K)
    runs = [make_run(train_outputs_dir(cycle, j)) for j in range(1, K + 1)]
    paths = write_ensemble_infer_tomls({"data": {"inputs_dir": "/infer"}}, cycle, K, infer_name="infer_md")
    assert paths == [fold_dir(cycle, j) / "infer_md" / "infer.toml" for j in range(1, K + 1)]
    for path, run in zip(paths, runs, strict=True):
        written = read_toml(path)
        assert written["model"]["model_dir"] == str((run / "model").resolve())
        assert written["data"]["inputs_dir"] == "/infer"


def test_infer_tomls_before_training_is_an_error(cycle):
    write_ensemble_train_tomls(OVERRIDES, cycle, K)
    (fold_dir(cycle, 1) / "outputs").mkdir()
    with pytest.raises(FileNotFoundError, match="No yyyy-mm-dd_hh-mm-ss"):
        write_ensemble_infer_tomls({}, cycle, K)


# ---------------------------------- per-fold settings and rounds ----------------------------------


def test_deep_merge_merges_tables_and_replaces_values():
    base = {"a": {"b": 1, "c": {"d": 2}}, "e": [1]}
    merged = deep_merge(base, {"a": {"c": {"d": 3, "f": 4}}, "e": [2]})
    assert merged == {"a": {"b": 1, "c": {"d": 3, "f": 4}}, "e": [2]}
    assert base == {"a": {"b": 1, "c": {"d": 2}}, "e": [1]}


def test_fold_overrides_are_merged_after_the_shared_ones(cycle):
    overrides = copy.deepcopy(OVERRIDES)
    overrides["process"] = {"train": {"scheduler": {"patience": 20, "cooldown": 10},
                                      "optimizer": {"init_learning_rate": 1e-3}}}
    fold_overrides = {2: {"process": {"train": {"scheduler": {"patience": 40, "min_learning_rate_scale": 4e-4},
                                                "optimizer": {"init_learning_rate": 2.5e-4}}}}}
    paths = write_ensemble_train_tomls(overrides, cycle, K, fold_overrides=fold_overrides)
    first, second = (read_toml(p)["process"]["train"] for p in paths[:2])
    assert (first["scheduler"]["patience"], first["optimizer"]["init_learning_rate"]) == (20, 1e-3)
    assert second["scheduler"]["patience"] == 40
    assert second["scheduler"]["cooldown"] == 10  # kept from the shared overrides
    assert second["scheduler"]["min_learning_rate_scale"] == 4e-4
    assert second["optimizer"]["init_learning_rate"] == 2.5e-4
    assert second["dataloader"]["train_size"] > 0
    assert fold_overrides[2]["process"]["train"]["scheduler"]["patience"] == 40  # not changed


def test_round_writes_its_own_toml_and_outputs_folder(cycle):
    paths = write_ensemble_train_tomls(OVERRIDES, cycle, K, round=2)
    assert paths == [fold_dir(cycle, j) / "train_r2.toml" for j in range(1, K + 1)]
    assert not (fold_dir(cycle, 1) / "train.toml").exists()
    assert read_toml(paths[0])["data"]["outputs_dir"] == "./outputs_r2"
    assert train_outputs_dir(cycle, 1, round=2) == fold_dir(cycle, 1) / "outputs_r2"


def test_round_outputs_default_and_absolute(cycle, tmp_path):
    path = write_ensemble_train_tomls({"data": {"inputs_dir": "/inputs"}}, cycle, K, round=1)[0]
    assert read_toml(path)["data"]["outputs_dir"] == "./outputs_r1"
    absolute = copy.deepcopy(OVERRIDES)
    absolute["data"]["outputs_dir"] = str(tmp_path / "out") + "/"
    path = write_ensemble_train_tomls(absolute, cycle, K, round=3)[0]
    assert read_toml(path)["data"]["outputs_dir"] == str(tmp_path / "out_r3")


@pytest.mark.parametrize("round_", [0, -1, 1.0, True, "2", 2.5])
def test_round_must_be_a_positive_integer(cycle, round_):
    with pytest.raises(ValueError, match="round must be"):
        write_ensemble_train_tomls(OVERRIDES, cycle, K, round=round_)


def test_continue_from_per_fold_outputs_folder_explicit_run_and_scratch(cycle, tmp_path):
    run_1 = make_run(fold_dir(cycle, 1) / "outputs_r1")
    run_2 = make_run(tmp_path / "elsewhere", "2026-10-22_11-00-00")
    continue_from = {
        1: "outputs_r1",
        2: {"previous_output_dir": str(run_2), "load_model_type": "best"},
        3: None,
    }
    paths = write_ensemble_train_tomls(OVERRIDES, cycle, K, continue_from=continue_from, round=2)
    c1, c2, c3 = (read_toml(p)["process"]["train"]["continued"] for p in paths)
    assert (c1["enable"], c1["new_training_data"], c1["previous_output_dir"]) == (True, False, str(run_1.resolve()))
    assert c1["load_model_type"] == "latest"  # the ContinuedConfig default
    assert (c2["previous_output_dir"], c2["load_model_type"]) == (str(run_2.resolve()), "best")
    assert c3["enable"] is False


def test_continue_from_per_fold_outputs_dir_with_load_model_type(cycle, tmp_path):
    runs = [make_run(fold_dir(cycle, j) / "outputs_r1") for j in (1, 2)]
    make_run(tmp_path / "abs")
    continue_from = {
        1: {"outputs_dir": "outputs_r1", "load_model_type": "latest"},
        2: {"outputs_dir": fold_dir(cycle, 2) / "outputs_r1", "load_model_type": "best"},
        3: {"outputs_dir": tmp_path / "abs"},
    }
    paths = write_ensemble_train_tomls(OVERRIDES, cycle, K, continue_from=continue_from)
    c1, c2, c3 = (read_toml(p)["process"]["train"]["continued"] for p in paths)
    assert (c1["previous_output_dir"], c1["load_model_type"]) == (str(runs[0].resolve()), "latest")
    assert (c2["previous_output_dir"], c2["load_model_type"]) == (str(runs[1].resolve()), "best")
    assert c3["previous_output_dir"] == str((tmp_path / "abs" / "2026-10-22_10-30-00").resolve())


def test_relative_previous_output_dir_is_resolved_against_the_fold(cycle):
    run = make_run(fold_dir(cycle, 1) / "outputs_r1")
    path = write_ensemble_train_tomls(
        OVERRIDES, cycle, K, folds=[1],
        continue_from={1: {"previous_output_dir": f"outputs_r1/{run.name}"}},
    )[0]
    assert read_toml(path)["process"]["train"]["continued"]["previous_output_dir"] == str(run.resolve())


def test_fold_overrides_carry_the_lr_carry_over_of_a_restart(cycle):
    """The supervisor's retune of one member: new optimizer, LR carried over, rescaled min scale."""
    run = make_run(fold_dir(cycle, 2) / "outputs_r1")
    fold_overrides = {2: {"process": {"train": {
        "optimizer": {"init_learning_rate": 1e-3 * 0.125},
        "scheduler": {"min_learning_rate_scale": 1e-7 / 0.125, "patience": 30, "cooldown": 15},
        "continued": {"new_optimizer": True},
    }}}}
    path = write_ensemble_train_tomls(OVERRIDES, cycle, K, folds=[2], round=2,
                                      continue_from={2: {"outputs_dir": "outputs_r1", "load_model_type": "best"}},
                                      fold_overrides=fold_overrides)[0]
    assert path == fold_dir(cycle, 2) / "train_r2.toml"
    train = read_toml(path)["process"]["train"]
    assert train["optimizer"]["init_learning_rate"] == pytest.approx(1.25e-4)
    assert train["scheduler"]["min_learning_rate_scale"] == pytest.approx(8e-7)
    assert (train["scheduler"]["patience"], train["scheduler"]["cooldown"]) == (30, 15)
    assert train["continued"] == {**train["continued"], "enable": True, "new_optimizer": True,
                                  "load_model_type": "best", "previous_output_dir": str(run.resolve())}


def test_folds_restricts_what_is_written(cycle):
    paths = write_ensemble_train_tomls(OVERRIDES, cycle, K, folds=[3, 1], round=2)
    assert paths == [fold_dir(cycle, 3) / "train_r2.toml", fold_dir(cycle, 1) / "train_r2.toml"]
    assert not (fold_dir(cycle, 2) / "train_r2.toml").exists()
    assert read_toml(paths[0])["system"]["random_seed"] == 100 + 3


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"folds": [0, 1]}, r"folds: folds \[0\] are not among \[1, 2, 3\]"),
        ({"folds": [1, 1]}, "duplicates"),
        ({"fold_overrides": {4: {}}}, r"fold_overrides: folds \[4\]"),
        ({"fold_overrides": {2: {}}, "folds": [1]}, r"fold_overrides: folds \[2\] are not among \[1\]"),
        ({"continue_from": {1: None}}, r"no entry for folds \[2, 3\]"),
        ({"continue_from": {1: None, 2: None, 3: None, 5: None}}, r"continue_from: folds \[5\]"),
        ({"continue_from": {1: {"outputs_dir": "a", "previous_output_dir": "b"}}, "folds": [1]}, "exactly one"),
        ({"continue_from": {1: {"load_model_type": "best"}}, "folds": [1]}, "exactly one"),
        ({"continue_from": {1: {"outputs_dir": "a", "load_model_epoch": 3}}, "folds": [1]}, "unknown keys"),
        ({"continue_from": {1: {"outputs_dir": "a", "load_model_type": "last"}}, "folds": [1]}, "load_model_type"),
    ],
)
def test_bad_per_fold_inputs_are_rejected(cycle, kwargs, match):
    with pytest.raises(ValueError, match=match):
        write_ensemble_train_tomls(OVERRIDES, cycle, K, **kwargs)
    assert not list(cycle.glob("train_*/*.toml"))  # nothing written


def test_continue_from_of_a_wrong_type_is_rejected(cycle):
    with pytest.raises(TypeError, match="continue_from must be"):
        write_ensemble_train_tomls(OVERRIDES, cycle, K, continue_from=["outputs_3"])
    with pytest.raises(TypeError, match=r"continue_from\[1\] must be"):
        write_ensemble_train_tomls(OVERRIDES, cycle, K, folds=[1], continue_from={1: 3})


def test_fold_overrides_of_a_wrong_type_are_rejected(cycle):
    with pytest.raises(TypeError, match=r"fold_overrides\[1\] must be a dict"):
        write_ensemble_train_tomls(OVERRIDES, cycle, K, fold_overrides={1: None})


def test_explicit_previous_output_dir_without_model_is_an_error(cycle, tmp_path):
    with pytest.raises(FileNotFoundError, match="no model/ folder"):
        write_ensemble_train_tomls(OVERRIDES, cycle, K, folds=[1],
                                   continue_from={1: {"previous_output_dir": str(tmp_path / "missing")}})


@pytest.mark.parametrize(
    ("fold_override", "match"),
    [
        ({"process": {"train": {"dataloader": {"train_size": 3}}}}, "train_size"),
        ({"data": {"outputs_dir": "./mine"}}, "outputs_dir"),
        ({"process": {"train": {"continued": {"load_model_type": "latest"}}}}, "load_model_type"),
        ({"process": {"train": {"continued": {"previous_output_dir": "/other"}}}}, "previous_output_dir"),
    ],
)
def test_fold_overrides_must_not_set_what_is_set_per_fold(cycle, fold_override, match):
    make_run(fold_dir(cycle, 1) / "outputs_r1")
    with pytest.raises(ValueError, match=match):
        write_ensemble_train_tomls(OVERRIDES, cycle, K, folds=[1], round=2, fold_overrides={1: fold_override},
                                   continue_from={1: {"outputs_dir": "outputs_r1", "load_model_type": "best"}})
    assert not list(cycle.glob("train_*/*.toml"))


def test_fold_overrides_may_repeat_a_continued_value(cycle):
    make_run(fold_dir(cycle, 1) / "outputs_r1")
    fold_override = {"process": {"train": {"continued": {"load_model_type": "best", "new_optimizer": True}}}}
    path = write_ensemble_train_tomls(OVERRIDES, cycle, K, folds=[1], fold_overrides={1: fold_override},
                                      continue_from={1: {"outputs_dir": "outputs_r1", "load_model_type": "best"}})[0]
    assert read_toml(path)["process"]["train"]["continued"]["new_optimizer"] is True


def test_unknown_keys_in_fold_overrides_are_rejected_before_writing_any_fold(cycle):
    with pytest.raises(KeyError, match="patiense"):
        write_ensemble_train_tomls(OVERRIDES, cycle, K,
                                   fold_overrides={3: {"process": {"train": {"scheduler": {"patiense": 3}}}}})
    assert not list(cycle.glob("train_*/*.toml"))


def test_warm_start_from_a_cycle_trained_in_rounds(cycle, tmp_path):
    """A supervised cycle only has train_r{n}.toml: previous_round says which round to warm-start from."""
    write_ensemble_train_tomls(OVERRIDES, cycle, K, round=1)
    runs = [make_run(train_outputs_dir(cycle, j, round=1)) for j in range(1, K + 1)]
    next_cycle = tmp_path / "cycle2"
    write_kfold_splits([*NAMES, "family9-000"], K, next_cycle, previous_dir=cycle)
    with pytest.raises(FileNotFoundError, match="train.toml"):
        write_ensemble_train_tomls(OVERRIDES, next_cycle, K, previous_cycle_dir=cycle)
    paths = write_ensemble_train_tomls(OVERRIDES, next_cycle, K, previous_cycle_dir=cycle, previous_round=1, round=1)
    assert paths == [fold_dir(next_cycle, j) / "train_r1.toml" for j in range(1, K + 1)]
    for path, run in zip(paths, runs, strict=True):
        continued = read_toml(path)["process"]["train"]["continued"]
        assert (continued["enable"], continued["new_training_data"]) == (True, True)
        assert continued["previous_output_dir"] == str(run.resolve())


def test_previous_round_needs_previous_cycle_dir_and_a_positive_int(cycle, tmp_path):
    with pytest.raises(ValueError, match="pass both"):
        write_ensemble_train_tomls(OVERRIDES, cycle, K, previous_round=1)
    with pytest.raises(ValueError, match="previous_round must be"):
        write_ensemble_train_tomls(OVERRIDES, cycle, K, previous_cycle_dir=tmp_path, previous_round=0)


def test_per_fold_continue_from_can_warm_start_on_new_data(cycle, tmp_path):
    """The per-fold route to a previous cycle's run, e.g. a different round per member."""
    run = make_run(tmp_path / "cycle0" / "train_1" / "outputs_r3")
    path = write_ensemble_train_tomls(
        OVERRIDES, cycle, K, folds=[1],
        continue_from={1: {"previous_output_dir": str(run), "new_training_data": True, "load_model_type": "best"}},
    )[0]
    continued = read_toml(path)["process"]["train"]["continued"]
    assert (continued["new_training_data"], continued["load_model_type"]) == (True, "best")
    with pytest.raises(TypeError, match="new_training_data must be a bool"):
        write_ensemble_train_tomls(OVERRIDES, cycle, K, folds=[1],
                                   continue_from={1: {"previous_output_dir": str(run), "new_training_data": 1}})


def test_fold_overrides_may_repeat_new_training_data(cycle):
    make_run(fold_dir(cycle, 1) / "outputs_r1")
    fold_override = {"process": {"train": {"continued": {"new_training_data": True}}}}
    path = write_ensemble_train_tomls(OVERRIDES, cycle, K, folds=[1], fold_overrides={1: fold_override},
                                      continue_from={1: {"outputs_dir": "outputs_r1", "new_training_data": True}})[0]
    assert read_toml(path)["process"]["train"]["continued"]["new_training_data"] is True
    with pytest.raises(ValueError, match="new_training_data"):
        write_ensemble_train_tomls(OVERRIDES, cycle, K, folds=[1], fold_overrides={1: fold_override},
                                   continue_from={1: "outputs_r1"})


@pytest.mark.parametrize(
    "fold_override",
    [{"process": 5}, {"process": {"train": [1]}}, {"process": {"train": {"continued": "x"}}},
     {"process": {"train": {"dataloader": 3}}}, {"data": "x"}],
)
def test_fold_overrides_with_a_scalar_instead_of_a_table_are_rejected(cycle, fold_override):
    with pytest.raises(TypeError, match=r"fold_overrides\[3\]\..* must be a table"):
        write_ensemble_train_tomls(OVERRIDES, cycle, K, round=2, fold_overrides={3: fold_override})
    assert not list(cycle.glob("train_*/*.toml"))


def test_numpy_integers_are_accepted_as_folds_and_rounds(cycle):
    np = pytest.importorskip("numpy")
    make_run(fold_dir(cycle, 2) / "outputs_r1")
    paths = write_ensemble_train_tomls(
        OVERRIDES, cycle, K, folds=np.array([1, 2]), round=np.int64(2),
        fold_overrides={np.int64(2): {"process": {"train": {"scheduler": {"patience": 7}}}}},
        continue_from={np.int64(1): None, np.int64(2): "outputs_r1"},
    )
    assert paths == [fold_dir(cycle, j) / "train_r2.toml" for j in (1, 2)]
    assert read_toml(paths[1])["process"]["train"]["scheduler"]["patience"] == 7
    assert read_toml(paths[1])["process"]["train"]["continued"]["enable"] is True


@pytest.mark.parametrize("outputs_dir", [".", "./", ""])
def test_round_of_the_fold_folder_itself_goes_to_outputs_rn(cycle, outputs_dir):
    overrides = copy.deepcopy(OVERRIDES)
    overrides["data"]["outputs_dir"] = outputs_dir
    path = write_ensemble_train_tomls(overrides, cycle, K, round=3)[0]
    assert read_toml(path)["data"]["outputs_dir"] == "./outputs_r3"
