"""The wiring of one AL cycle: splits -> per-fold train.toml -> trained run -> per-fold infer.toml."""

import copy
import json
import tomllib

import pytest

from hamilflow.training.dataset_split import write_kfold_splits
from hamilflow.training.ensemble import (
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
