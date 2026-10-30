"""Writing train.toml and infer.toml: validation, overrides and the DeepH-pack constraints."""

import copy
import tomllib

import pytest

from hamilflow.training.write_train_infer_toml import (
    InferConfig,
    TrainConfig,
    apply_overrides,
    validate_keys,
    write_infer_toml,
    write_train_toml,
)

# The settings of a real training run, without its paths
REAL_OVERRIDES = {
    "system": {"device": "gpu*2:0-1", "float_type": "tf32", "random_seed": 3971, "jax_memory_preallocate": False},
    "data": {
        "inputs_dir": "/inputs",
        "outputs_dir": "./outputs_4",
        "graph": {
            "dataset_name": "tbmos2",
            "graph_type": "HS",
            "storage_type": "disk",
            "common_orbital_types": "s2p2d2f1",
        },
    },
    "model": {
        "net_type": "eagle",
        "advanced": {
            "gaussian_basis_rmax": 12.0,
            "net_irreps": "16x0e+16x1e+20x2e+16x3e+8x4e+4x5e+2x6e",
            "consider_parity": False,
            "standardize_gauge": True,
        },
    },
    "process": {
        "train": {
            "optimizer": {"init_learning_rate": 0.001},
            "scheduler": {"min_learning_rate_scale": 1e-07, "patience": 20, "cooldown": 10},
        }
    },
}


def read_toml(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def test_real_settings_are_written_and_read_back(tmp_path):
    written = read_toml(write_train_toml(REAL_OVERRIDES, tmp_path))
    assert written["model"]["net_type"] == "eagle"
    assert written["model"]["advanced"]["net_irreps"] == "16x0e+16x1e+20x2e+16x3e+8x4e+4x5e+2x6e"
    assert written["data"]["graph"]["common_orbital_types"] == "s2p2d2f1"
    assert written["process"]["train"]["scheduler"]["patience"] == 20


def test_keys_not_overridden_keep_their_defaults(tmp_path):
    written = read_toml(write_train_toml(REAL_OVERRIDES, tmp_path))
    assert written["process"]["train"]["scheduler"]["factor"] == 0.5
    assert written["data"]["model_save"]["latest_interval"] == 100
    assert written["process"]["train"]["optimizer"]["betas"] == [0.9, 0.999]


def test_writer_creates_folders_and_honours_filename(tmp_path):
    path = write_train_toml({}, tmp_path / "a" / "b", filename="other.toml")
    assert path == tmp_path / "a" / "b" / "other.toml"
    assert path.is_file()


def test_writer_does_not_change_the_overrides(tmp_path):
    overrides = copy.deepcopy(REAL_OVERRIDES)
    write_train_toml(overrides, tmp_path)
    assert overrides == REAL_OVERRIDES


@pytest.mark.parametrize(
    ("overrides", "full_key"),
    [
        ({"nonsense": 1}, "nonsense"),
        ({"data": {"nonsense": 1}}, "data.nonsense"),
        ({"process": {"train": {"scheduler": {"patiense": 1}}}}, "process.train.scheduler.patiense"),
    ],
)
def test_unknown_key_is_reported_with_its_full_path(overrides, full_key):
    with pytest.raises(KeyError, match=full_key):
        validate_keys(TrainConfig, overrides)


@pytest.mark.parametrize(
    ("overrides", "full_key"),
    [
        ({"model": {"net_type": "owl"}}, "model.net_type"),
        ({"system": {"float_type": "fp16"}}, "system.float_type"),
        ({"process": {"train": {"scheduler": {"type": "cosine"}}}}, "process.train.scheduler.type"),
    ],
)
def test_value_outside_the_allowed_set_is_rejected(overrides, full_key):
    with pytest.raises(ValueError, match=full_key):
        validate_keys(TrainConfig, overrides)


def test_table_given_for_a_scalar_is_rejected():
    with pytest.raises(TypeError, match="system.device"):
        validate_keys(TrainConfig, {"system": {"device": {"name": "gpu"}}})


@pytest.mark.parametrize(
    ("advanced", "net_type", "message"),
    [
        ({"consider_parity": True}, "eagle", "requires model.advanced.consider_parity=False"),
        ({"consider_parity": False, "net_irreps": "8x0e+4x1o"}, "normal", "forbids odd-parity"),
        ({"standardize_gauge": True}, "normal", "requires data.graph.graph_type='HS'"),
    ],
)
def test_train_constraints_are_enforced(tmp_path, advanced, net_type, message):
    overrides = {"model": {"net_type": net_type, "advanced": advanced}}
    with pytest.raises(ValueError, match=message):
        write_train_toml(overrides, tmp_path)
    assert not (tmp_path / "train.toml").exists()


def test_odd_parity_is_allowed_when_parity_is_considered(tmp_path):
    overrides = {"model": {"advanced": {"consider_parity": True, "net_irreps": "8x0e + 4x1o"}}}
    assert read_toml(write_train_toml(overrides, tmp_path))["model"]["advanced"]["net_irreps"] == "8x0e + 4x1o"


def test_odd_parity_term_is_found_anywhere_in_the_irreps(tmp_path):
    overrides = {"model": {"advanced": {"consider_parity": False, "net_irreps": "4x1o + 8x0e"}}}
    with pytest.raises(ValueError, match="forbids odd-parity"):
        write_train_toml(overrides, tmp_path)


def test_apply_overrides_leaves_the_defaults_alone():
    default = TrainConfig()
    changed = apply_overrides(default, {"process": {"train": {"optimizer": {"betas": [0.5, 0.5]}}}})
    assert changed.process.train.optimizer.betas == [0.5, 0.5]
    assert default.process.train.optimizer.betas == [0.9, 0.999]
    assert TrainConfig().process.train.optimizer.betas == [0.9, 0.999]


def test_infer_toml_has_infer_defaults_and_no_training_keys(tmp_path):
    written = read_toml(write_infer_toml({"model": {"model_dir": "/model"}}, tmp_path))
    assert written["model"]["model_dir"] == "/model"
    assert written["data"]["graph"]["graph_type"] == "S"
    assert "common_orbital_types" not in written["data"]["graph"]
    assert "model_save" not in written["data"]
    assert "train" not in written["process"]
    assert written["process"]["infer"]["output_type"] == "h5"


@pytest.mark.parametrize("graph_type", ["H", "HS", "Rho"])
def test_infer_rejects_training_graph_types(tmp_path, graph_type):
    with pytest.raises(ValueError, match="data.graph.graph_type"):
        write_infer_toml({"data": {"graph": {"graph_type": graph_type}}}, tmp_path)


def test_infer_rejects_training_only_keys():
    with pytest.raises(KeyError, match="common_orbital_types"):
        validate_keys(InferConfig, {"data": {"graph": {"common_orbital_types": "s1"}}})
