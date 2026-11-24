"""examples/training/supervised_training.py builds a valid round-1 flow (its submission is replaced)."""

import importlib.util
import py_compile
from pathlib import Path

import pytest

from hamilflow.training.dataset_split import write_kfold_splits

pytest.importorskip("jobflow_remote")
EXAMPLE = Path(__file__).parents[2] / "examples" / "training" / "supervised_training.py"


def load_example():
    spec = importlib.util.spec_from_file_location("supervised_training_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_example_compiles():
    py_compile.compile(str(EXAMPLE), doraise=True)


def test_the_example_builds_and_submits_round_1(tmp_path, monkeypatch):
    example = load_example()
    cycle = tmp_path / "cycle2"
    write_kfold_splits([f"family{f}-{v:03d}" for f in range(4) for v in range(3)], 2, cycle)
    monkeypatch.setattr(example, "CYCLE_DIR", cycle)
    monkeypatch.setattr(example, "K", 2)
    submitted = []
    monkeypatch.setattr(example, "submit_flow", lambda flow, **kwargs: submitted.append((flow, kwargs)))
    example.main()
    (flow, kwargs), = submitted
    assert kwargs["worker"] == example.LIGHT_WORKER and kwargs["project"] == example.JOBFLOW_PROJECT
    names = [job.name for job in flow.jobs]
    assert names == ["train_member_1_r1", "train_member_2_r1", "supervise_round_r1"]
    train = flow.jobs[0].config.manager_config
    assert train["worker"] == example.GPU_WORKER and train["resources"]["gres"] == "gpu:2"
    assert train["exec_config"] == {"pre_run": example.PRE_RUN}
    assert flow.jobs[2].config.manager_config["worker"] == example.LIGHT_WORKER
    assert (cycle / "train_1" / "train_r1.toml").exists()
    assert example.CONFIG.train_command[-1].endswith("exec deeph-train {toml}")
