"""The jobflow jobs of the supervisor, end to end with run_locally and the fake deeph-train."""

import json
import os
import signal
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
from synthetic_deeph_logs import decay_then_noise

from hamilflow.training.dataset_split import write_kfold_splits
from hamilflow.training.ensemble import fold_dir
from hamilflow.training.supervisor import SupervisorConfig

jobflow = pytest.importorskip("jobflow")
from hamilflow.training.supervisor import jobs

FAKE = Path(__file__).with_name("fake_deeph_train.py")
NAMES = [f"family{f}-{v:03d}" for f in range(4) for v in range(3)]
BASE = {
    "data": {"inputs_dir": "/inputs", "outputs_dir": "./outputs"},
    "process": {"train": {"scheduler": {"min_learning_rate_scale": 1e-7, "patience": 120, "cooldown": 50},
                          "continued": {"new_optimizer": True}}},  # as in the example: the supervisor overrides it
}
DECAY = [float(v) for v in 10 ** (-2 - 0.01 * np.arange(300))]
STAGES = [float(v) for v in np.repeat(np.array([1.0, 0.5, 0.499, 0.4985, 0.4984]) * 1e-3, 5)]
STAGE_SCALES = [float(v) for v in np.repeat(0.5 ** np.arange(5), 5)]


@pytest.fixture
def cycle(tmp_path, monkeypatch):
    for var in ("SLURM_JOB_END_TIME", "SLURM_JOB_ID"):
        monkeypatch.delenv(var, raising=False)
    cycle_dir = tmp_path / "cycle1"
    write_kfold_splits(NAMES, 2, cycle_dir)
    yield cycle_dir
    for pidfile in tmp_path.glob("pid_*"):  # never leave a fake trainer behind
        try:
            os.killpg(int(pidfile.read_text()), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, ValueError):
            pass


def make_cfg(tmp_path, by_run, **kwargs):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"pidfile": str(tmp_path / "pid_{}"), "by_run": by_run}))
    return SupervisorConfig(**{
        "train_command": [sys.executable, str(FAKE), "{toml}", str(spec)], "poll_s": 0.05, "stop_grace": 0.5,
        "kill_grace": 0.3, "checkpoint_settle_s": 0.05, **kwargs,
    })


WORKERS = {"train_worker": "gpu", "supervise_worker": "login"}


def run(cycle, cfg, **maker):
    flow = jobs.TrainingSupervisorMaker(cfg, BASE, 2, **{**WORKERS, **maker}).make(cycle)
    responses = jobflow.run_locally(flow, ensure_success=False)
    return flow, responses, jobs.read_state(cycle)


def toml(cycle, fold, round_):
    with open(fold_dir(cycle, fold) / f"train_r{round_}.toml", "rb") as f:
        return tomllib.load(f)


def replaced_flows(responses):
    return [r.replace for by_index in responses.values() for r in by_index.values() if r.replace is not None]


# ---------------------------------- files ----------------------------------


def test_state_is_written_atomically(tmp_path):
    path = jobs.write_state(tmp_path, {"round": 1})
    assert json.loads(path.read_text()) == {"round": 1}
    with pytest.raises(TypeError):
        jobs.write_state(tmp_path, {"round": object()})
    assert json.loads(path.read_text()) == {"round": 1}  # the old state is intact
    assert [p.name for p in path.parent.iterdir()] == ["state.json"]  # no temporary file left


def test_round_flow_sets_the_run_configs(tmp_path):
    state = {"name": "sup", "cfg": SupervisorConfig().to_dict(), "base_overrides": BASE,
             "members": {"1": {"logs": ["/a/deepx.log"]}, "2": {"logs": []}},
             "run_config": {"train": {"worker": "gpu", "resources": {"gres": "gpu:2"}, "exec_config": "deeph"},
                            "supervise": {"worker": "login", "resources": None, "exec_config": None}}}
    flow = jobs.round_flow(tmp_path, 3, state, [1, 2], {1: {"retune": "off"}})
    train_1, train_2, supervise = flow.jobs
    assert train_1.name == "train_member_1_r3"
    assert train_1.config.manager_config == {"worker": "gpu", "resources": {"gres": "gpu:2"}, "exec_config": "deeph"}
    assert train_1.function_kwargs["member_logs_before"] == ["/a/deepx.log"]
    assert train_1.function_kwargs["live_actions"] == {"retune": "off"}
    assert train_2.function_kwargs["live_actions"] == {}
    assert supervise.name == "supervise_round_r3"
    assert supervise.config.manager_config == {"worker": "login"}
    assert supervise.config.on_missing_references == jobflow.OnMissing.NONE
    assert supervise.config.pass_manager_config is False


def test_make_refuses_a_cycle_already_supervised(cycle, tmp_path):
    cfg = make_cfg(tmp_path, {})
    jobs.TrainingSupervisorMaker(cfg, BASE, 2, **WORKERS).make(cycle)
    with pytest.raises(FileExistsError, match="already supervised"):
        jobs.TrainingSupervisorMaker(cfg, BASE, 2, **WORKERS).make(cycle)


@pytest.mark.parametrize("missing", ["train_worker", "supervise_worker"])
def test_make_needs_both_workers(cycle, tmp_path, missing):
    """Without a train worker, jobflow-remote would give round 2's train jobs the supervise job's worker."""
    workers = {**WORKERS, missing: None}
    with pytest.raises(ValueError, match=missing):
        jobs.TrainingSupervisorMaker(make_cfg(tmp_path, {}), BASE, 2, **workers).make(cycle)
    assert not jobs.state_path(cycle).exists()


def test_make_writes_round_1_and_the_state(cycle, tmp_path):
    cfg = make_cfg(tmp_path, {})
    flow = jobs.TrainingSupervisorMaker(cfg, BASE, 2, train_worker="gpu", resources_train={"gres": "gpu:2"},
                                        supervise_worker="login", project="p").make(cycle)
    assert [j.name for j in flow.jobs] == ["train_member_1_r1", "train_member_2_r1", "supervise_round_r1"]
    written = toml(cycle, 1, 1)
    assert written["data"]["outputs_dir"] == "./outputs_r1"
    assert written["data"]["model_save"]["latest_interval"] == 10  # cfg.shared_overrides()
    state = jobs.read_state(cycle)
    assert (state["status"], state["round"], state["k"], state["project"]) == ("running", 1, 2, "p")
    assert state["run_config"]["train"] == {"worker": "gpu", "resources": {"gres": "gpu:2"}, "exec_config": None}
    assert state["history"]["start"] == {"1": {"continue_from": None, "fold_overrides": {}},
                                         "2": {"continue_from": None, "fold_overrides": {}}}
    assert state["members"] == {"1": {"logs": [], "runs": []}, "2": {"logs": [], "runs": []}}
    assert SupervisorConfig.from_dict(state["cfg"]) == cfg


def test_warm_start_records_round_1s_start_and_no_earlier_chain(cycle, tmp_path):
    previous = tmp_path / "cycle0"
    write_kfold_splits(NAMES, 2, previous)
    jobs.write_ensemble_train_tomls(BASE, previous, 2)
    for j in (1, 2):
        (fold_dir(previous, j) / "outputs" / "2026-10-22_10-30-00" / "model").mkdir(parents=True)
    jobs.TrainingSupervisorMaker(make_cfg(tmp_path, {}), BASE, 2, **WORKERS).make(cycle, previous_cycle_dir=previous)
    state = jobs.read_state(cycle)
    start = state["history"]["start"]["1"]["continue_from"]
    assert start["new_training_data"] is True
    assert start["previous_output_dir"].endswith("train_1/outputs/2026-10-22_10-30-00")
    assert state["members"]["1"]["logs"] == []  # the chain starts with this cycle


def test_proposal_round_trips(tmp_path):
    from hamilflow.training.supervisor.policy import MemberAction, RoundDecision

    decision = RoundDecision(2, {
        1: MemberAction(1, "retune", "mistuned", "why", "mistuned", "retune", "ask", True,
                        {"previous_output_dir": "/r", "load_model_type": "best"},
                        {"process": {"train": {"scheduler": {"patience": 25}}}}, live_actions={}),
        2: MemberAction(2, "resume", "crash", "crash", "crashed", "resume", fold_overrides={}),
        3: MemberAction(3, "done", "converged", "flat", "converged", "stop_converged"),
    }, {"data": {"model_save": {"latest_interval": 10}}})
    path = jobs.write_proposal(tmp_path / "proposal.toml", decision)
    proposal = jobs.read_proposal(path)
    assert proposal["round"] == 3 and proposal["after_round"] == 2
    assert proposal["members"][1]["continue_from"] == {"previous_output_dir": "/r", "load_model_type": "best"}
    assert proposal["members"][1]["fold_overrides"] == {"process": {"train": {"scheduler": {"patience": 25}}}}
    assert proposal["members"][2]["continue_from"] is None  # from scratch
    assert proposal["members"][3]["action"] == "done"
    path.write_text(path.read_text().replace('action = "done"', 'action = "retrain"'))
    with pytest.raises(ValueError, match="action must be one of"):
        jobs.read_proposal(path)


# ---------------------------------- end to end ----------------------------------


@pytest.mark.slow
def test_walltime_stop_then_resume_then_converged(cycle, tmp_path):
    by_run = {}
    for j in (1, 2):
        by_run[f"train_{j}/outputs_r1"] = {"n_epochs": 10000, "epoch_s": 0.02, "val": DECAY}
        by_run[f"train_{j}/outputs_r2"] = {"n_epochs": 60, "epoch_s": 0.01, "val": STAGES, "scale": STAGE_SCALES}
    cfg = make_cfg(tmp_path, by_run, walltime=2.0, walltime_margin=1.0)
    _, responses, state = run(cycle, cfg, train_worker="gpu", resources_train={"gres": "gpu:2"},
                              supervise_worker="login")
    rounds = state["history"]["rounds"]
    assert [r["round"] for r in rounds] == [1, 2]
    assert {r["members"]["1"]["action"] for r in rounds} == {"resume", "done"}
    assert [rounds[0]["members"][j]["cause"] for j in ("1", "2")] == ["walltime", "walltime"]
    assert [rounds[1]["members"][j]["cause"] for j in ("1", "2")] == ["converged", "converged"]
    assert state["status"] == "finished"
    # round 2 resumed from round 1's latest with the optimizer kept, the user's floor and the supervisor's interval
    r2 = toml(cycle, 1, 2)
    assert r2["process"]["train"]["continued"]["load_model_type"] == "latest"
    assert r2["process"]["train"]["continued"]["new_optimizer"] is False
    assert r2["process"]["train"]["continued"]["previous_output_dir"] == state["members"]["1"]["runs"][0]["run_dir"]
    assert r2["process"]["train"]["scheduler"]["min_learning_rate_scale"] == pytest.approx(1e-7)
    assert r2["data"]["model_save"]["latest_interval"] == 10
    # the chains and run dirs come from the ExitInfos
    member = state["members"]["1"]
    assert [Path(p).parent.parent.name for p in member["logs"]] == ["outputs_r1", "outputs_r2"]
    assert [run["exit"]["reason"] for run in member["runs"]] == ["walltime", "live_rule"]
    final = jobs.summary(state)
    assert final["members"]["1"]["final_run_dir"] == member["runs"][-1]["run_dir"]
    assert (jobs.round_dir(cycle, 1) / "report.txt").exists() and (jobs.round_dir(cycle, 2) / "decision.json").exists()
    # the replacement jobs got the GPU run config again, the supervise job and store_inputs the light worker
    (replacement,) = replaced_flows(responses)
    configs = {j.name: j.config.manager_config for j in replacement.jobs}
    assert configs["train_member_1_r2"] == {"worker": "gpu", "resources": {"gres": "gpu:2"}}
    assert configs["supervise_round_r2"] == {"worker": "login"}
    assert configs["store_inputs"] == {}  # jobflow-remote gives it the supervise job's worker


def test_retune_round_writes_the_carried_lr_new_patience_and_the_users_floor(cycle, tmp_path):
    val = [float(v) for v in decay_then_noise(280)]
    by_run = {f"train_{j}/outputs_r1": {"n_epochs": 280, "epoch_s": 0.001, "val": val} for j in (1, 2)}
    by_run.update({f"train_{j}/outputs_r2": {"n_epochs": 5} for j in (1, 2)})
    # W3 off live (min_epochs) so both members finish round 1: the retune comes from the pooled suggestion
    cfg = make_cfg(tmp_path, by_run, actions={"retune": "auto"}, min_epochs_before_rules=10000, max_rounds=2)
    _, _, state = run(cycle, cfg)
    first = state["history"]["rounds"][0]["members"]
    assert [first[j]["action"] for j in ("1", "2")] == ["retune", "retune"]
    r2 = toml(cycle, 1, 2)["process"]["train"]
    suggested = r2["scheduler"]["patience"]
    assert suggested != 120 and suggested == first["1"]["fold_overrides"]["process"]["train"]["scheduler"]["patience"]
    lr0 = toml(cycle, 1, 1)["process"]["train"]["optimizer"]["init_learning_rate"]
    assert r2["optimizer"]["init_learning_rate"] == pytest.approx(lr0)  # lr0 x last scale 1.0
    assert r2["scheduler"]["min_learning_rate_scale"] == pytest.approx(1e-7)  # the user's floor
    assert (r2["continued"]["new_optimizer"], r2["continued"]["load_model_type"]) == (True, "best")
    assert r2["continued"]["previous_output_dir"] == state["members"]["1"]["runs"][0]["run_dir"]
    # max_rounds 2: given up after round 2
    assert [m["action"] for m in state["history"]["rounds"][1]["members"].values()] == ["give_up", "give_up"]
    assert state["status"] == "finished"


def test_ask_round_stops_with_a_proposal(cycle, tmp_path):
    val = [float(v) for v in decay_then_noise(280)]
    by_run = {f"train_{j}/outputs_r1": {"n_epochs": 280, "epoch_s": 0.001, "val": val} for j in (1, 2)}
    cfg = make_cfg(tmp_path, by_run, min_epochs_before_rules=10000)  # retune: ask (default)
    _, responses, state = run(cycle, cfg)
    assert state["status"] == "awaiting_approval"
    assert not replaced_flows(responses)
    assert not (fold_dir(cycle, 1) / "train_r2.toml").exists()
    proposal = jobs.read_proposal(jobs.round_dir(cycle, 1) / "proposal.toml")
    assert proposal["round"] == 2
    assert proposal["members"][1]["action"] == "retune" and proposal["members"][1]["needs_approval"]
    decided = state["history"]["rounds"][0]["members"]["1"]
    assert proposal["members"][1]["fold_overrides"] == decided["fold_overrides"]
    assert proposal["members"][1]["continue_from"] == decided["continue_from"]
    outputs = [r.output for by_index in responses.values() for r in by_index.values()
               if isinstance(r.output, dict) and r.output.get("status") == "awaiting_approval"]
    assert outputs and outputs[0]["proposal"].endswith("round_1/proposal.toml")


def test_a_failed_train_job_is_a_crash_the_supervise_job_restarts(cycle, tmp_path, monkeypatch):
    real = jobs.run_supervised

    def failing(toml_path, cwd, *args, **kwargs):
        if Path(cwd).name == "train_2" and Path(toml_path).name == "train_r1.toml":
            raise RuntimeError("node failure")
        return real(toml_path, cwd, *args, **kwargs)

    monkeypatch.setattr(jobs, "run_supervised", failing)
    by_run = {"train_1/outputs_r1": {"n_epochs": 10000, "epoch_s": 0.02, "val": DECAY}}
    by_run.update({f"train_{j}/outputs_r2": {"n_epochs": 5} for j in (1, 2)})
    cfg = make_cfg(tmp_path, by_run, walltime=2.0, walltime_margin=1.0, max_rounds=2)
    _, _, state = run(cycle, cfg)
    first = state["history"]["rounds"][0]["members"]
    assert (first["2"]["action"], first["2"]["cause"]) == ("resume", "crash")
    assert first["2"]["continue_from"] is None  # round 1 started from scratch: so again
    assert state["members"]["2"]["runs"][0] == {"round": 1, "run_dir": None, "exit": None, "log": None}
    assert toml(cycle, 2, 2)["process"]["train"]["continued"]["enable"] is False
    assert (first["1"]["action"], first["1"]["cause"]) == ("resume", "walltime")
    assert len(state["history"]["rounds"]) == 2 and state["status"] == "finished"


# ---------------------------------- reruns, final run dir, complete TOMLs ----------------------------------


def first_round(cycle, tmp_path, **cfg_kwargs):
    """Round 1 trained by hand (no flow): both members train 30 epochs, reported as stopped at the walltime
    (so round 2 resumes them); returns their ExitInfo dicts."""
    by_run = {f"train_{j}/outputs_r1": {"n_epochs": 30, "epoch_s": 0.002, "val": DECAY} for j in (1, 2)}
    cfg = make_cfg(tmp_path, by_run, **cfg_kwargs)
    jobs.TrainingSupervisorMaker(cfg, BASE, 2, **WORKERS).make(cycle)
    infos = {}
    for j in (1, 2):
        folder = fold_dir(cycle, j)
        infos[str(j)] = jobs.run_supervised(folder / "train_r1.toml", folder, [], cfg, jobs.round_dir(cycle, 1), j,
                                            BASE, end_time=None).to_dict()
        infos[str(j)]["reason"] = "walltime"
    return infos


def without_timestamps(state):
    return json.loads(json.dumps(state))


def test_a_rerun_supervise_job_records_its_round_once(cycle, tmp_path):
    infos = first_round(cycle, tmp_path)
    jobs.supervise_round_job.original(str(cycle), 1, infos)
    once = without_timestamps(jobs.read_state(cycle))
    toml_once = (fold_dir(cycle, 1) / "train_r2.toml").read_text()
    jobs.supervise_round_job.original(str(cycle), 1, infos)  # as jf job rerun would
    twice = jobs.read_state(cycle)
    assert twice == once
    assert [r["round"] for r in twice["history"]["rounds"]] == [1]
    assert [run["round"] for run in twice["members"]["1"]["runs"]] == [1]
    assert len(twice["members"]["1"]["logs"]) == 1
    assert (fold_dir(cycle, 1) / "train_r2.toml").read_text() == toml_once


def test_a_rerun_after_a_failure_past_write_state_decides_the_round_again(cycle, tmp_path, monkeypatch):
    infos = first_round(cycle, tmp_path)
    real_flow = jobs.round_flow
    monkeypatch.setattr(jobs, "round_flow", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("lost the DB")))
    with pytest.raises(RuntimeError, match="lost the DB"):
        jobs.supervise_round_job.original(str(cycle), 1, infos)
    assert jobs.read_state(cycle)["round"] == 2  # recorded before the failure
    monkeypatch.setattr(jobs, "round_flow", real_flow)
    response = jobs.supervise_round_job.original(str(cycle), 1, infos)
    state = jobs.read_state(cycle)
    assert [r["round"] for r in state["history"]["rounds"]] == [1]
    assert state["round"] == 2 and response.output == {"status": "running", "round": 2}


def test_a_rerun_after_a_failure_before_write_state(cycle, tmp_path, monkeypatch):
    infos = first_round(cycle, tmp_path)
    real_write = jobs.write_ensemble_train_tomls
    monkeypatch.setattr(jobs, "write_ensemble_train_tomls",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        jobs.supervise_round_job.original(str(cycle), 1, infos)
    assert jobs.read_state(cycle)["history"]["rounds"] == []
    monkeypatch.setattr(jobs, "write_ensemble_train_tomls", real_write)
    jobs.supervise_round_job.original(str(cycle), 1, infos)
    assert [r["round"] for r in jobs.read_state(cycle)["history"]["rounds"]] == [1]


def test_round_decision_checks_the_round_against_the_history(tmp_path):
    from hamilflow.training.supervisor import round_decision

    with pytest.raises(ValueError, match="history holds 0 rounds"):
        round_decision({}, {}, SupervisorConfig(), {"rounds": []}, round_=2)


def test_final_run_dir_is_the_lineages_best_run_with_a_best_checkpoint(tmp_path):
    def run_with(name, val, best=True, latest=False):
        run = tmp_path / name
        (run / "model").mkdir(parents=True)
        from synthetic_deeph_logs import write_log

        write_log(run / "deepx.log", np.ones(len(val)), val=val)
        for kind, present in (("best", best), ("latest", latest)):
            if present:
                for part in ("params", "states"):
                    path = run / "model" / part / f"{kind}.pytree" / "epoch_1"
                    path.mkdir(parents=True)
                    (path / "_CHECKPOINT_METADATA").write_text("{}")
        return run

    good = run_with("2026-01-01_00-00-00", [3.0, 1.0])
    crashed = run_with("2026-01-02_00-00-00", [2.0], best=False)  # a later crash without a checkpoint
    member = {"logs": [str(good / "deepx.log"), str(crashed / "deepx.log")],
              "runs": [{"round": 1, "run_dir": str(good)}, {"round": 2, "run_dir": str(crashed)}]}
    assert jobs.final_run_dir(member) == str(good)
    # the best val in a run without a best checkpoint: the latest run with any checkpoint
    worse = run_with("2026-01-03_00-00-00", [0.5], best=False, latest=True)
    member = {"logs": [str(good / "deepx.log"), str(worse / "deepx.log")],
              "runs": [{"round": 1, "run_dir": str(good)}, {"round": 2, "run_dir": str(worse)}]}
    (worse / "model" / "params" / "latest.pytree" / "epoch_1" / "_CHECKPOINT_METADATA").unlink()
    assert jobs.final_run_dir(member) == str(good)
    assert jobs.final_run_dir({"logs": [], "runs": [{"round": 1, "run_dir": None}]}) is None


def test_every_written_toml_has_the_full_continued_table(cycle, tmp_path):
    """DeepH reads all six keys of [process.train.continued], even with enable = false."""
    six = {"enable", "new_training_data", "new_optimizer", "previous_output_dir", "load_model_type", "load_model_epoch"}
    run_dir = fold_dir(cycle, 1) / "outputs_r1" / "2026-01-01_00-00-00"
    (run_dir / "model").mkdir(parents=True)
    paths = [
        *jobs.write_ensemble_train_tomls(BASE, cycle, 2),
        *jobs.write_ensemble_train_tomls(BASE, cycle, 2, round=2, continue_from={1: str(run_dir.parent), 2: None},
                                         fold_overrides={1: {"process": {"train": {"continued": {
                                             "new_optimizer": False, "load_model_epoch": 10}}}}}),
    ]
    other = tmp_path / "cycle2"
    write_kfold_splits(NAMES, 2, other)
    jobs.TrainingSupervisorMaker(make_cfg(tmp_path, {}), {"data": BASE["data"]}, 2, **WORKERS).make(other)
    paths += [fold_dir(other, j) / "train_r1.toml" for j in (1, 2)]
    for path in paths:
        with open(path, "rb") as f:
            assert set(tomllib.load(f)["process"]["train"]["continued"]) == six, path


def test_final_run_dir_needs_only_a_complete_params_best(tmp_path):
    """The reviewer's case from run 17142370: run 1 holds the best val with params best 9 complete but
    states best 8 + a tmp dir; run 2 continues from it with a worse val and a matching best."""
    from synthetic_deeph_logs import KEPT_OPTIMIZER, write_log

    def checkpoint(run, part, epoch):
        path = run / "model" / part / "best.pytree" / f"epoch_{epoch}"
        path.mkdir(parents=True)
        (path / "_CHECKPOINT_METADATA").write_text("{}")

    run_1, run_2 = tmp_path / "2026-11-24_10-19-48", tmp_path / "2026-11-24_10-30-00"
    write_log(run_1 / "deepx.log", np.ones(9), val=np.linspace(9, 1, 9))
    checkpoint(run_1, "params", 9)
    checkpoint(run_1, "states", 8)
    (run_1 / "model" / "states" / "best.pytree" / "epoch_9.orbax-checkpoint-tmp-0").mkdir()
    write_log(run_2 / "deepx.log", np.ones(3), val=[2.0, 1.5, 1.2], continued_from=f"/x/{run_1.name}",
              load_model_type="latest", continuation=KEPT_OPTIMIZER, first_epoch=10)
    for part in ("params", "states"):
        checkpoint(run_2, part, 12)
    member = {"logs": [str(run_1 / "deepx.log"), str(run_2 / "deepx.log")],
              "runs": [{"round": 1, "run_dir": str(run_1)}, {"round": 2, "run_dir": str(run_2)}]}
    assert jobs.final_run_dir(member) == str(run_1)


# ---------------------------------- warm start of the next AL cycle from a supervised cycle ----------------------


@pytest.mark.slow
def test_warm_start_from_a_supervised_cycle_whose_members_ended_in_different_rounds(cycle, tmp_path):
    by_run = {
        "train_1/outputs_r1": {"n_epochs": 60, "epoch_s": 0.01, "val": STAGES, "scale": STAGE_SCALES},  # done in r1
        "train_2/outputs_r1": {"n_epochs": 10000, "epoch_s": 0.02, "val": DECAY},  # walltime, then r2
        "train_2/outputs_r2": {"n_epochs": 60, "epoch_s": 0.01, "val": STAGES, "scale": STAGE_SCALES},
    }
    _, _, state = run(cycle, make_cfg(tmp_path, by_run, walltime=2.0, walltime_margin=1.0))
    assert state["status"] == "finished"
    assert [len(state["members"][j]["runs"]) for j in ("1", "2")] == [1, 2]  # different last rounds
    final = json.loads(jobs.summary_path(cycle).read_text())

    # previous_round only works when every member ended in that round
    next_cycle = tmp_path / "cycle2"
    write_kfold_splits([*NAMES, "family9-000"], 2, next_cycle, previous_dir=cycle)
    with pytest.raises(FileNotFoundError):
        jobs.write_ensemble_train_tomls(BASE, next_cycle, 2, previous_cycle_dir=cycle, previous_round=2)

    warm = jobs.warm_start_from(cycle)
    assert warm == {j: {"previous_output_dir": final["members"][str(j)]["final_run_dir"], "load_model_type": "best",
                        "new_training_data": True, "new_optimizer": True} for j in (1, 2)}
    jobs.TrainingSupervisorMaker(make_cfg(tmp_path, {}), BASE, 2, **WORKERS).make(next_cycle, continue_from=warm)
    for j in (1, 2):
        continued = toml(next_cycle, j, 1)["process"]["train"]["continued"]
        assert continued["enable"] and continued["new_training_data"] and continued["new_optimizer"]
        assert continued["load_model_type"] == "best"
        assert continued["previous_output_dir"] == str(Path(final["members"][str(j)]["final_run_dir"]).resolve())
    start = jobs.read_state(next_cycle)["history"]["start"]["1"]["continue_from"]
    assert (start["new_optimizer"], start["new_training_data"], start["load_model_type"]) == (True, True, "best")
    assert jobs.read_state(next_cycle)["members"]["1"]["logs"] == []


def test_warm_start_from_reads_the_state_without_a_summary_and_refuses_members_without_a_run(tmp_path):
    cycle = tmp_path / "prev"
    run_dir = tmp_path / "prev" / "train_1" / "outputs_r1" / "2026-01-01_00-00-00"
    path = run_dir / "model" / "params" / "best.pytree" / "epoch_3"
    path.mkdir(parents=True)
    (path / "_CHECKPOINT_METADATA").write_text("{}")
    state = {"status": "finished", "round": 1, "cycle_dir": str(cycle), "history": {"rounds": []},
             "members": {"1": {"logs": [], "runs": [{"round": 1, "run_dir": str(run_dir)}]},
                         "2": {"logs": [], "runs": [{"round": 1, "run_dir": None}]}}}
    jobs.write_state(cycle, state)
    with pytest.raises(ValueError, match=r"members \[2\]"):
        jobs.warm_start_from(cycle)
    state["members"].pop("2")
    jobs.write_state(cycle, state)
    assert jobs.warm_start_from(cycle)[1]["previous_output_dir"] == str(run_dir)


def test_make_refuses_to_load_best_with_the_optimizer_kept(cycle, tmp_path):
    runs = []
    for j in (1, 2):
        run_dir = tmp_path / f"prev_{j}" / "2026-01-01_00-00-00"
        (run_dir / "model").mkdir(parents=True)
        runs.append(run_dir)
    kept = {j: {"previous_output_dir": str(runs[j - 1]), "load_model_type": "best", "new_training_data": True}
            for j in (1, 2)}
    with pytest.raises(ValueError, match="optimizer kept"):
        jobs.TrainingSupervisorMaker(make_cfg(tmp_path, {}), {"data": BASE["data"]}, 2, **WORKERS).make(
            cycle, continue_from=kept)
    assert not jobs.state_path(cycle).exists()


def test_final_run_dir_never_falls_back_to_a_run_without_a_best(tmp_path):
    from synthetic_deeph_logs import write_log

    only_latest = tmp_path / "2026-01-01_00-00-00"
    write_log(only_latest / "deepx.log", np.ones(2), val=[2.0, 1.0])
    path = only_latest / "model" / "params" / "latest.pytree" / "epoch_2"
    path.mkdir(parents=True)
    (path / "_CHECKPOINT_METADATA").write_text("{}")
    member = {"logs": [str(only_latest / "deepx.log")], "runs": [{"round": 1, "run_dir": str(only_latest)}]}
    assert jobs.final_run_dir(member) is None  # inference loads best
