"""The ``hamilflow-train-supervisor`` CLI, on states from run_locally rounds with the fake trainer."""

import io
import json
import os
import signal
import stat
import sys
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from synthetic_deeph_logs import decay_then_noise

from hamilflow.training.dataset_split import write_kfold_splits
from hamilflow.training.ensemble import fold_dir
from hamilflow.training.supervisor import SupervisorConfig

jobflow = pytest.importorskip("jobflow")
from hamilflow.training.supervisor import cli, jobs

FAKE = Path(__file__).with_name("fake_deeph_train.py")
NAMES = [f"family{f}-{v:03d}" for f in range(4) for v in range(3)]
BASE = {"data": {"inputs_dir": "/inputs", "outputs_dir": "./outputs"},
        "process": {"train": {"scheduler": {"min_learning_rate_scale": 1e-7, "patience": 120, "cooldown": 50}}}}
WORKERS = {"train_worker": "gpu", "supervise_worker": "login", "resources_train": {"gres": "gpu:2"}, "project": "p"}


@pytest.fixture
def tmp(tmp_path, monkeypatch):
    for var in ("SLURM_JOB_END_TIME", "SLURM_JOB_ID"):
        monkeypatch.delenv(var, raising=False)
    yield tmp_path
    for pidfile in tmp_path.glob("pid_*"):
        try:
            os.killpg(int(pidfile.read_text()), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, ValueError):
            pass


def supervised(tmp_path, **cfg_kwargs):
    """Round 1 of a 2-member cycle, run with run_locally: both members train 280 epochs to a plateau with
    patience 120, so the pooled suggestion asks for a retune (ask by default)."""
    cycle = tmp_path / "cycle1"
    write_kfold_splits(NAMES, 2, cycle)
    val = [float(v) for v in decay_then_noise(280)]
    by_run = {f"train_{j}/outputs_r1": {"n_epochs": 280, "epoch_s": 0.001, "val": val} for j in (1, 2)}
    by_run.update({f"train_{j}/outputs_r2": {"n_epochs": 5} for j in (1, 2)})
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"pidfile": str(tmp_path / "pid_{}"), "by_run": by_run}))
    cfg = SupervisorConfig(**{
        "train_command": [sys.executable, str(FAKE), "{toml}", str(spec)], "poll_s": 0.05, "stop_grace": 0.5,
        "kill_grace": 0.3, "checkpoint_settle_s": 0.05, "min_epochs_before_rules": 10000, **cfg_kwargs})
    flow = jobs.TrainingSupervisorMaker(cfg, BASE, 2, **WORKERS).make(cycle)
    jobflow.run_locally(flow, ensure_success=False)
    return cycle


@pytest.fixture
def awaiting(tmp):
    cycle = supervised(tmp)
    assert jobs.read_state(cycle)["status"] == "awaiting_approval"
    return cycle


def run_cli(*argv, **kwargs):
    out = io.StringIO()
    code = cli.main([str(a) for a in argv], out=out, **kwargs)
    return code, out.getvalue()


class Submitted:
    def __init__(self):
        self.calls = []

    def __call__(self, flow, worker=None, project=None):
        self.calls.append((flow, worker, project))
        return [flow.uuid]


# ---------------------------------- status / review ----------------------------------


def test_status_without_the_db(awaiting):
    code, out = run_cli("status", awaiting)
    assert code == cli.OK, out
    assert "cycle1: round 1 of 10, awaiting approval" in out
    row = next(line for line in out.splitlines() if line.strip().startswith("1 "))
    assert "waiting for approval" in row and " r1 " in row and " 280 " in row and "2.0e-03" in row
    assert row.rstrip().endswith("retune *")  # what is proposed, * = needs approval
    assert f"review {awaiting}" in out and f"approve {awaiting}" in out
    code, out = run_cli("status", awaiting, "--json")
    view = json.loads(out)
    assert view["status"] == "awaiting_approval" and view["members"][0]["epoch"] == 280


def _live_round(tmp_path):
    """A cycle in round 2: member 1 training (log + journal), member 2 queued (no journal yet), member 3
    stopped at the walltime; an alert for member 1, STOP_3 and STOP_ROUND set."""
    cycle = tmp_path / "cycle"
    round_dir = jobs.round_dir(cycle, 2)
    (round_dir / "alerts").mkdir(parents=True)
    t0 = 1_800_000_000.0
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0))
    for j in (1, 2, 3):
        (fold_dir(cycle, j)).mkdir(parents=True)
        (fold_dir(cycle, j) / "train_r2.toml").write_text('[data]\noutputs_dir = "./outputs_r2"\n')
    run = fold_dir(cycle, 1) / "outputs_r2" / "2027-01-15_08-00-00"
    run.mkdir(parents=True)
    epochs = "".join(f"[ x ] [train] Epoch {e} | Time {122.0 if e == 21 else 25.0} s | Train-Loss 1e-3 | "
                     f"Val-Loss {1e-3 / e:.6e} | Scale 0.5\n" for e in range(21, 31))
    (run / "deepx.log").write_text("[ x ] [optimizer] ... initial learning rate `0.001`.\n" + epochs)
    (round_dir / "watchdog_1.log").write_text(
        f"[{stamp}] job end time from SLURM_JOB_END_TIME: in 1800 s, stop 600 s before\n"
        f"[{stamp}] alert: overfit: val/train grew from 1.02 to 1.36\n")
    (round_dir / "watchdog_3.log").write_text(
        f"[{stamp}] job end time from SLURM_JOB_END_TIME: in 1800 s, stop 600 s before\n"
        f"[{stamp}] end: walltime, exit code -2, last epoch 40, last checkpoint 40\n")
    (round_dir / "alerts" / "member_1.txt").write_text("status: overfit\nreason: val/train grew from 1.02 to 1.36\n")
    (round_dir / "STOP_3").write_text("")
    (round_dir / "STOP_ROUND").write_text("")
    resume = {"action": "resume", "cause": "walltime", "reason": "walltime: resume from latest"}
    state = {"cycle_dir": str(cycle), "round": 2, "status": "running", "cfg": {"max_rounds": 6},
             "history": {"rounds": [{"round": 1, "members": {str(j): resume for j in (1, 2, 3)}}]},
             "members": {str(j): {"logs": [], "runs": []} for j in (1, 2, 3)}}
    return cycle, state, t0


def test_status_of_a_live_round(tmp_path):
    from hamilflow.training.supervisor import status
    cycle, state, t0 = _live_round(tmp_path)
    view = status.build_status(cycle, state, now=t0 + 600)
    m1, m2, m3 = view["members"]
    assert (m1["doing"], m1["epoch"], m1["run_round"]) == ("training", 30, 2)
    assert m1["s_per_epoch"] == 25.0  # the 122 s JIT epoch left out
    assert m1["lr"] == pytest.approx(5e-4) and m1["stop_in_s"] == pytest.approx(600)  # 1800 - 600 - 600
    assert "overfit" in m1["alert"]
    assert m2["doing"] == "queued" and m2["epoch"] is None
    assert m3["doing"] == "stopped (walltime)" and m3["stop_in_s"] is None and m3["stop_file"]
    text = status.format_status(view, cycle="C")
    assert "cycle: round 2 of 6, running" in text and "10 min" in text
    assert "! member 1: overfit: val/train grew from 1.02 to 1.36" in text
    assert "STOP_3 is set" in text and "STOP_ROUND is set" in text
    assert "hamilflow-train-supervisor stop C --member J" in text


def test_status_jobs_are_grouped_per_round(awaiting):
    rows = [("train_member_1_r1", "COMPLETED"), ("train_member_2_r1", "COMPLETED"),
            ("supervise_round_r1", "COMPLETED"), ("store_inputs", "COMPLETED"),
            ("train_member_1_r2", "RUNNING"), ("train_member_2_r2", "FAILED"), ("supervise_round_r2", "WAITING")]
    infos = [SimpleNamespace(name=n, db_id=str(i), state=SimpleNamespace(value=st)) for i, (n, st) in enumerate(rows)]

    class Controller:
        def get_jobs_info(self, flow_ids):
            return infos

    code, out = run_cli("status", awaiting, "--jobs", job_controller=lambda project: Controller())
    assert code == cli.OK
    assert "r1  supervise COMPLETED, train 1 COMPLETED, train 2 COMPLETED" in out
    assert "r2  supervise WAITING, train 1 RUNNING, train 2 FAILED" in out
    assert "store_inputs etc.: COMPLETED 1" in out
    assert "train_member_2_r2 [5] FAILED: the job itself died" in out
    assert out.index("jobs:") < out.index("next:") and "status " + str(awaiting) + " --jobs" not in out


def test_status_names_the_cycle_when_given_as_dot(awaiting, monkeypatch):
    monkeypatch.chdir(awaiting)
    code, out = run_cli("status", ".")
    assert code == cli.OK and out.startswith("cycle1: round 1 of 10")


def test_status_with_jobs_points_out_a_failed_supervise_job_and_remote_errors(awaiting):
    infos = [SimpleNamespace(name="train_member_1_r1", db_id="11", state=SimpleNamespace(value="COMPLETED")),
             SimpleNamespace(name="supervise_round_r1", db_id="13", state=SimpleNamespace(value="FAILED")),
             SimpleNamespace(name="train_member_2_r1", db_id="12", state=SimpleNamespace(value="REMOTE_ERROR"))]
    seen = {}

    class Controller:
        def get_jobs_info(self, flow_ids):
            seen["flow_ids"] = flow_ids
            return infos

    def controller(project):
        seen["project"] = project
        return Controller()

    code, out = run_cli("status", awaiting, "--jobs", job_controller=controller)
    assert code == cli.OK
    assert seen == {"project": "p", "flow_ids": [jobs.read_state(awaiting)["flows"][0]["uuid"]]}
    assert "`jf job rerun 13`" in out and "`jf job retry 12`" in out


def test_status_with_jobs_when_jobflow_remote_is_not_available(awaiting):
    def broken(project):
        raise ConnectionError("no DB")

    code, out = run_cli("status", awaiting, "--jobs", job_controller=broken)
    assert code == cli.ERROR and "jobflow-remote not available" in out


def test_review_is_compact_by_default_and_full_on_request(awaiting):
    code, out = run_cli("review", awaiting)
    assert code == cli.OK, out
    assert "round 1 → proposal for round 2: 2 members, 2 need approval (*)" in out
    assert "1 *" in out and "retune" in out
    assert "patience 120 → " in out and "NEW optimizer" in out and "from best of r1" in out
    assert "noise-dominated onset" in out and "suggested (rtol" in out  # the evidence of a retune
    assert f"approve {awaiting} --set 1.patience=15" in out and "lr " in out  # commands; the LR a retune starts at
    assert "---- LR reductions ----" not in out and 'action = "retune"' not in out
    code, full = run_cli("review", awaiting, "--full")
    assert code == cli.OK and "proposal.toml" in full and 'action = "retune"' in full and "---- LR reductions" in full


def test_review_one_member_and_json(awaiting):
    code, out = run_cli("review", awaiting, "--member", 2)
    assert code == cli.OK and "member 2 *: retune" in out and "member 1" not in out
    code, out = run_cli("review", awaiting, "--json")
    view = json.loads(out)
    assert [m["fold"] for m in view["members"]] == [1, 2]
    assert view["members"][0]["action"] == "retune" and view["members"][0]["spark"]
    assert run_cli("review", awaiting, "--member", 9)[0] == cli.ERROR


def test_sparkline():
    from hamilflow.training.supervisor.review import sparkline
    line = sparkline([1.0, 0.1, 0.01, 0.001], width=4)
    assert line == "█▅▃▁" or (len(line) == 4 and line[0] == "█" and line[-1] == "▁")
    assert sparkline([1.0]) == "" and sparkline([0.5, 0.5]) == "▁▁"


def test_unknown_cycle_is_an_error(tmp_path):
    code, out = run_cli("status", tmp_path)
    assert code == cli.ERROR and "no supervisor state" in out


# ---------------------------------- approve ----------------------------------


def test_approve_writes_the_tomls_updates_the_state_and_submits(awaiting):
    submit = Submitted()
    code, out = run_cli("approve", awaiting, submit=submit)
    assert code == cli.OK, out
    state = jobs.read_state(awaiting)
    assert (state["status"], state["round"]) == ("running", 2)
    record = state["history"]["rounds"][0]["members"]["1"]
    assert record["approved"] is True and record["needs_approval"] is False
    proposal_patience = record["fold_overrides"]["process"]["train"]["scheduler"]["patience"]
    with open(fold_dir(awaiting, 1) / "train_r2.toml", "rb") as f:
        r2 = tomllib.load(f)
    assert r2["process"]["train"]["scheduler"]["patience"] == proposal_patience
    assert r2["process"]["train"]["scheduler"]["min_learning_rate_scale"] == pytest.approx(1e-7)
    assert r2["data"]["model_save"]["latest_interval"] == 10
    (flow, worker, project), = submit.calls
    assert (worker, project) == ("login", "p")
    configs = {j.name: j.config.manager_config for j in flow.jobs}
    assert configs["train_member_1_r2"] == {"worker": "gpu", "resources": {"gres": "gpu:2"}}
    assert configs["supervise_round_r2"] == {"worker": "login"}
    assert state["flows"][-1] == {"round": 2, "uuid": flow.uuid}


def test_approve_dry_run_writes_and_submits_nothing(awaiting):
    before = jobs.read_state(awaiting)
    submit = Submitted()
    code, out = run_cli("approve", awaiting, "--dry-run", submit=submit)
    assert code == cli.OK and "dry run" in out and "member 1: retune" in out
    assert jobs.read_state(awaiting) == before and not submit.calls
    assert not (fold_dir(awaiting, 1) / "train_r2.toml").exists()


def test_approve_edit_applies_the_users_patience(awaiting, tmp_path):
    editor = tmp_path / "editor.py"
    editor.write_text("import re, sys\nfrom pathlib import Path\np = Path(sys.argv[1])\n"
                      "p.write_text(re.sub(r'patience = \\d+', 'patience = 33', p.read_text()))\n")
    editor.chmod(editor.stat().st_mode | stat.S_IEXEC)
    code, out = run_cli("approve", awaiting, "--edit", submit=Submitted(),
                        editor_env={"EDITOR": f"{sys.executable} {editor}"})
    assert code == cli.OK, out
    with open(fold_dir(awaiting, 2) / "train_r2.toml", "rb") as f:
        assert tomllib.load(f)["process"]["train"]["scheduler"]["patience"] == 33
    record = jobs.read_state(awaiting)["history"]["rounds"][0]["members"]["2"]
    assert record["fold_overrides"]["process"]["train"]["scheduler"]["patience"] == 33  # what the user changed


def test_approve_rejects_an_invalid_edit(awaiting, tmp_path):
    editor = tmp_path / "editor.py"
    editor.write_text("import sys\nfrom pathlib import Path\np = Path(sys.argv[1])\n"
                      "p.write_text(p.read_text().replace('patience = ', 'patiense = '))\n")
    code, out = run_cli("approve", awaiting, "--edit", submit=Submitted(),
                        editor_env={"EDITOR": f"{sys.executable} {editor}"})
    assert code == cli.ERROR and "patiense" in out
    assert jobs.read_state(awaiting)["status"] == "awaiting_approval"


def test_approve_refusals(awaiting):
    path = jobs.round_dir(awaiting, 1) / "proposal.toml"
    path.write_text(path.read_text().replace("after_round = 1", "after_round = 7"))
    code, out = run_cli("approve", awaiting, submit=Submitted())
    assert code == cli.REFUSED and "round 7" in out
    state = jobs.read_state(awaiting)
    state["status"] = "running"
    jobs.write_state(awaiting, state)
    code, out = run_cli("approve", awaiting, submit=Submitted())
    assert code == cli.REFUSED and "no proposal pending" in out
    code, out = run_cli("review", awaiting)
    assert code == cli.REFUSED


def test_a_failed_submission_restores_the_state(awaiting):
    before = jobs.read_state(awaiting)

    def failing(flow, worker=None, project=None):
        raise RuntimeError("runner down")

    code, out = run_cli("approve", awaiting, submit=failing)
    assert code == cli.ERROR and "state.json restored" in out
    assert jobs.read_state(awaiting) == before
    assert run_cli("approve", awaiting, submit=Submitted())[0] == cli.OK  # it can be approved again


# ---------------------------------- stop / finish / summary ----------------------------------


def test_stop_refuses_when_no_round_is_training(awaiting):
    code, out = run_cli("stop", awaiting)
    assert code == cli.REFUSED and "awaiting_approval" in out
    assert not (jobs.round_dir(awaiting, 1) / "STOP_ROUND").exists()
    assert run_cli("stop", awaiting, "--clear")[0] == cli.OK  # clearing is always allowed


def test_stop_and_clear(awaiting):
    assert run_cli("approve", awaiting, submit=Submitted())[0] == cli.OK  # round 2 running
    here = jobs.round_dir(awaiting, 2)
    assert run_cli("stop", awaiting)[0] == cli.OK and (here / "STOP_ROUND").exists()
    assert run_cli("stop", awaiting, "--member", 2)[0] == cli.OK and (here / "STOP_2").exists()
    assert "STOP_2 is set" in run_cli("status", awaiting)[1]
    assert run_cli("stop", awaiting, "--member", 2, "--clear")[0] == cli.OK and not (here / "STOP_2").exists()
    assert run_cli("stop", awaiting, "--clear")[0] == cli.OK and not (here / "STOP_ROUND").exists()
    assert run_cli("stop", awaiting, "--member", 5)[0] == cli.ERROR


def test_finish_refuses_while_a_proposal_is_pending_unless_forced(awaiting):
    code, out = run_cli("finish", awaiting)
    assert code == cli.REFUSED and "proposal is pending" in out
    code, out = run_cli("finish", awaiting, "--force")
    assert code == cli.OK
    state = jobs.read_state(awaiting)
    assert state["status"] == "finished" and state["finish_requested"] is True
    summary = json.loads(jobs.summary_path(awaiting).read_text())
    assert set(summary["members"]) == {"1", "2"}


def test_finish_while_running_lets_the_supervise_job_end_the_cycle(awaiting):
    assert run_cli("approve", awaiting, submit=Submitted())[0] == cli.OK
    before = jobs.read_state(awaiting)
    code, out = run_cli("finish", awaiting)
    assert code == cli.OK and "may still be training" in out
    assert jobs.read_state(awaiting) == before  # the CLI does not write state.json while a round runs
    assert jobs.finish_path(awaiting).read_text().startswith("set by")
    assert "FINISH is set" in run_cli("status", awaiting)[1]
    # the round-2 supervise job then ends the cycle instead of submitting round 3
    infos = {}
    for j in (1, 2):
        run = fold_dir(awaiting, j) / "outputs_r2" / "2026-11-24_12-00-00"
        (run / "model").mkdir(parents=True)
        infos[str(j)] = {"reason": "walltime", "return_code": 1, "run_dir": None, "last_epoch": None,
                         "last_checkpoint_epoch": None, "live_status": None, "live_reason": None,
                         "newest_checkpoint_epoch": None}
    output = jobs.supervise_round_job.original(str(awaiting), 2, infos)
    assert output["status"] == "finished"
    state = jobs.read_state(awaiting)
    assert state["status"] == "finished" and state["finish_requested"] is True
    assert not jobs.finish_path(awaiting).exists()  # converted into finish_requested, then removed
    # summary.json: members that would have trained on are "finished by user"; settings only for trained rounds
    final = json.loads(jobs.summary_path(awaiting).read_text())
    member = final["members"]["1"]
    assert member["action"] == "finished_by_user"
    assert [s["round"] for s in member["settings"]] == [1, 2]  # round 1 included; round 3 was decided but never ran


def test_summary_and_infer_tomls(tmp):
    cycle = supervised(tmp, actions={"retune": "auto"}, max_rounds=2)
    assert jobs.read_state(cycle)["status"] == "finished"
    code, out = run_cli("summary", cycle)
    assert code == cli.OK and "member 1: give_up (max_rounds)" in out and "round 2 retune" in out
    code, out = run_cli("summary", cycle, "--json")
    final = json.loads(out)
    run_1 = final["members"]["1"]["final_run_dir"]
    assert run_1 is not None and Path(run_1).parent.name in ("outputs_r1", "outputs_r2")
    overrides = tmp / "infer.json"
    overrides.write_text(json.dumps({"data": {"inputs_dir": "/infer"}}))
    code, out = run_cli("summary", cycle, "--infer-tomls", overrides, "--infer-name", "infer_md")
    assert code == cli.OK, out
    with open(fold_dir(cycle, 1) / "infer_md" / "infer.toml", "rb") as f:
        written = tomllib.load(f)
    assert written["model"]["model_dir"] == str((Path(run_1) / "model").resolve())
    assert written["data"]["inputs_dir"] == "/infer"


def test_an_interrupted_submission_keeps_the_round_running_for_resubmit(awaiting):
    """Ctrl-C during submit_flow: the flow may already be in the DB, so the state stays running (with the new
    flow's uuid) and the user is sent to `approve --resubmit`, which checks the DB first."""
    submitted = []

    def interrupted(flow, worker=None, project=None):
        submitted.append(flow)
        raise KeyboardInterrupt

    out = io.StringIO()
    with pytest.raises(KeyboardInterrupt):
        cli.main(["approve", str(awaiting)], out=out, submit=interrupted)
    state = jobs.read_state(awaiting)
    assert (state["status"], state["round"]) == ("running", 2)
    assert state["flows"][-1] == {"round": 2, "uuid": submitted[0].uuid}
    assert "approve" in out.getvalue() and "--resubmit" in out.getvalue()


class Controller:
    def __init__(self, infos=(), error=None):
        self.infos, self.error = list(infos), error

    def __call__(self, project):
        if self.error:
            raise self.error
        return self

    def get_jobs_info(self, flow_ids):
        return self.infos


def test_resubmit_a_round_that_never_reached_jobflow_remote(awaiting):
    def killed(flow, worker=None, project=None):  # approve killed hard: the state says running, nothing submitted
        return None

    assert run_cli("approve", awaiting, submit=killed)[0] == cli.OK
    old_uuid = jobs.read_state(awaiting)["flows"][-1]["uuid"]
    code, out = run_cli("approve", awaiting, "--resubmit", submit=Submitted(),
                        job_controller=Controller([SimpleNamespace(name="train_member_1_r2")]))
    assert code == cli.REFUSED and "it was submitted" in out
    code, out = run_cli("approve", awaiting, "--resubmit", submit=Submitted(),
                        job_controller=Controller(error=ConnectionError("no DB")))
    assert code == cli.REFUSED and "jf job list" in out
    submit = Submitted()
    code, out = run_cli("approve", awaiting, "--resubmit", submit=submit, job_controller=Controller())
    assert code == cli.OK, out
    (flow, worker, _), = submit.calls
    assert worker == "login" and sorted(j.name for j in flow.jobs if j.name.startswith("train")) == [
        "train_member_1_r2", "train_member_2_r2"]
    flows = jobs.read_state(awaiting)["flows"]
    assert [f["round"] for f in flows] == [1, 2] and flows[-1]["uuid"] == flow.uuid != old_uuid


def test_resubmit_refuses_unless_a_round_is_running(awaiting):
    code, out = run_cli("approve", awaiting, "--resubmit", job_controller=Controller())
    assert code == cli.REFUSED and "awaiting_approval" in out


def test_approve_names_the_members_left_out_of_an_edited_proposal(awaiting, tmp_path):
    editor = tmp_path / "editor.py"
    editor.write_text("import re, sys\nfrom pathlib import Path\np = Path(sys.argv[1])\n"
                      "text = p.read_text()\n"
                      "p.write_text(text[:text.index('[members.2]')])\n")
    code, out = run_cli("approve", awaiting, "--edit", submit=Submitted(),
                        editor_env={"EDITOR": f"{sys.executable} {editor}"})
    assert code == cli.OK, out
    assert "members [2] are not in the proposal: treated as not training" in out
    assert not (fold_dir(awaiting, 2) / "train_r2.toml").exists()
    record = jobs.read_state(awaiting)["history"]["rounds"][0]["members"]["2"]
    assert (record["action"], record["cause"]) == ("done", "removed_by_user")


# ---------------------------------- approve --set ----------------------------------


def _scheduler(cycle, fold):
    with open(fold_dir(cycle, fold) / "train_r2.toml", "rb") as f:
        return tomllib.load(f)["process"]["train"]


def test_approve_set_changes_values_and_records_them(awaiting):
    code, out = run_cli("approve", awaiting, "--set", "1.patience=15", "--set", "1.cooldown=7", submit=Submitted())
    assert code == cli.OK, out
    assert "set 1.patience=15" in out and "patience 120 → 15" in out
    train1, train2 = _scheduler(awaiting, 1), _scheduler(awaiting, 2)
    assert (train1["scheduler"]["patience"], train1["scheduler"]["cooldown"]) == (15, 7)
    assert train2["scheduler"]["patience"] != 15  # member 2 as proposed
    record = jobs.read_state(awaiting)["history"]["rounds"][0]["members"]["1"]
    assert record["fold_overrides"]["process"]["train"]["scheduler"]["patience"] == 15
    proposal = jobs.read_proposal(jobs.round_dir(awaiting, 1) / "proposal.toml")
    assert proposal["members"][1]["fold_overrides"]["process"]["train"]["scheduler"]["patience"] == 15


def test_approve_set_action_resume_and_done(awaiting):
    code, out = run_cli("approve", awaiting, "--set", "1.action=resume", "--set", "2.action=done", submit=Submitted())
    assert code == cli.OK, out
    train1 = _scheduler(awaiting, 1)
    assert train1["continued"]["load_model_type"] == "latest" and train1["continued"]["new_optimizer"] is False
    assert train1["scheduler"]["patience"] == 120  # its current settings, no retune
    assert "/outputs_r1/" in train1["continued"]["previous_output_dir"]
    assert not (fold_dir(awaiting, 2) / "train_r2.toml").exists()
    record = jobs.read_state(awaiting)["history"]["rounds"][0]["members"]
    assert (record["1"]["action"], record["2"]["action"], record["2"]["cause"]) == ("resume", "done", "user")


@pytest.mark.parametrize(("sets", "message"), [
    (["1.patiense=3"], "unknown key"),
    (["1.patience=abc"], "must be a int"),
    (["patience=3"], "expected J.key=value"),
    (["9.patience=3"], "not a member"),
    (["1.action=retune"], "edit the proposal"),
    (["2.action=done", "2.patience=5"], "does not train"),
])
def test_approve_set_rejects_bad_input_and_changes_nothing(awaiting, sets, message):
    before = jobs.read_state(awaiting)
    proposal_path = jobs.round_dir(awaiting, 1) / "proposal.toml"
    proposal_before = proposal_path.read_text()
    argv = [x for item in sets for x in ("--set", item)]
    submit = Submitted()
    code, out = run_cli("approve", awaiting, *argv, submit=submit)
    assert code == cli.ERROR and message in out
    assert jobs.read_state(awaiting) == before and proposal_path.read_text() == proposal_before and not submit.calls


def test_approve_set_dry_run_leaves_the_proposal(awaiting):
    proposal_path = jobs.round_dir(awaiting, 1) / "proposal.toml"
    before = proposal_path.read_text()
    code, out = run_cli("approve", awaiting, "--set", "1.patience=15", "--dry-run", submit=Submitted())
    assert code == cli.OK and "patience 120 → 15" in out and "dry run" in out
    assert proposal_path.read_text() == before
