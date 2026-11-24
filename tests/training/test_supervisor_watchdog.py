"""The watchdog against a fake deeph-train (fake_deeph_train.py) that writes real-format logs
and checkpoints at a fast pace. Every test kills the fake's process group at teardown."""

import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from synthetic_deeph_logs import decay_then_noise

from hamilflow.training.supervisor import ExitInfo, SupervisorConfig
from hamilflow.training.supervisor.watchdog import (
    clean_env,
    find_new_run_dir,
    job_end_time,
    last_checkpoint_epoch,
    run_supervised,
)
from hamilflow.training.write_train_infer_toml import write_train_toml

FAKE = Path(__file__).with_name("fake_deeph_train.py")


@pytest.fixture
def fold(tmp_path):
    """A fold dir with its round-1 train.toml, and the round dir; kills what the fake left behind."""
    fold_dir = tmp_path / "train_1"
    toml = write_train_toml({
        "data": {"inputs_dir": "/inputs", "outputs_dir": "./outputs_r1", "model_save": {"latest_interval": 10}},
        "process": {"train": {"scheduler": {"patience": 120, "cooldown": 50}}},
    }, fold_dir, filename="train_r1.toml")
    round_dir = tmp_path / "supervisor" / "round_1"
    yield fold_dir, toml, round_dir
    for pidfile in tmp_path.glob("pid_*"):
        try:
            os.killpg(int(pidfile.read_text()), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, ValueError):
            pass


_N = [0]


def run(fold, spec=None, *, before=(), end_time=None, **cfg_kwargs):
    fold_dir, toml, round_dir = fold
    _N[0] += 1
    spec = {"pidfile": str(fold_dir.parent / f"pid_{_N[0]}"), **(spec or {})}
    spec_path = fold_dir.parent / f"spec_{_N[0]}.json"
    spec_path.write_text(json.dumps(spec))
    cfg = SupervisorConfig(**{
        "train_command": [sys.executable, str(FAKE), "{toml}", str(spec_path)],
        "poll_s": 0.05, "stop_grace": 0.5, "kill_grace": 0.3, "checkpoint_settle_s": 0.05, **cfg_kwargs,
    })
    info = run_supervised(toml, fold_dir, list(before), cfg, round_dir, 1, end_time=end_time)
    return info, (round_dir / "watchdog_1.log").read_text()


# ---------------------------------- helpers ----------------------------------


def make_checkpoint(run_dir, kind, epoch, parts=("params", "states"), metadata=True):
    for part in parts:
        path = run_dir / "model" / part / f"{kind}.pytree" / f"epoch_{epoch}"
        path.mkdir(parents=True)
        (path / "data").write_text("x")
        if metadata:
            (path / "_CHECKPOINT_METADATA").write_text("{}")


def test_last_checkpoint_epoch_needs_params_and_states_and_settled_files(tmp_path):
    make_checkpoint(tmp_path, "latest", 10)
    make_checkpoint(tmp_path, "latest", 20, parts=("params",))  # states not written yet
    make_checkpoint(tmp_path, "best", 25)
    assert last_checkpoint_epoch(tmp_path, settle_s=0.0) == 10
    assert last_checkpoint_epoch(tmp_path, settle_s=0.0, kind="best") == 25
    make_checkpoint(tmp_path, "latest", 30)
    assert last_checkpoint_epoch(tmp_path, settle_s=0.0) == 30
    # just written: within the settle time it is not trusted
    assert last_checkpoint_epoch(tmp_path, settle_s=5.0, now=time.time()) is None
    assert last_checkpoint_epoch(tmp_path, settle_s=5.0, now=time.time() + 10) == 30
    assert last_checkpoint_epoch(tmp_path / "nothing") is None
    assert last_checkpoint_epoch(None) is None


def test_find_new_run_dir_ignores_the_runs_there_before(tmp_path):
    (tmp_path / "2099-01-01_00-00-00").mkdir()
    (tmp_path / "not_a_run").mkdir()
    assert find_new_run_dir(tmp_path, {"2099-01-01_00-00-00"}) is None
    (tmp_path / "2026-11-23_10-00-00").mkdir()
    assert find_new_run_dir(tmp_path, {"2099-01-01_00-00-00"}) == tmp_path / "2026-11-23_10-00-00"
    assert find_new_run_dir(tmp_path / "missing") is None


def test_job_end_time_lookup_order():
    cfg = SupervisorConfig(walltime=3600.0)
    env = {"SLURM_JOB_END_TIME": "5000", "SLURM_JOB_ID": "7"}
    assert job_end_time(cfg, 100.0, env) == (5000.0, "SLURM_JOB_END_TIME")
    calls = []

    def squeue(job_id):
        calls.append(job_id)
        return 4000.0

    assert job_end_time(cfg, 100.0, {"SLURM_JOB_ID": "7"}, squeue) == (4000.0, "squeue")
    assert calls == ["7"]  # a single call
    assert job_end_time(cfg, 100.0, {"SLURM_JOB_ID": "7"}, lambda _: None) == (3700.0, "cfg.walltime")
    assert job_end_time(SupervisorConfig(), 100.0, {}) == (None, "none")


def test_clean_env_drops_the_hamilflow_python_variables():
    env = clean_env({"PYTHONPATH": "/x", "PYTHONHOME": "/y", "VIRTUAL_ENV": "/v", "PATH": "/bin",
                     "OMP_NUM_THREADS": "4"})
    assert env == {"PATH": "/bin", "OMP_NUM_THREADS": "4"}


def test_train_command_must_hold_the_toml_placeholder():
    with pytest.raises(ValueError, match="toml"):
        SupervisorConfig(train_command=["deeph-train"])
    with pytest.raises(ValueError, match="walltime_margin"):
        SupervisorConfig(walltime_margin=100.0, stop_grace=90.0, kill_grace=30.0)


# ---------------------------------- exits ----------------------------------


def test_normal_finish(fold):
    info, journal = run(fold, {"n_epochs": 20})
    assert (info.reason, info.return_code, info.last_epoch, info.last_checkpoint_epoch) == ("finished", 0, 20, 20)
    assert Path(info.run_dir).parent == fold[0] / "outputs_r1"
    assert (Path(info.run_dir) / "deepx.log").exists()
    assert "no walltime limit known" in journal and "end: finished" in journal
    assert (fold[0] / "train_r1.out").exists()


def test_preexisting_run_dirs_are_ignored(fold):
    old = fold[0] / "outputs_r1" / "2099-01-01_00-00-00"
    (old / "model").mkdir(parents=True)
    (old / "deepx.log").write_text("[ 01.01-00:00:00 ] [train] Epoch 1 | Time 1.0 s | Train-Loss nan | "
                                   "Val-Loss nan | Scale 1.0\n")
    info, _ = run(fold, {"n_epochs": 5})
    assert info.reason == "finished"
    assert Path(info.run_dir).name != old.name


def test_trainer_crash_is_crashed(fold):
    info, _ = run(fold, {"n_epochs": 50, "crash_at": 12})
    assert (info.reason, info.return_code, info.last_epoch, info.last_checkpoint_epoch) == ("crashed", 1, 12, 10)
    assert info.run_dir is not None


def test_trainer_exiting_before_its_run_dir_is_crashed(fold):
    info, journal = run(fold, {"no_run_dir": True})
    assert (info.reason, info.return_code, info.run_dir, info.last_checkpoint_epoch) == ("crashed", 1, None, None)
    assert "before creating a run dir" in journal


def test_trainer_that_cannot_start_is_crashed(fold):
    fold_dir, toml, round_dir = fold
    cfg = SupervisorConfig(train_command=["/nonexistent/deeph-train", "{toml}"])
    info = run_supervised(toml, fold_dir, [], cfg, round_dir, 1, end_time=None)
    assert (info.reason, info.return_code, info.run_dir) == ("crashed", None, None)


def test_env_is_cleaned(fold, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/leak/from/hamilflow")
    monkeypatch.setenv("VIRTUAL_ENV", "/hamilflow-venv")
    monkeypatch.setenv("FAKE_MARK", "kept")
    run(fold, {"n_epochs": 1})
    out = (fold[0] / "train_r1.out").read_text()
    assert "PYTHONPATH=None" in out and "VIRTUAL_ENV=None" in out and "FAKE_MARK=kept" in out


# ---------------------------------- stops ----------------------------------


@pytest.mark.parametrize("name", ["STOP_1", "STOP_ROUND"])
def test_control_files_stop_the_member(fold, name):
    fold[2].mkdir(parents=True)
    (fold[2] / name).touch()
    info, journal = run(fold, {"n_epochs": 1000})
    assert info.reason == "stopped"
    # our SIGINT, answered by the fake (1, as deeph-train) or before it set its handler (-2): no crash either way
    assert info.return_code in (1, -signal.SIGINT)
    assert f"control file {name}" in journal and "sending SIGINT" in journal


def test_walltime_stop(fold):
    t0 = time.time()
    info, journal = run(fold, {"n_epochs": 10000, "epoch_s": 0.02}, end_time=t0 + 1.5, walltime_margin=1.0)
    assert info.reason == "walltime" and info.return_code == 1  # deeph-train's KeyboardInterrupt exit
    assert time.time() - t0 < 1.5
    assert "walltime margin reached" in journal or "right after the latest checkpoint" in journal


@pytest.mark.slow
def test_walltime_stop_right_after_a_latest_checkpoint(fold):
    """Within one latest_interval (10 x 0.05 s) of the deadline, stop as soon as a new latest checkpoint lands."""
    t0 = time.time()
    # a flat val: one best checkpoint only, so the stop does not wait for best checkpoints being written.
    # The logged epoch time is twice the real one: the window (10 logged epochs) then holds two real
    # checkpoint intervals, so one surely lands in it whatever the start-up time
    info, journal = run(fold, {"n_epochs": 10000, "epoch_s": 0.05, "logged_epoch_s": 0.1, "val": [1e-2]},
                        end_time=t0 + 3.6, walltime_margin=1.0)
    assert info.reason == "walltime"
    assert "right after the latest checkpoint" in journal
    # stopped soon after a latest checkpoint: at most half an interval (10 epochs) later
    assert info.last_checkpoint_epoch is not None and info.last_epoch - info.last_checkpoint_epoch <= 10 // 2
    assert time.time() - t0 < 2.6 + 1.0  # before the deadline, plus the stop


def test_hard_stop_on_nan_even_with_divergence_ask(fold):
    val = list(10 ** (-2 - 0.01 * np.arange(200)))
    val[8] = float("nan")
    info, journal = run(fold, {"n_epochs": 200, "val": val, "epoch_s": 0.01})
    assert SupervisorConfig().actions["divergence"] == "ask"
    assert (info.reason, info.live_status) == ("live_rule", "diverged")
    assert "non-finite loss at epoch 9" in info.live_reason
    assert "hard stop" in journal


def test_auto_stop_when_converged(fold):
    stages = [1.0, 0.5, 0.499, 0.4985, 0.4984]
    val = list(np.repeat(stages, 5))
    scale = list(np.repeat(0.5 ** np.arange(5), 5))
    info, journal = run(fold, {"n_epochs": 1000, "val": val, "scale": scale, "epoch_s": 0.01})
    assert (info.reason, info.live_status) == ("live_rule", "converged")
    assert "stop_converged auto" in journal
    assert not (fold[2] / "STOP_ROUND").exists()


def test_ask_writes_one_alert_and_keeps_training(fold):
    """val x 100 from epoch 30 (divergence by the factor rule, not a hard stop) with divergence = ask."""
    val = 10 ** (-2 - 0.01 * np.arange(60))
    val[30:] *= 100
    info, journal = run(fold, {"n_epochs": 60, "val": list(val), "epoch_s": 0.01})
    assert info.reason == "finished"  # not stopped
    assert journal.count("alert: diverged") == 1
    alert = (fold[2] / "alerts" / "member_1.txt").read_text()
    assert "status: diverged" in alert and "STOP_1" in alert


def test_off_neither_stops_nor_alerts(fold):
    val = 10 ** (-2 - 0.01 * np.arange(60))
    val[30:] *= 100
    info, journal = run(fold, {"n_epochs": 60, "val": list(val), "epoch_s": 0.01}, actions={"divergence": "off"})
    assert info.reason == "finished"
    assert "alert" not in journal


def test_mistuned_auto_stops_the_round(fold):
    """Patience 120 and still at the initial LR well after the onset: W3, retune = auto."""
    val = list(decay_then_noise(n_epochs=600))
    info, journal = run(fold, {"n_epochs": 600, "val": val, "epoch_s": 0.002}, actions={"retune": "auto"})
    assert (info.reason, info.live_status) == ("live_rule", "mistuned")
    assert "patience too large" in info.live_reason
    assert (fold[2] / "STOP_ROUND").exists()
    assert "wrote STOP_ROUND" in journal


def stop_later(round_dir, delay=0.5):
    """Touch STOP_1 once the fake has set its signal handlers."""
    round_dir.mkdir(parents=True, exist_ok=True)
    timer = threading.Timer(delay, (round_dir / "STOP_1").touch)
    timer.start()
    return timer


def test_escalation_to_sigterm_then_sigkill(fold):
    stop_later(fold[2])
    info, journal = run(fold, {"n_epochs": 10000, "sigint": "ignore", "sigterm": "ignore"})
    assert info.reason == "stopped"
    assert info.return_code == -signal.SIGKILL
    assert journal.index("sending SIGINT") < journal.index("sending SIGTERM") < journal.index("sending SIGKILL")


def test_sigint_ignored_but_sigterm_stops(fold):
    stop_later(fold[2])
    info, journal = run(fold, {"n_epochs": 10000, "sigint": "ignore"})
    assert (info.reason, info.return_code) == ("stopped", -signal.SIGTERM)
    assert "sending SIGKILL" not in journal


def test_an_error_in_the_watchdog_kills_the_trainer(fold):
    fold_dir, toml, round_dir = fold
    pidfile = fold_dir.parent / "pid_error"
    spec = fold_dir.parent / "spec_error.json"
    spec.write_text(json.dumps({"n_epochs": 10000, "pidfile": str(pidfile)}))
    cfg = SupervisorConfig(train_command=[sys.executable, str(FAKE), "{toml}", str(spec)], poll_s=0.05)

    def broken_sleep(_):
        raise RuntimeError("watchdog bug")

    with pytest.raises(RuntimeError, match="watchdog bug"):
        run_supervised(toml, fold_dir, [], cfg, round_dir, 1, end_time=None, sleep=broken_sleep)
    for _ in range(50):  # the fake writes its pid file once it is up; it must be gone by now
        if pidfile.exists():
            break
        time.sleep(0.02)
    if pidfile.exists():
        with pytest.raises(ProcessLookupError):
            os.killpg(int(pidfile.read_text()), 0)
    assert "watchdog error: killing the trainer" in (round_dir / "watchdog_1.log").read_text()


def test_exit_info_is_json_ready(fold):
    info, _ = run(fold, {"n_epochs": 3})
    assert ExitInfo.from_dict(json.loads(json.dumps(info.to_dict()))) == info


# ---------------------------------- STOP_ROUND wait, live_actions, survivors, nits ----------------------------


def test_stop_round_of_another_member_waits_for_the_first_latest(fold):
    fold[2].mkdir(parents=True)
    (fold[2] / "STOP_ROUND").write_text("member 2: patience too large\n")
    info, journal = run(fold, {"n_epochs": 10000, "epoch_s": 0.02})
    assert info.reason == "stopped"
    assert info.last_checkpoint_epoch is not None and info.last_checkpoint_epoch >= 10
    assert "training on until the first one" in journal


def test_stop_round_wait_is_bounded(fold):
    fold[2].mkdir(parents=True)
    (fold[2] / "STOP_ROUND").touch()
    info, journal = run(fold, {"n_epochs": 10000, "startup_s": 2.0}, round_stop_max_wait=0.3)
    assert (info.reason, info.last_checkpoint_epoch) == ("stopped", None)
    assert "stopping anyway" in journal


def test_stop_round_wait_yields_to_the_walltime(fold):
    fold[2].mkdir(parents=True)
    (fold[2] / "STOP_ROUND").touch()
    info, _ = run(fold, {"n_epochs": 10000, "startup_s": 1.0}, end_time=time.time() + 1.3, walltime_margin=1.0)
    assert info.reason == "walltime"


def test_stop_round_wait_yields_to_a_hard_stop(fold):
    fold[2].mkdir(parents=True)
    (fold[2] / "STOP_ROUND").touch()
    # deeph-train saves a latest at the first epoch and logs it after the save: the wait ends right there,
    # unless that checkpoint is still within the settle time. A nan at that first epoch then wins
    val = list(10 ** (-2 - 0.01 * np.arange(100)))
    val[0] = float("nan")
    info, journal = run(fold, {"n_epochs": 100, "val": val, "epoch_s": 0.05}, checkpoint_settle_s=1.0)
    assert (info.reason, info.live_status) == ("live_rule", "diverged")
    assert "training on until the first one" in journal


def test_live_actions_keep_w3_from_stopping_the_round_again(fold):
    val = list(decay_then_noise(n_epochs=600))
    fold_dir, toml, round_dir = fold
    spec = fold_dir.parent / "spec_live.json"
    spec.write_text(json.dumps({"n_epochs": 600, "val": val, "epoch_s": 0.002,
                                "pidfile": str(fold_dir.parent / "pid_live")}))
    cfg = SupervisorConfig(train_command=[sys.executable, str(FAKE), "{toml}", str(spec)], poll_s=0.05,
                           stop_grace=0.5, kill_grace=0.3, checkpoint_settle_s=0.05, actions={"retune": "auto"})
    info = run_supervised(toml, fold_dir, [], cfg, round_dir, 1, end_time=None, live_actions={"retune": "off"})
    assert info.reason == "finished"
    assert not (round_dir / "STOP_ROUND").exists()
    assert "live rules overridden for this member" in (round_dir / "watchdog_1.log").read_text()


def test_survivors_of_a_wrapper_that_exits_are_killed(fold):
    fold_dir, toml, round_dir = fold
    pidfile = fold_dir.parent / "pid_orphan"
    spec = fold_dir.parent / "spec_orphan.json"
    spec.write_text(json.dumps({"n_epochs": 10000, "pidfile": str(pidfile), "orphan_ok": True}))
    command = ["bash", "-c", f"{sys.executable} {FAKE} {{toml}} {spec} & sleep 0.5; exit 0"]
    cfg = SupervisorConfig(train_command=command, poll_s=0.05, checkpoint_settle_s=0.05)
    info = run_supervised(toml, fold_dir, [], cfg, round_dir, 1, end_time=None)
    # the wrapper's exit 0 is not a finished training: its trainer was still running, and was killed
    assert (info.reason, info.return_code) == ("crashed", 0)
    journal = (round_dir / "watchdog_1.log").read_text()
    assert "were still running: killed" in journal and "a crash" in journal
    pid = int(pidfile.read_text())
    time.sleep(0.1)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_sleep_never_overshoots_the_deadline(fold):
    pauses = []

    def sleep(seconds):
        pauses.append(seconds)
        time.sleep(seconds)

    fold_dir, toml, round_dir = fold
    spec = fold_dir.parent / "spec_sleep.json"
    spec.write_text(json.dumps({"n_epochs": 10000, "pidfile": str(fold_dir.parent / "pid_sleep")}))
    cfg = SupervisorConfig(train_command=[sys.executable, str(FAKE), "{toml}", str(spec)], poll_s=30.0,
                           walltime_margin=1.0, stop_grace=0.5, kill_grace=0.3, checkpoint_settle_s=0.05)
    t0 = time.time()
    info = run_supervised(toml, fold_dir, [], cfg, round_dir, 1, end_time=t0 + 1.5, sleep=sleep)
    assert info.reason == "walltime"
    assert time.time() - t0 < 3.0
    assert max(pauses) <= 0.6


def test_deadline_already_passed_does_not_start_the_trainer(fold):
    info, journal = run(fold, {"n_epochs": 10}, end_time=time.time() - 10, walltime_margin=1.0)
    assert (info.reason, info.run_dir, info.return_code) == ("walltime", None, None)
    assert "not starting the trainer" in journal
    assert not list(fold[0].parent.glob("pid_*"))


def test_a_failing_live_check_is_journaled_and_training_goes_on(fold, monkeypatch):
    from hamilflow.training.supervisor import watchdog

    def broken(*_args, **_kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(watchdog, "live_check", broken)
    info, journal = run(fold, {"n_epochs": 20})
    assert info.reason == "finished"
    assert "live_check failed" in journal and "RuntimeError: unexpected" in journal


def test_log_tail_starts_again_after_a_truncation(tmp_path):
    from hamilflow.training.supervisor.watchdog import _LogTail

    path = tmp_path / "deepx.log"
    line = "[ 01.01-00:00:00 ] [train] Epoch {} | Time 2.0 s | Train-Loss 1e-1 | Val-Loss 1e-1 | Scale 1.0\n"
    path.write_text("".join(line.format(i) for i in range(1, 6)))
    tail = _LogTail(path)
    tail.update()
    assert tail.epochs == [1, 2, 3, 4, 5]
    path.write_text(line.format(1))
    tail.update()
    assert tail.epochs == [1]


def test_exit_info_reports_the_newest_checkpoint_on_disk(fold):
    info, _ = run(fold, {"n_epochs": 20})
    assert (info.last_checkpoint_epoch, info.newest_checkpoint_epoch) == (20, 20)


def test_a_wrapper_that_execs_the_trainer_finishes_normally(fold):
    fold_dir, toml, round_dir = fold
    spec = fold_dir.parent / "spec_exec.json"
    spec.write_text(json.dumps({"n_epochs": 5, "pidfile": str(fold_dir.parent / "pid_exec")}))
    cfg = SupervisorConfig(train_command=["bash", "-c", f"exec {sys.executable} {FAKE} {{toml}} {spec}"],
                           poll_s=0.05, checkpoint_settle_s=0.05)
    assert run_supervised(toml, fold_dir, [], cfg, round_dir, 1, end_time=None).reason == "finished"


def test_last_checkpoint_epoch_needs_the_orbax_metadata(tmp_path):
    make_checkpoint(tmp_path, "latest", 10)
    make_checkpoint(tmp_path, "latest", 20, metadata=False)  # renamed but incomplete: never the case with orbax
    leftover = tmp_path / "model" / "params" / "latest.pytree" / "epoch_30.orbax-checkpoint-tmp-7"
    leftover.mkdir()
    (tmp_path / "model" / "states" / "latest.pytree" / "epoch_30.orbax-checkpoint-tmp-7").mkdir()
    assert last_checkpoint_epoch(tmp_path, settle_s=0.0) == 10
    from hamilflow.training.supervisor.watchdog import newest_checkpoint_epoch
    assert newest_checkpoint_epoch(tmp_path) == 20  # the tmp dirs are not checkpoints at all


def test_sigint_mid_save_leaves_a_tmp_dir_and_rc_1(fold):
    fold[2].mkdir(parents=True)
    timer = threading.Timer(0.5, (fold[2] / "STOP_1").touch)
    timer.start()
    info, _ = run(fold, {"n_epochs": 10000, "leave_tmp_on_sigint": True, "sigint_exit_s": 0.3})
    assert (info.reason, info.return_code) == ("stopped", 1)
    pytree = Path(info.run_dir) / "model" / "params" / "latest.pytree"
    assert any(".orbax-checkpoint-tmp-" in p.name for p in pytree.iterdir())
    assert info.last_checkpoint_epoch is not None and info.last_checkpoint_epoch % 10 == 0


def test_a_toml_without_the_full_continued_table_fails_at_start_up(fold):
    fold_dir, toml, _ = fold
    toml.write_text(toml.read_text().replace('load_model_epoch = -1\n', ''))
    info, _ = run(fold, {"n_epochs": 5})
    assert (info.reason, info.return_code, info.run_dir) == ("crashed", 1, None)
    assert "previous_output_dir" in (fold_dir / "train_r1.out").read_text()


def test_a_kept_optimizer_resume_continues_the_epoch_numbers(fold):
    """The fake, like deeph-train: resuming latest epoch 10 with the optimizer kept logs from epoch 11 on."""
    first, _ = run(fold, {"n_epochs": 12})
    assert first.last_checkpoint_epoch == 10
    fold_dir, _, round_dir = fold
    toml2 = write_train_toml({
        "data": {"inputs_dir": "/inputs", "outputs_dir": "./outputs_r2", "model_save": {"latest_interval": 10}},
        "process": {"train": {"continued": {"enable": True, "previous_output_dir": first.run_dir,
                                            "load_model_type": "latest", "new_optimizer": False}}},
    }, fold_dir, filename="train_r2.toml")
    spec = fold_dir.parent / "spec_resume.json"
    spec.write_text(json.dumps({"n_epochs": 5, "pidfile": str(fold_dir.parent / "pid_resume")}))
    cfg = SupervisorConfig(train_command=[sys.executable, str(FAKE), "{toml}", str(spec)], poll_s=0.05,
                           checkpoint_settle_s=0.05)
    second = run_supervised(toml2, fold_dir, [Path(first.run_dir) / "deepx.log"], cfg, round_dir, 1, end_time=None)
    assert (second.reason, second.last_epoch) == ("finished", 15)
    text = (Path(second.run_dir) / "deepx.log").read_text()
    assert "Epoch 11 |" in text and "NOT applying new optimizer" in text


def test_the_first_epoch_saves_a_latest_and_the_interval_counts_on_continued_numbers(fold):
    """As deeph-train: latest at the run's first epoch, then at the multiples of latest_interval (10)."""
    info, _ = run(fold, {"n_epochs": 12})
    pytree = Path(info.run_dir) / "model" / "params" / "latest.pytree"
    assert sorted(int(p.name[6:]) for p in pytree.iterdir()) == [1, 10]


def test_run_b_newest_latest_params_ahead_of_states_trusts_and_pins_the_common_one(tmp_path):
    """The real run B after a SIGINT: params latest {10, 12, 14}, states {9, 10, 12}."""
    from hamilflow.training.supervisor.watchdog import newest_checkpoint_epoch

    for epoch in (10, 12, 14):
        make_checkpoint(tmp_path, "latest", epoch, parts=("params",))
    for epoch in (9, 10, 12):
        make_checkpoint(tmp_path, "latest", epoch, parts=("states",))
    assert last_checkpoint_epoch(tmp_path, settle_s=0.0) == 12
    assert newest_checkpoint_epoch(tmp_path) == 14
    from hamilflow.training.supervisor.policy import MemberAction, _pin_checkpoint

    action = MemberAction(1, "resume", "walltime", "", "running", "resume",
                          continue_from={"previous_output_dir": str(tmp_path), "load_model_type": "latest"})
    _pin_checkpoint(action, ExitInfo("walltime", 1, str(tmp_path), 14, 12, newest_checkpoint_epoch=14))
    assert action.fold_overrides["process"]["train"]["continued"]["load_model_epoch"] == 12


def test_inference_needs_only_the_params_best(tmp_path):
    make_checkpoint(tmp_path, "best", 9, parts=("params",))
    make_checkpoint(tmp_path, "best", 8, parts=("states",))
    (tmp_path / "model" / "states" / "best.pytree" / "epoch_9.orbax-checkpoint-tmp-1").mkdir()
    assert last_checkpoint_epoch(tmp_path, settle_s=0.0, kind="best") is None
    assert last_checkpoint_epoch(tmp_path, settle_s=0.0, kind="best", parts=("params",)) == 9
