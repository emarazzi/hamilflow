"""The supervisor's decision logic, on synthetic DeepH logs only."""

import json
import tomllib

import numpy as np
import pytest
from synthetic_deeph_logs import KEPT_OPTIMIZER, NEW_OPTIMIZER, decay_then_noise, write_log

from hamilflow.training.dataset_split import write_kfold_splits
from hamilflow.training.ensemble import deep_merge, write_ensemble_train_tomls
from hamilflow.training.log_analysis import has_epochs, read_training_curve, replay_reduce_on_plateau
from hamilflow.training.supervisor import (
    DEFAULT_ACTIONS,
    ExitInfo,
    MemberStatus,
    SupervisorConfig,
    append_history,
    carry_over,
    live_check,
    new_history,
    resolve_min_scale,
    round_decision,
)

RUN = "2026-10-22_10-30-00"


def member_log(root, fold, train, round_=1, run=RUN, **kwargs):
    """``root/train_{fold}/outputs_r{round}/<run>/deepx.log``, with the run's model/ folder."""
    run_dir = root / f"train_{fold}" / f"outputs_r{round_}" / run
    (run_dir / "model").mkdir(parents=True, exist_ok=True)
    return write_log(run_dir / "deepx.log", train, **kwargs)


def still_improving(n=150):
    """No onset: the loss falls steadily with little noise."""
    return decay_then_noise(n_epochs=n, flat_from=1000, noise=0.002)


def stepped(stage_vals, stage_len=5, first_scale=1.0):
    """val per stage and the scale halving at each stage."""
    val = np.repeat(stage_vals, stage_len)
    return val, first_scale * np.repeat(0.5 ** np.arange(len(stage_vals)), stage_len)


def converged_log(root, fold, **kwargs):
    val, scale = stepped([1.0, 0.5, 0.499, 0.4985, 0.4984])
    return member_log(root, fold, val, val=val, scale=scale, **kwargs)


def diverged_log(root, fold, **kwargs):
    val = still_improving()
    val[-5:] = 100 * val.min()
    return member_log(root, fold, still_improving(), val=val, **kwargs)


WALLTIME = ExitInfo("walltime", return_code=-2, last_epoch=150, last_checkpoint_epoch=150)


def decide(logs, exits, cfg=None, history=None):
    return round_decision(logs, exits, cfg or SupervisorConfig(), history)


# ---------------------------------- config ----------------------------------


def test_default_policy_is_the_one_agreed_with_the_user():
    cfg = SupervisorConfig()
    assert cfg.actions == {"resume": "auto", "stop_converged": "auto", "retune": "ask", "divergence": "ask",
                           "stop_overfit": "ask", "finished_not_converged": "ask"}
    assert (cfg.poll_s, cfg.walltime_margin, cfg.walltime) == (300.0, 900.0, None)
    assert (cfg.resume_from, cfg.latest_interval) == ("latest", 10)
    assert cfg.shared_overrides()["data"]["model_save"]["latest_interval"] == 10


def test_partial_actions_keep_the_other_defaults():
    cfg = SupervisorConfig(actions={"retune": "auto"})
    assert cfg.actions == {**DEFAULT_ACTIONS, "retune": "auto"}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"actions": {"retrain": "auto"}}, "unknown"),
        ({"actions": {"retune": "yes"}}, "modes must be"),
        ({"resume_from": "last"}, "resume_from"),
        ({"monitor": "best"}, "monitor"),
        ({"divergence_factor": 0}, "divergence_factor"),
        ({"divergence_epochs": 0}, "divergence_epochs"),
        ({"max_rounds": 1.5}, "max_rounds"),
        ({"max_crash_restarts": -1}, "max_crash_restarts"),
        ({"lr_drop_on_divergence": 1.0}, "lr_drop_on_divergence"),
        ({"min_scale": 1.0}, "min_scale must be in"),
        ({"done_scale": 1.0}, "done_scale must be < 1"),
        ({"walltime": 600}, "walltime_margin"),
    ],
)
def test_bad_config_is_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SupervisorConfig(**kwargs)


def test_config_round_trips_through_json():
    cfg = SupervisorConfig(actions={"divergence": "off"}, walltime=172800.0, max_rounds=4)
    again = SupervisorConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert again == cfg
    with pytest.raises(ValueError, match="unknown keys"):
        SupervisorConfig.from_dict({"polls": 3})


def test_exit_info_round_trips_through_json_and_is_validated():
    info = ExitInfo("live_rule", 0, "/runs/x", 120, 110, "diverged", "nan loss")
    assert ExitInfo.from_dict(json.loads(json.dumps(info.to_dict()))) == info
    with pytest.raises(ValueError, match="reason"):
        ExitInfo("killed")
    with pytest.raises(ValueError):
        ExitInfo("live_rule", live_status="exploded")


# ---------------------------------- live rules ----------------------------------


def test_live_check_running(tmp_path):
    assert live_check([write_log(tmp_path / "deepx.log", still_improving())], SupervisorConfig()) == (
        MemberStatus.RUNNING, "", False)


def test_live_check_before_the_first_epoch(tmp_path):
    path = tmp_path / "deepx.log"
    path.write_text("[ 10.22-10:30:47 ] [dataloader] Train size: `40`. Val size: `10`. Test size: `0`.\n")
    assert live_check([path, tmp_path / "missing.log"], SupervisorConfig())[0] == MemberStatus.RUNNING


def test_w1_nan_loss_is_divergence(tmp_path):
    val = still_improving(30)
    val[20] = np.nan
    log = write_log(tmp_path / "deepx.log", still_improving(30), val=val)
    status, reason, _ = live_check([log], SupervisorConfig())
    assert status == MemberStatus.DIVERGED
    assert "epoch 21" in reason


def test_w1_val_far_above_the_best_for_n_epochs(tmp_path):
    status, reason, _ = live_check([diverged_log(tmp_path, 1)], SupervisorConfig())
    assert status == MemberStatus.DIVERGED and "last 5 epochs" in reason
    # 4 bad epochs out of 5 required: not yet
    val = still_improving()
    val[-4:] = 100 * val.min()
    assert live_check([write_log(tmp_path / "b.log", still_improving(), val=val)], SupervisorConfig())[0] == (
        MemberStatus.RUNNING)


def test_w1_ignores_a_nan_in_an_earlier_job(tmp_path):
    """The earlier job's divergence was already restarted from: only the last job counts."""
    val = still_improving(30)
    val[10] = np.nan
    first = member_log(tmp_path, 1, still_improving(30), val=val)
    second = member_log(tmp_path, 1, still_improving(30), round_=2, continued_from=f"/x/{RUN}",
                        load_model_type="latest", continuation=KEPT_OPTIMIZER, first_epoch=31)
    assert live_check([first, second], SupervisorConfig())[0] == MemberStatus.RUNNING


def test_w2_converged_and_overfit(tmp_path):
    assert live_check([converged_log(tmp_path, 1)], SupervisorConfig())[0] == MemberStatus.CONVERGED
    val, scale = stepped([1.0, 0.9, 0.85])
    path = write_log(tmp_path / "over.log", np.repeat([1.0, 0.5, 0.3], 5), val=val, scale=scale)
    status, reason, _ = live_check([path], SupervisorConfig())
    assert status == MemberStatus.OVERFIT and "val/train grew" in reason


def test_w3_patience_too_large(tmp_path):
    """Still at the initial LR 150 epochs after the onset (~250) with patience 120: mistuned."""
    status, reason, _ = live_check([write_log(tmp_path / "deepx.log", decay_then_noise())], SupervisorConfig())
    assert status == MemberStatus.MISTUNED
    assert reason.startswith("patience too large")


def test_w3_patience_too_small(tmp_path):
    val = 10 ** (-2 - 0.01 * np.arange(300))
    path = write_log(tmp_path / "deepx.log", val, scale=np.r_[np.ones(119), np.full(181, 0.5)])
    status, reason, _ = live_check([path], SupervisorConfig())
    assert status == MemberStatus.MISTUNED
    assert reason.startswith("patience too small: first reduction at epoch 120")


def test_w3_waits_for_min_epochs_before_rules(tmp_path):
    path = write_log(tmp_path / "deepx.log", decay_then_noise())
    assert live_check([path], SupervisorConfig(min_epochs_before_rules=401))[0] == MemberStatus.RUNNING


def test_w3_only_judges_the_settings_a_scratch_run_started_with(tmp_path):
    # a continued first log: not a run from scratch
    path = write_log(tmp_path / "deepx.log", decay_then_noise(), continued_from="/x/run", load_model_type="latest",
                     continuation=KEPT_OPTIMIZER)
    assert live_check([path], SupervisorConfig())[0] == MemberStatus.RUNNING
    # retuned in a later job (another patience): W3 is off
    first = member_log(tmp_path, 1, decay_then_noise())
    second = member_log(tmp_path, 1, np.full(5, decay_then_noise()[-1]), round_=2, patience=25, cooldown=12,
                        continued_from=f"/x/{RUN}", lr0="0.001")
    assert live_check([first, second], SupervisorConfig())[0] == MemberStatus.RUNNING


# ---------------------------------- LR carry-over ----------------------------------


def test_carry_over_keeps_the_absolute_lr_and_the_final_floor(tmp_path):
    val, scale = stepped([1.0, 0.5, 0.25])  # last scale 0.25 with lr0 1e-3
    curve = read_training_curve(write_log(tmp_path / "deepx.log", val, val=val, scale=scale))
    cfg = SupervisorConfig(min_scale=1e-4)
    lr, min_scale = carry_over(curve, cfg)
    assert lr == pytest.approx(2.5e-4)
    assert min_scale == pytest.approx(4e-4)
    assert lr * min_scale == pytest.approx(1e-3 * 1e-4)  # the run still ends at lr0_first x min_scale
    lr, min_scale = carry_over(curve, cfg, drop=0.5)
    assert (lr, min_scale) == (pytest.approx(1.25e-4), pytest.approx(8e-4))


def test_carry_over_through_a_chain_with_a_new_optimizer(tmp_path):
    first = member_log(tmp_path, 1, np.ones(2), scale=[1.0, 0.25])
    second = member_log(tmp_path, 1, np.ones(2), round_=2, scale=[1.0, 0.5], lr0="0.00025",
                        continued_from=f"/x/{RUN}", load_model_type="latest")
    lr, min_scale = carry_over(read_training_curve([first, second]), SupervisorConfig(min_scale=1e-4))
    assert lr == pytest.approx(1.25e-4)
    assert min_scale == pytest.approx(1e-4 / 0.125)


def test_carry_over_at_the_floor_is_impossible(tmp_path):
    curve = read_training_curve(write_log(tmp_path / "deepx.log", np.ones(2), scale=[1.0, 1e-4]))
    assert carry_over(curve, SupervisorConfig(min_scale=1e-4)) is None
    assert carry_over(curve, SupervisorConfig(min_scale=1e-5)) is not None


# ---------------------------------- round decision: one rule each ----------------------------------


def test_walltime_resumes_from_latest_with_the_current_settings(tmp_path):
    log = member_log(tmp_path, 1, still_improving(), patience=30, cooldown=15)
    action = decide({1: [log]}, {1: WALLTIME}).actions[1]
    assert (action.action, action.cause, action.status, action.needs_approval) == (
        "resume", "walltime", "stopped_walltime", False)
    assert action.continue_from == {"previous_output_dir": str(log.parent), "load_model_type": "latest"}
    train = action.fold_overrides["process"]["train"]
    assert train["continued"] == {"new_optimizer": False}
    assert train["optimizer"]["init_learning_rate"] == 1e-3
    assert train["scheduler"] == {"min_learning_rate_scale": 1e-4, "patience": 30, "cooldown": 15, "rtol": 0.05}


def test_plain_resume_keeps_a_retuned_members_lr0_and_rescaled_floor(tmp_path):
    first = member_log(tmp_path, 1, np.ones(2), scale=[1.0, 0.25])
    second = member_log(tmp_path, 1, still_improving(), round_=2, lr0="0.00025", patience=25,
                        continued_from=f"/x/{RUN}")
    action = decide({1: [first, second]}, {1: WALLTIME}).actions[1]
    train = action.fold_overrides["process"]["train"]
    assert train["optimizer"]["init_learning_rate"] == 2.5e-4
    assert train["scheduler"]["min_learning_rate_scale"] == pytest.approx(4e-4)
    assert train["scheduler"]["patience"] == 25


def test_resume_from_best_when_configured(tmp_path):
    log = member_log(tmp_path, 1, still_improving())
    action = decide({1: [log]}, {1: WALLTIME}, SupervisorConfig(resume_from="best")).actions[1]
    assert action.continue_from["load_model_type"] == "best"


def test_crash_resumes_from_latest_until_the_cap(tmp_path):
    log = member_log(tmp_path, 1, still_improving())
    cfg = SupervisorConfig(max_crash_restarts=1, resume_from="best")
    first = decide({1: [log]}, {1: None}, cfg)
    action = first.actions[1]
    assert (action.action, action.cause, action.status) == ("resume", "crash", "crashed")
    assert action.continue_from["load_model_type"] == "latest"  # the last checkpoint, whatever resume_from
    assert "the job never returned" in action.reason
    again = decide({1: [log]}, {1: ExitInfo("crashed", return_code=1)}, cfg, append_history(new_history(), first))
    assert (again.actions[1].action, again.actions[1].cause) == ("give_up", "crash")
    assert "after 1 consecutive crash restarts" in again.actions[1].reason


def test_crash_before_the_first_epoch_restarts_from_scratch(tmp_path):
    action = decide({1: [tmp_path / "never_written.log"]}, {1: None}).actions[1]
    assert action.action == "resume" and action.continue_from is None


def test_converged_member_is_done(tmp_path):
    action = decide({1: [converged_log(tmp_path, 1)]}, {1: WALLTIME}).actions[1]
    assert (action.action, action.cause, action.needs_approval, action.policy) == (
        "done", "converged", False, "stop_converged")


def test_overfit_member_is_done_after_approval(tmp_path):
    val, scale = stepped([1.0, 0.9, 0.85])
    log = member_log(tmp_path, 1, np.repeat([1.0, 0.5, 0.3], 5), val=val, scale=scale)
    action = decide({1: [log]}, {1: WALLTIME}).actions[1]
    assert (action.action, action.cause, action.needs_approval, action.overfitting) == ("done", "overfit", True, True)


def test_diverged_member_restarts_from_best_at_a_lower_lr(tmp_path):
    log = diverged_log(tmp_path, 1)
    action = decide({1: [log]}, {1: WALLTIME}).actions[1]
    assert (action.action, action.cause, action.needs_approval) == ("restart_lower_lr", "diverged", True)
    assert action.continue_from == {"previous_output_dir": str(log.parent), "load_model_type": "best"}
    train = action.fold_overrides["process"]["train"]
    assert train["continued"] == {"new_optimizer": True}
    assert train["optimizer"]["init_learning_rate"] == pytest.approx(5e-4)  # 1e-3 x scale 1 x 0.5
    assert train["scheduler"]["min_learning_rate_scale"] == pytest.approx(2e-4)


def test_divergence_reported_by_the_watchdog_counts(tmp_path):
    log = member_log(tmp_path, 1, still_improving())
    info = ExitInfo("live_rule", live_status="diverged", live_reason="val spike")
    action = decide({1: [log]}, {1: info}).actions[1]
    assert action.action == "restart_lower_lr" and "watchdog: val spike" in action.reason


def test_divergence_restarts_are_capped(tmp_path):
    log = diverged_log(tmp_path, 1)
    cfg = SupervisorConfig(max_divergence_restarts=1, actions={"divergence": "auto"})
    first = decide({1: [log]}, {1: WALLTIME}, cfg)
    assert first.actions[1].action == "restart_lower_lr"
    second = decide({1: [log]}, {1: WALLTIME}, cfg, append_history(new_history(), first))
    assert (second.actions[1].action, second.actions[1].cause) == ("give_up", "diverged")


def test_mistuned_member_is_retuned_from_best_with_the_pooled_suggestion(tmp_path):
    log = member_log(tmp_path, 1, decay_then_noise())  # patience 120, still at scale 1 at epoch 400
    decision = decide({1: [log]}, {1: ExitInfo("live_rule", live_status="mistuned")})
    action, suggested = decision.actions[1], decision.analysis.suggested
    assert (action.action, action.cause, action.needs_approval) == ("retune", "mistuned", True)
    assert action.continue_from["load_model_type"] == "best"
    train = action.fold_overrides["process"]["train"]
    assert train["continued"] == {"new_optimizer": True}
    assert train["scheduler"]["patience"] == suggested.patience
    assert train["scheduler"]["cooldown"] == suggested.cooldown
    assert train["scheduler"]["rtol"] == suggested.rtol
    assert train["optimizer"]["init_learning_rate"] == pytest.approx(1e-3)
    assert train["scheduler"]["min_learning_rate_scale"] == pytest.approx(1e-4)


def test_retune_when_the_suggestion_differs_even_without_w3(tmp_path):
    """A continued first log is not judged by W3, but the pooled suggestion is far from patience 120."""
    log = member_log(tmp_path, 1, decay_then_noise(), continued_from="/x/run", load_model_type="latest",
                     continuation=KEPT_OPTIMIZER)
    action = decide({1: [log]}, {1: WALLTIME}).actions[1]
    assert (action.action, action.cause) == ("retune", "retune")
    assert "pooled suggestion patience" in action.reason
    # within the threshold: no retune
    suggested = decide({1: [log]}, {1: WALLTIME}).analysis.suggested
    near = member_log(tmp_path, 2, decay_then_noise(), patience=suggested.patience, continued_from="/x/run",
                      load_model_type="latest", continuation=KEPT_OPTIMIZER)
    assert decide({2: [near]}, {2: WALLTIME}).actions[2].action == "resume"


def test_finished_without_converging_is_given_up_after_approval(tmp_path):
    log = member_log(tmp_path, 1, still_improving())
    action = decide({1: [log]}, {1: ExitInfo("finished", return_code=0)}).actions[1]
    assert (action.action, action.cause, action.status, action.needs_approval) == (
        "give_up", "finished", "finished_not_converged", True)


def test_member_at_the_lr_floor_cannot_resume(tmp_path):
    val = still_improving()
    log = member_log(tmp_path, 1, val, scale=np.r_[np.ones(149), 1e-3])
    # with the default done_scale (1e-3 > min_scale) a member at the floor is converged: lower it
    cfg = SupervisorConfig(min_scale=1e-3, done_scale=1e-5, actions={"retune": "off"})
    action = decide({1: [log]}, {1: WALLTIME}, cfg).actions[1]
    assert (action.action, action.cause) == ("give_up", "finished")
    assert "at the LR floor" in action.reason


# ---------------------------------- policy gating ----------------------------------


@pytest.mark.parametrize(
    ("policy", "make", "expected"),
    [
        ("stop_converged", converged_log, "resume"),
        ("resume", lambda root, fold: member_log(root, fold, still_improving()), "done"),
    ],
)
def test_off_replaces_an_action_by_not_acting(tmp_path, policy, make, expected):
    action = decide({1: [make(tmp_path, 1)]}, {1: WALLTIME}, SupervisorConfig(actions={policy: "off"})).actions[1]
    assert action.action == expected
    assert action.reason.startswith(f"{policy} is off")


def test_off_for_retune_and_overfit_resumes_and_for_finished_is_done(tmp_path):
    log = member_log(tmp_path, 1, decay_then_noise())
    info = ExitInfo("live_rule", live_status="mistuned")
    assert decide({1: [log]}, {1: info}, SupervisorConfig(actions={"retune": "off"})).actions[1].action == "resume"
    val, scale = stepped([1.0, 0.9, 0.85])
    over = member_log(tmp_path, 2, np.repeat([1.0, 0.5, 0.3], 5), val=val, scale=scale)
    action = decide({2: [over]}, {2: WALLTIME}, SupervisorConfig(actions={"stop_overfit": "off"})).actions[2]
    assert (action.action, action.overfitting) == ("resume", True)
    log3 = member_log(tmp_path, 3, still_improving())
    action = decide({3: [log3]}, {3: ExitInfo("finished")},
                    SupervisorConfig(actions={"finished_not_converged": "off"})).actions[3]
    assert action.action == "done"


def test_off_falls_back_to_a_resume_that_is_itself_gated(tmp_path):
    cfg = SupervisorConfig(actions={"stop_converged": "off", "resume": "ask"})
    action = decide({1: [converged_log(tmp_path, 1)]}, {1: WALLTIME}, cfg).actions[1]
    assert (action.action, action.needs_approval) == ("resume", True)


def test_divergence_off_gives_up(tmp_path):
    """Resuming a diverged run is pointless (orchestrator's decision)."""
    action = decide({1: [diverged_log(tmp_path, 1)]}, {1: WALLTIME},
                    SupervisorConfig(actions={"divergence": "off"})).actions[1]
    assert (action.action, action.cause, action.mode) == ("give_up", "diverged", "off")
    assert action.reason.startswith("divergence is off")


@pytest.mark.parametrize(("mode", "approval"), [("auto", False), ("ask", True)])
def test_auto_and_ask(tmp_path, mode, approval):
    decision = decide({1: [diverged_log(tmp_path, 1)]}, {1: WALLTIME}, SupervisorConfig(actions={"divergence": mode}))
    assert decision.actions[1].action == "restart_lower_lr"
    assert decision.actions[1].mode == mode
    assert decision.needs_approval is approval


# ---------------------------------- limits and history ----------------------------------


def test_max_rounds_gives_up_the_members_still_training(tmp_path):
    logs = {1: [member_log(tmp_path, 1, still_improving())], 2: [converged_log(tmp_path, 2)]}
    exits = {1: WALLTIME, 2: WALLTIME}
    cfg = SupervisorConfig(max_rounds=2)
    first = decide(logs, exits, cfg)
    assert first.round == 1 and first.actions[1].action == "resume"
    second = decide(logs, {1: WALLTIME}, cfg, append_history(new_history(), first))
    assert second.round == 2
    assert (second.actions[1].action, second.actions[1].cause) == ("give_up", "max_rounds")
    assert second.actions[2].action == "done"  # not trained this round: done in round 1
    assert second.finished


def test_member_given_up_earlier_stays_given_up(tmp_path):
    logs = {1: [member_log(tmp_path, 1, still_improving())]}
    first = decide(logs, {1: ExitInfo("finished")}, SupervisorConfig(actions={"finished_not_converged": "auto"}))
    assert first.actions[1].action == "give_up"
    second = decide(logs, {}, history=append_history(new_history(), first))
    assert (second.actions[1].action, second.actions[1].cause) == ("give_up", "earlier")


def test_exit_infos_as_dicts_and_unknown_folds(tmp_path):
    log = member_log(tmp_path, 1, still_improving())
    assert decide({1: [log]}, {1: WALLTIME.to_dict()}).actions[1].action == "resume"
    with pytest.raises(ValueError, match="not in member_logs"):
        decide({1: [log]}, {2: WALLTIME})


def test_history_is_plain_json(tmp_path):
    decision = decide({1: [diverged_log(tmp_path, 1)], 2: [converged_log(tmp_path, 2)]}, {1: None, 2: WALLTIME})
    history = append_history(new_history(), decision)
    assert json.loads(json.dumps(history)) == history
    record = history["rounds"][0]["members"]["1"]
    assert (record["action"], record["cause"], record["status"]) == ("resume", "crash", "crashed")
    assert history["rounds"][0]["round"] == 1


# ---------------------------------- end to end ----------------------------------


def test_three_members_in_different_states_write_the_next_round(tmp_path):
    cycle = tmp_path / "cycle"
    write_kfold_splits([f"family{f}-{v:03d}" for f in range(3) for v in range(4)], 3, cycle)
    logs = {
        1: [converged_log(cycle, 1)],
        2: [member_log(cycle, 2, still_improving(), patience=30, cooldown=15)],
        3: [diverged_log(cycle, 3)],
    }
    exits = {1: WALLTIME, 2: WALLTIME, 3: ExitInfo("live_rule", live_status="diverged", live_reason="val spike")}
    cfg = SupervisorConfig(actions={"divergence": "auto"})
    decision = round_decision(logs, exits, cfg, new_history())
    assert {j: a.action for j, a in decision.actions.items()} == {1: "done", 2: "resume", 3: "restart_lower_lr"}
    assert decision.pending_folds == [2, 3] and not decision.needs_approval and not decision.finished

    base = {"data": {"inputs_dir": "/inputs", "outputs_dir": "./outputs"},
            "process": {"train": {"continued": {"new_optimizer": True}}}}  # shared settings, as in the example
    paths = write_ensemble_train_tomls(
        deep_merge(base, decision.shared_overrides), cycle, 3, round=2, folds=decision.pending_folds,
        continue_from=decision.continue_from, fold_overrides=decision.fold_overrides,
    )
    resumed, restarted = (tomllib.loads(p.read_text()) for p in paths)
    assert resumed["data"]["outputs_dir"] == "./outputs_r2"
    assert resumed["data"]["model_save"]["latest_interval"] == 10
    assert resumed["process"]["train"]["continued"]["new_optimizer"] is False  # the fold override wins
    assert resumed["process"]["train"]["continued"]["load_model_type"] == "latest"
    assert resumed["process"]["train"]["continued"]["previous_output_dir"] == str(logs[2][0].parent.resolve())
    assert resumed["process"]["train"]["scheduler"]["patience"] == 30
    assert restarted["process"]["train"]["continued"]["new_optimizer"] is True
    assert restarted["process"]["train"]["continued"]["load_model_type"] == "best"
    assert restarted["process"]["train"]["optimizer"]["init_learning_rate"] == pytest.approx(5e-4)
    assert json.loads(json.dumps(append_history(new_history(), decision)))["rounds"][0]["round"] == 1


def test_new_optimizer_continuation_used_by_a_restart_is_parsed(tmp_path):
    """The continuation line written by a restart (new optimizer, best) keeps the chain continuous."""
    first = member_log(tmp_path, 1, np.ones(4), val=[4.0, 3.0, 2.0, 1.0], scale=[1, 1, 0.5, 0.5])
    second = member_log(tmp_path, 1, np.ones(2), round_=2, lr0="0.0005", continued_from=f"/x/{RUN}",
                        continuation=NEW_OPTIMIZER)
    assert live_check([first, second], SupervisorConfig())[0] == MemberStatus.RUNNING


def test_retune_or_divergence_restart_at_the_lr_floor_gives_up(tmp_path):
    """The carried-over min scale would be >= 1: neither a retune nor a restart can continue."""
    cfg = SupervisorConfig(min_scale=1e-3, done_scale=1e-5, actions={"retune": "auto", "divergence": "auto"})
    floor = np.r_[np.ones(399), 1e-3]
    log = member_log(tmp_path, 1, decay_then_noise(), scale=floor)  # pooled suggestion far from patience 120
    action = decide({1: [log]}, {1: WALLTIME}, cfg).actions[1]
    assert (action.action, action.cause) == ("give_up", "finished")
    # the best-val epoch, the checkpoint a divergence restart loads, is at the floor
    val = 10 ** (-0.01 * np.arange(400))
    val[-5:] *= 1e4
    log = member_log(tmp_path, 2, val, scale=np.r_[np.ones(300), np.full(100, 1e-3)])
    action = decide({2: [log]}, {2: WALLTIME}, cfg).actions[2]
    assert (action.action, action.cause) == ("give_up", "diverged")
    assert "no LR to restart from" in action.reason


# ---------------------------------- divergence LR, crashes, W3, min_scale, hard stop ----------------------------


def test_divergence_restart_carries_the_lr_at_the_best_epoch_not_the_decayed_one(tmp_path):
    v = decay_then_noise(850, flat_from=200, seed=1)
    v[300:] *= 1000
    s = replay_reduce_on_plateau(np.where(np.isfinite(v), v, np.inf), 30, 15, 0.05, min_scale=1e-4)
    log = member_log(tmp_path, 1, v, scale=s, patience=30, cooldown=15)
    a = decide({1: [log]}, {1: WALLTIME}, SupervisorConfig(actions={"divergence": "auto"})).actions[1]
    assert a.action == "restart_lower_lr"
    lr = a.fold_overrides["process"]["train"]["optimizer"]["init_learning_rate"]
    assert lr == pytest.approx(1e-3 * s[int(np.argmin(v))] * 0.5)
    min_scale = a.fold_overrides["process"]["train"]["scheduler"]["min_learning_rate_scale"]
    assert lr * min_scale == pytest.approx(1e-3 * 1e-4)  # the same absolute floor


def test_carry_over_needs_a_resolved_min_scale(tmp_path):
    curve = read_training_curve(write_log(tmp_path / "deepx.log", np.ones(2)))
    with pytest.raises(ValueError, match="resolve_min_scale"):
        carry_over(curve, SupervisorConfig())


def retuned_round_1(tmp_path):
    """Round 1: a mistuned member retuned (auto) from best; returns its log, the decision and the history."""
    log = member_log(tmp_path, 1, decay_then_noise())
    decision = decide({1: [log]}, {1: ExitInfo("live_rule", live_status="mistuned")},
                      SupervisorConfig(actions={"retune": "auto"}))
    assert decision.actions[1].action == "retune"
    return log, decision, append_history(new_history(), decision)


def test_crash_before_any_epoch_repeats_the_previous_action(tmp_path):
    log, first, history = retuned_round_1(tmp_path)
    header_only = tmp_path / "train_1" / "outputs_r2" / RUN / "deepx.log"
    header_only.parent.mkdir(parents=True)
    header_only.write_text("[ 10.22-10:30:47 ] [dataloader] Train size: `40`. Val size: `10`. Test size: `0`.\n")
    action = decide({1: [log, header_only]}, {1: None}, SupervisorConfig(), history).actions[1]
    assert (action.action, action.cause, action.status) == ("resume", "crash", "crashed")
    assert action.continue_from == first.actions[1].continue_from
    assert action.fold_overrides == first.actions[1].fold_overrides
    assert "before its first latest checkpoint" in action.reason


def test_crash_without_a_checkpoint_per_exit_info_repeats_the_previous_action(tmp_path):
    log, first, history = retuned_round_1(tmp_path)
    second = member_log(tmp_path, 1, still_improving(8), round_=2, continued_from=f"/x/{RUN}",
                        continuation=NEW_OPTIMIZER)
    info = ExitInfo("crashed", return_code=1, last_epoch=8, last_checkpoint_epoch=None)
    action = decide({1: [log, second]}, {1: info}, SupervisorConfig(), history).actions[1]
    assert action.fold_overrides == first.actions[1].fold_overrides


def test_crash_after_a_checkpoint_of_a_retuned_run_resumes_it_with_its_own_settings(tmp_path):
    log, first, history = retuned_round_1(tmp_path)
    retuned = first.actions[1].fold_overrides["process"]["train"]
    lr0 = retuned["optimizer"]["init_learning_rate"]
    second = member_log(tmp_path, 1, still_improving(15), round_=2, lr0=str(lr0),
                        patience=retuned["scheduler"]["patience"], cooldown=retuned["scheduler"]["cooldown"],
                        continued_from=f"/x/{RUN}", continuation=NEW_OPTIMIZER)
    for info in (None, ExitInfo("crashed", return_code=1, last_epoch=15, last_checkpoint_epoch=10)):
        action = decide({1: [log, second]}, {1: info}, SupervisorConfig(), history).actions[1]
        assert (action.action, action.cause) == ("resume", "crash")
        assert action.continue_from == {"previous_output_dir": str(second.parent), "load_model_type": "latest"}
        train = action.fold_overrides["process"]["train"]
        assert train["continued"] == {"new_optimizer": False}
        assert train["optimizer"]["init_learning_rate"] == lr0
        assert train["scheduler"]["patience"] == retuned["scheduler"]["patience"]
        assert train["scheduler"]["cooldown"] == retuned["scheduler"]["cooldown"]
        assert train["scheduler"]["min_learning_rate_scale"] == pytest.approx(
            retuned["scheduler"]["min_learning_rate_scale"])


def test_crash_in_round_1_without_epochs_starts_as_round_1_did(tmp_path):
    missing = tmp_path / "missing.log"
    action = decide({1: [missing]}, {1: None}).actions[1]
    assert (action.continue_from, action.fold_overrides) == (None, {})
    start = {"1": {"continue_from": {"previous_output_dir": "/prev/run", "new_training_data": True},
                   "fold_overrides": {"system": {"random_seed": 3}}}}
    action = decide({1: [missing]}, {1: None}, history={"start": start, "rounds": []}).actions[1]
    assert action.continue_from == start["1"]["continue_from"]
    assert action.fold_overrides == start["1"]["fold_overrides"]


def test_crash_count_resets_after_a_round_without_crash(tmp_path):
    log = member_log(tmp_path, 1, still_improving())
    cfg = SupervisorConfig(max_crash_restarts=1)
    history = new_history()
    for info, expected in [(None, "resume"), (WALLTIME, "resume"), (None, "resume"), (None, "give_up")]:
        decision = decide({1: [log]}, {1: info}, cfg, history)
        assert decision.actions[1].action == expected
        history = append_history(history, decision)


def test_w3_is_off_after_a_new_optimizer_restart_with_the_same_settings(tmp_path):
    first = write_log(tmp_path / "r1" / "deepx.log", decay_then_noise(400, seed=1), patience=120, cooldown=50)
    assert live_check([first], SupervisorConfig())[0] == MemberStatus.MISTUNED
    flat = 1e-4 * (1 + 0.05 * np.random.default_rng(5).standard_normal(30))
    second = write_log(tmp_path / "r2" / "deepx.log", flat, patience=120, cooldown=50, lr0="0.0005",
                       continued_from=str(tmp_path / "r1"), load_model_type="best", continuation=NEW_OPTIMIZER)
    assert live_check([first, second], SupervisorConfig())[0] == MemberStatus.RUNNING


def test_w3_too_small_counts_a_lower_lr_at_a_join(tmp_path):
    """A later job at a lower LR with the optimizer kept also leaves the initial LR (reductions leave joins out)."""
    val = 10 ** (-2 - 0.01 * np.arange(300))
    first = member_log(tmp_path, 1, val[:119])
    second = member_log(tmp_path, 1, val[119:], round_=2, lr0="0.0005", continued_from=f"/x/{RUN}",
                        load_model_type="latest", continuation=KEPT_OPTIMIZER, first_epoch=120)
    status, reason, _ = live_check([first, second], SupervisorConfig())
    assert status == MemberStatus.MISTUNED
    assert reason.startswith("patience too small: first reduction at epoch 120")


def test_no_retune_to_the_current_settings(tmp_path):
    probe = member_log(tmp_path, 9, decay_then_noise())
    s = decide({9: [probe]}, {9: WALLTIME}).analysis.suggested
    log = member_log(tmp_path, 1, decay_then_noise(), patience=s.patience, cooldown=s.cooldown, rtol=s.rtol)
    info = ExitInfo("live_rule", live_status="mistuned", live_reason="too large")
    action = decide({1: [log]}, {1: info}, SupervisorConfig(actions={"retune": "auto"})).actions[1]
    assert action.action == "resume"
    assert "the suggestion is the current settings, no retune" in action.reason


def test_min_scale_comes_from_the_shared_train_overrides(tmp_path):
    shared = {"process": {"train": {"scheduler": {"min_learning_rate_scale": 1e-7}}}}
    first = member_log(tmp_path, 1, np.ones(2), scale=[1.0, 0.25])
    second = member_log(tmp_path, 1, still_improving(), round_=2, lr0="0.00025", continued_from=f"/x/{RUN}")
    decision = round_decision({1: [first, second]}, {1: WALLTIME}, SupervisorConfig(), shared_overrides=shared)
    scheduler = decision.actions[1].fold_overrides["process"]["train"]["scheduler"]
    assert scheduler["min_learning_rate_scale"] == pytest.approx(1e-7 / 0.25)
    # without shared overrides: DeepH's default
    assert resolve_min_scale(SupervisorConfig(), None).min_scale == 1e-4
    assert resolve_min_scale(SupervisorConfig(min_scale=1e-7), shared).min_scale == 1e-7
    with pytest.raises(ValueError, match="leave min_scale None"):
        round_decision({1: [first, second]}, {1: WALLTIME}, SupervisorConfig(min_scale=1e-4), shared_overrides=shared)


def test_non_finite_loss_is_a_hard_stop_the_factor_rule_is_not(tmp_path):
    val = still_improving(30)
    val[20] = np.inf
    check = live_check([write_log(tmp_path / "nan.log", still_improving(30), val=val)], SupervisorConfig())
    assert (check.status, check.hard_stop) == (MemberStatus.DIVERGED, True)
    check = live_check([diverged_log(tmp_path, 1)], SupervisorConfig())
    assert (check.status, check.hard_stop) == (MemberStatus.DIVERGED, False)
    assert not live_check([converged_log(tmp_path, 2)], SupervisorConfig()).hard_stop


def test_has_epochs(tmp_path):
    assert has_epochs(write_log(tmp_path / "a.log", [1.0]))
    assert not has_epochs(tmp_path / "missing.log")
    (tmp_path / "b.log").write_text("no epochs\n")
    assert not has_epochs(tmp_path / "b.log")


def test_live_check_waits_for_the_first_epoch_of_the_current_run(tmp_path):
    v = decay_then_noise(300, seed=1)
    v[-3:] = np.nan
    first = write_log(tmp_path / "r1" / "deepx.log", v, patience=40, cooldown=20)
    second = write_log(tmp_path / "r2" / "deepx.log", [1.0], lr0="0.0005", patience=40, cooldown=20,
                       continued_from=str(tmp_path / "r1"), continuation=NEW_OPTIMIZER)
    second.write_text("\n".join(l for l in second.read_text().splitlines() if "[train] Epoch" not in l) + "\n")
    check = live_check([first, second], SupervisorConfig())
    assert check.status == MemberStatus.RUNNING and not check.hard_stop


def test_live_check_before_the_current_log_exists(tmp_path):
    """The run dir of the current run is not there yet: the previous run's nan is not judged again."""
    v = still_improving(30)
    v[-1] = np.nan
    first = write_log(tmp_path / "r1" / "deepx.log", still_improving(30), val=v)
    assert live_check([first], SupervisorConfig()).hard_stop
    check = live_check([first, tmp_path / "r2" / "deepx.log"], SupervisorConfig())
    assert check == (MemberStatus.RUNNING, "no epoch in the current run yet", False)


# ---------------------------------- stops without a checkpoint, pinning, W3 suppression ----------------------------


def test_stop_before_the_first_latest_starts_the_run_again_as_before(tmp_path):
    """The reviewer's case: round 1 retunes, round 2 is stopped after 3 epochs, no checkpoint yet."""
    log, first, history = retuned_round_1(tmp_path)
    retuned = first.actions[1].fold_overrides["process"]["train"]
    second = member_log(tmp_path, 1, np.full(3, decay_then_noise()[-1]), round_=2,
                        lr0=str(retuned["optimizer"]["init_learning_rate"]),
                        patience=retuned["scheduler"]["patience"], cooldown=retuned["scheduler"]["cooldown"],
                        continued_from=f"/x/{RUN}", continuation=NEW_OPTIMIZER)
    info = ExitInfo("stopped", 130, str(second.parent), 3, None)
    action = decide({1: [log, second]}, {1: info}, SupervisorConfig(), history).actions[1]
    assert (action.action, action.cause) == ("resume", "stopped")  # not a crash
    assert action.continue_from == first.actions[1].continue_from
    assert action.fold_overrides == first.actions[1].fold_overrides
    assert "no latest checkpoint in this run" in action.reason


def test_stop_before_the_first_epoch_starts_the_run_again_as_before(tmp_path):
    log, first, history = retuned_round_1(tmp_path)
    info = ExitInfo("walltime", -2, None, None, None)
    action = decide({1: [log, tmp_path / "not_there.log"]}, {1: info}, SupervisorConfig(), history).actions[1]
    assert action.fold_overrides == first.actions[1].fold_overrides


def test_resume_pins_the_trusted_checkpoint_when_a_newer_one_is_untrusted(tmp_path):
    log = member_log(tmp_path, 1, still_improving())
    pinned = ExitInfo("walltime", -2, str(log.parent), 150, 140, newest_checkpoint_epoch=150)
    action = decide({1: [log]}, {1: pinned}).actions[1]
    assert action.fold_overrides["process"]["train"]["continued"]["load_model_epoch"] == 140
    assert action.continue_from["load_model_type"] == "latest"
    same = ExitInfo("walltime", -2, str(log.parent), 150, 150, newest_checkpoint_epoch=150)
    assert "load_model_epoch" not in decide({1: [log]}, {1: same}).actions[1].fold_overrides["process"]["train"][
        "continued"]
    crash = ExitInfo("crashed", 1, str(log.parent), 150, 140, newest_checkpoint_epoch=150)
    action = decide({1: [log]}, {1: crash}).actions[1]
    assert (action.cause, action.fold_overrides["process"]["train"]["continued"]["load_model_epoch"]) == ("crash", 140)


def two_members_one_mistuned(root):
    """Fold 1: patience 120, still at the initial LR well after its onset (W3); fold 2 still improves steadily
    over 400 epochs, so it never reduces at any patience and the pooled median of the two is "never": no
    pooled suggestion. (A member too short to judge is left out of the pool instead.)"""
    logs = {1: [member_log(root, 1, decay_then_noise())], 2: [member_log(root, 2, still_improving(400))]}
    exits = {1: ExitInfo("live_rule", 0, str(logs[1][0].parent), 400, 400, "mistuned", "patience too large"),
             2: ExitInfo("walltime", -2, str(logs[2][0].parent), 400, 400)}
    return logs, exits


def test_w3_the_pool_cannot_act_on_is_suppressed_for_the_next_round(tmp_path):
    logs, exits = two_members_one_mistuned(tmp_path)
    cfg = SupervisorConfig(actions={"retune": "auto"})
    decision = decide(logs, exits, cfg)
    assert decision.analysis.suggested is None
    assert decision.actions[1].action == "resume"
    assert decision.live_actions == {1: {"retune": "off"}, 2: {}}
    record = append_history(new_history(), decision)["rounds"][0]["members"]["1"]
    assert record["live_actions"] == {"retune": "off"}  # kept in state.json
    # retune ask: the watchdog only alerts, nothing to suppress
    assert decide(logs, exits, SupervisorConfig()).live_actions[1] == {}


def test_w3_suppression_ends_with_the_retune_once_the_pool_can_act(tmp_path):
    logs, exits = two_members_one_mistuned(tmp_path)
    cfg = SupervisorConfig(actions={"retune": "auto"})
    history = append_history(new_history(), decide(logs, exits, cfg))
    # round 2: fold 2 reached its plateau, fold 1 trained on (W3 suppressed) to walltime
    logs[1].append(member_log(tmp_path, 1, decay_then_noise()[-50:], round_=2, continued_from=f"/x/{RUN}",
                              load_model_type="latest", continuation=KEPT_OPTIMIZER, first_epoch=401))
    plateau = still_improving(400)[-1] * 10 ** (0.02 * np.random.default_rng(3).standard_normal(300))
    logs[2].append(member_log(tmp_path, 2, plateau, round_=2, continued_from=f"/x/{RUN}", load_model_type="latest",
                              continuation=KEPT_OPTIMIZER, first_epoch=401))
    exits = {j: ExitInfo("walltime", -2, str(logs[j][-1].parent), 400, 400) for j in (1, 2)}
    decision = decide(logs, exits, cfg, history)
    assert decision.analysis.suggested is not None
    assert decision.actions[1].action == "retune"
    assert decision.live_actions[1] == {}


def test_a_member_too_short_to_judge_is_left_out_of_the_pooled_scan(tmp_path):
    """Two members (an even count): one long with an onset, one stopped after 25 epochs (e.g. by another's
    STOP_ROUND). The short one used to make the median "never" and the suggestion vanish."""
    logs = {1: [member_log(tmp_path, 1, decay_then_noise())], 2: [member_log(tmp_path, 2, decay_then_noise(25))]}
    exits = {j: ExitInfo("walltime", -2, str(logs[j][0].parent), 10, 10) for j in (1, 2)}
    analysis = decide(logs, exits).analysis
    assert [m.segment_too_short for m in analysis.members] == [False, True]
    assert analysis.suggested is not None and analysis.suggested.n_never == 0
    alone = decide({1: logs[1]}, {1: exits[1]}).analysis.suggested
    assert (analysis.suggested.patience, analysis.suggested.cooldown) == (alone.patience, alone.cooldown)


def test_a_crash_without_exit_info_after_one_logged_epoch_has_a_checkpoint(tmp_path):
    """deeph-train saves a latest at a run's first epoch and logs the epoch after the save: one logged epoch
    of the crashed run means a checkpoint to resume from."""
    log, first, history = retuned_round_1(tmp_path)
    retuned = first.actions[1].fold_overrides["process"]["train"]
    second = member_log(tmp_path, 1, still_improving(2), round_=2,
                        lr0=str(retuned["optimizer"]["init_learning_rate"]),
                        patience=retuned["scheduler"]["patience"], cooldown=retuned["scheduler"]["cooldown"],
                        continued_from=f"/x/{RUN}", continuation=NEW_OPTIMIZER)
    action = decide({1: [log, second]}, {1: None}, SupervisorConfig(), history).actions[1]
    assert action.continue_from == {"previous_output_dir": str(second.parent), "load_model_type": "latest"}
