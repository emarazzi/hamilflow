"""
Training log analysis, on synthetic logs written in the exact line format of real DeepH-pack 1.0.6 logs.

The expected values are worked out by hand from the module docstring's description of the scheduler.
"""

import numpy as np
import pytest

from hamilflow.training.log_analysis import (
    analyze_training_logs,
    format_report,
    lr_stages,
    read_training_curve,
    replay_reduce_on_plateau,
)


def write_log(
    path,
    train,
    val=None,
    scale=None,
    patience=120,
    cooldown=50,
    rtol=0.05,
    factor=0.5,
    accumulation=40,
    n_train=40,
    batch_size=1,
):
    """A log with the header and epoch lines of a real run (paths and machine names left out)."""
    train = np.asarray(train, dtype=float)
    val = train if val is None else np.asarray(val, dtype=float)
    scale = np.ones(len(train)) if scale is None else np.asarray(scale, dtype=float)
    lines = [
        f"[ 10.22-10:30:47 ] [dataloader] Train size: `{n_train}`. Val size: `10`. Test size: `0`.",
        (
            f"[ 10.22-10:30:47 ] [dataloader] Data sharding way: `2`. Batch size: `{batch_size}`. "
            "Number of nodes each batch: `[944]`, `[944]`, `[]`."
        ),
        (
            "[ 10.22-10:31:49 ] [optimizer] Using the optimizer `AdamW` with: betas `[0.9, 0.999]`, eps `1e-08`, "
            "weight decay strength `0.001`, and initial learning rate `0.001`."
        ),
        (
            f"[ 10.22-10:31:49 ] [optimizer] Using the scheduler `ReduceOnPlateau` with: factor `{factor}`, "
            f"patience `{patience}`, rtol `{rtol}`, cooldown `{cooldown}`, and accumulation size `{accumulation}`."
        ),
    ]
    lines += [
        f"[ 10.22-10:37:39 ] [train] Epoch {i + 1} | Time 200.00 s | Train-Loss {t:.6e} | Val-Loss {v:.6e} | Scale {s}"
        for i, (t, v, s) in enumerate(zip(train, val, scale, strict=True))
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def decay_then_noise(n_epochs=400, flat_from=200, noise=0.02, seed=0):
    """log10(loss) falls by 0.01 per epoch until `flat_from`, then only fluctuates."""
    rng = np.random.default_rng(seed)
    log_loss = -2.0 - 0.01 * np.minimum(np.arange(n_epochs), flat_from) + noise * rng.standard_normal(n_epochs)
    return 10**log_loss


# ---------------------------------- reading ----------------------------------


def test_real_format_lines_are_parsed(tmp_path):
    path = write_log(tmp_path / "deepx.log", [6.5e-2, 3.5e-2], val=[4.9e-2, 2.3e-2], scale=[1.0, 0.5])
    curve = read_training_curve(path)
    np.testing.assert_array_equal(curve.epoch, [1, 2])
    np.testing.assert_allclose(curve.train, [6.5e-2, 3.5e-2])
    np.testing.assert_allclose(curve.val, [4.9e-2, 2.3e-2])
    np.testing.assert_allclose(curve.scale, [1.0, 0.5])
    np.testing.assert_allclose(curve.time, [200.0, 200.0])
    assert (curve.n_train, curve.batch_size, curve.lr0) == (40, 1, 0.001)
    assert (curve.factor, curve.patience, curve.rtol, curve.cooldown) == (0.5, 120, 0.05, 50)
    assert curve.accumulation_size == 40


def test_steps_per_epoch_and_epochs_per_check_follow_the_header(tmp_path):
    """From a real run: 160 structures, batch 1, accumulation 384 -> 2.4 epochs per check."""
    curve = read_training_curve(write_log(tmp_path / "a.log", [1.0], accumulation=384, n_train=160))
    assert curve.steps_per_epoch == 160
    assert curve.epochs_per_check == pytest.approx(2.4)
    curve = read_training_curve(write_log(tmp_path / "b.log", [1.0], accumulation=160, n_train=160, batch_size=2))
    assert curve.steps_per_epoch == 80
    assert curve.epochs_per_check == pytest.approx(2.0)


def test_log_without_epochs_is_rejected(tmp_path):
    path = tmp_path / "deepx.log"
    path.write_text("[ 10.22-10:30:36 ] Welcome to DeepH-pack (deepx)!\n")
    with pytest.raises(ValueError, match="No epoch lines"):
        read_training_curve(path)


def test_log_of_another_scheduler_is_rejected(tmp_path):
    path = tmp_path / "deepx.log"
    path.write_text("[ 10.22 ] [train] Epoch 1 | Time 1.0 s | Train-Loss 1e-1 | Val-Loss 1e-1 | Scale 1.0\n")
    with pytest.raises(ValueError, match="reduce_on_plateau"):
        read_training_curve(path)


def test_consecutive_jobs_are_joined_and_renumbered_with_the_last_settings(tmp_path):
    first = write_log(tmp_path / "1.log", [3.0, 2.0], patience=120)
    second = write_log(tmp_path / "2.log", [1.5, 1.0, 0.5], patience=20)
    curve = read_training_curve([first, second])
    np.testing.assert_array_equal(curve.epoch, [1, 2, 3, 4, 5])
    np.testing.assert_allclose(curve.train, [3.0, 2.0, 1.5, 1.0, 0.5])
    np.testing.assert_array_equal(curve.joins, [2])
    assert curve.patience == 20


# ---------------------------------- scheduler ----------------------------------


def test_replay_reduces_after_patience_checks_without_improvement():
    # check 0 sets the best, checks 1, 2, 3 do not improve: decided on check 3, in effect from check 4
    scale = replay_reduce_on_plateau(np.ones(6), patience=3, cooldown=0, rtol=0.05)
    np.testing.assert_allclose(scale, [1, 1, 1, 1, 0.5, 0.5])


def test_replay_cooldown_delays_the_next_reduction():
    # reduced on check 3; checks 4, 5 are the cooldown; then 6, 7, 8 count: in effect from check 9
    scale = replay_reduce_on_plateau(np.ones(10), patience=3, cooldown=2, rtol=0.05)
    np.testing.assert_allclose(scale, [1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25])


def test_replay_period_is_patience_plus_cooldown():
    scale = replay_reduce_on_plateau(np.ones(200), patience=40, cooldown=40, rtol=0.05, factor=0.85)
    reductions = np.nonzero(np.diff(scale) < 0)[0] + 1
    np.testing.assert_array_equal(np.diff(reductions), 80)


def test_improvement_must_beat_rtol_to_reset_the_count():
    # each check is 1% better: below rtol = 5%, so none of them counts
    small = replay_reduce_on_plateau(0.99 ** np.arange(5), patience=3, cooldown=0, rtol=0.05)
    np.testing.assert_allclose(small, [1, 1, 1, 1, 0.5])
    # each check is 6% better: all of them count
    large = replay_reduce_on_plateau(0.94 ** np.arange(5), patience=3, cooldown=0, rtol=0.05)
    np.testing.assert_allclose(large, np.ones(5))


def test_small_improvements_accumulate_against_the_unchanged_best():
    # 2.5% per check: the best stays at check 0 until the drop passes 5%, on check 3 (1 - 0.975^3 = 7.3%)
    scale = replay_reduce_on_plateau(0.975 ** np.arange(7), patience=3, cooldown=0, rtol=0.05)
    np.testing.assert_allclose(scale, np.ones(7))


def test_atol_also_has_to_be_beaten():
    scale = replay_reduce_on_plateau(0.9 ** np.arange(5), patience=3, cooldown=0, rtol=0.0, atol=10.0)
    np.testing.assert_allclose(scale, [1, 1, 1, 1, 0.5])


def test_scale_stops_at_min_scale():
    scale = replay_reduce_on_plateau(np.ones(30), patience=1, cooldown=0, rtol=0.05, min_scale=0.2)
    assert scale.min() == pytest.approx(0.2)


# ---------------------------------- stages ----------------------------------


def test_stages_split_at_each_scale_change_with_gains(tmp_path):
    path = write_log(
        tmp_path / "deepx.log", np.ones(6), val=[4.0, 2.0, 1.0, 1.5, 0.5, 0.8], scale=[1, 1, 0.5, 0.5, 0.25, 0.25]
    )
    stages = lr_stages(read_training_curve(path))
    assert [(s.start, s.end, s.n_epochs) for s in stages] == [(1, 2, 2), (3, 4, 2), (5, 6, 2)]
    assert [s.scale for s in stages] == [1.0, 0.5, 0.25]
    assert [s.best_val for s in stages] == [2.0, 1.0, 0.5]
    assert np.isnan(stages[0].gain)
    assert stages[1].gain == pytest.approx(1.0)  # 2.0 / 1.0 - 1
    assert stages[2].gain == pytest.approx(1.0)  # 1.0 / 0.5 - 1
    assert [s.running for s in stages] == [False, False, True]


def test_stage_that_gets_worse_has_no_gain_and_keeps_the_cumulative_best(tmp_path):
    path = write_log(tmp_path / "deepx.log", np.ones(4), val=[1.0, 1.0, 2.0, 2.0], scale=[1, 1, 0.5, 0.5])
    stages = lr_stages(read_training_curve(path))
    assert stages[1].gain == 0.0
    assert stages[1].cumulative_best_val == 1.0


def test_new_job_is_a_new_stage_but_not_a_reduction(tmp_path):
    first = write_log(tmp_path / "1.log", [1.0, 1.0], scale=[1.0, 1.0])
    second = write_log(tmp_path / "2.log", [1.0, 1.0], scale=[1.0, 1.0])
    stages = lr_stages(read_training_curve([first, second]))
    assert [(s.start, s.end, s.restart) for s in stages] == [(1, 2, False), (3, 4, True)]


def test_scale_lowered_exactly_at_a_join_is_not_counted_as_a_reduction(tmp_path):
    first = write_log(tmp_path / "1.log", [1.0, 1.0], scale=[1.0, 1.0])
    second = write_log(tmp_path / "2.log", [1.0, 1.0, 1.0], scale=[0.5, 0.5, 0.25])
    member = analyze_training_logs([[first, second]]).members[0]
    assert member.reductions == [5]


def test_val_over_train_is_the_median_of_the_last_ten_epochs(tmp_path):
    val = np.r_[np.full(10, 100.0), np.full(10, 2.0)]
    stages = lr_stages(read_training_curve(write_log(tmp_path / "deepx.log", np.ones(20), val=val)))
    assert stages[0].val_over_train == pytest.approx(2.0)


# ---------------------------------- convergence ----------------------------------


def stepped_log(path, stage_vals, stage_len=5, train=None):
    val = np.repeat(stage_vals, stage_len)
    scale = np.repeat(0.5 ** np.arange(len(stage_vals)), stage_len)
    return write_log(path, val if train is None else np.repeat(train, stage_len), val=val, scale=scale)


def test_converged_when_the_last_completed_stages_are_flat(tmp_path):
    # gains: stage 1 100%, stage 2 0.2%, stage 3 0.1%; stage 4 is still running and does not count
    path = stepped_log(tmp_path / "deepx.log", [1.0, 0.5, 0.499, 0.4985, 0.4984])
    member = analyze_training_logs([path]).members[0]
    assert member.converged
    assert member.converged_since == 20  # end of stage 3
    assert member.converged_reason == "flat stages"


def test_not_converged_while_a_completed_stage_still_gains(tmp_path):
    path = stepped_log(tmp_path / "deepx.log", [1.0, 0.5, 0.25, 0.125, 0.06])
    member = analyze_training_logs([path]).members[0]
    assert not member.converged
    assert member.converged_reason is None


def test_single_flat_stage_is_not_enough_with_two_required(tmp_path):
    path = stepped_log(tmp_path / "deepx.log", [1.0, 0.5, 0.499, 0.25, 0.1])
    assert not analyze_training_logs([path], conv_stages=2).members[0].converged
    assert analyze_training_logs([path], conv_stages=1).members[0].converged


def test_converged_when_the_scale_is_down_to_done_scale(tmp_path):
    val = 0.5 ** np.arange(6)
    path = write_log(tmp_path / "deepx.log", val, scale=[1, 1, 1, 1e-2, 1e-3, 1e-3])
    member = analyze_training_logs([path], done_scale=1e-3).members[0]
    assert member.converged_since == 5
    assert member.converged_reason == "scale <= 0.001"


def test_running_stage_does_not_count_towards_convergence(tmp_path):
    # the completed reduced stage gained 100%; only the running one is flat
    path = stepped_log(tmp_path / "deepx.log", [1.0, 0.5, 0.25, 0.2499])
    assert not analyze_training_logs([path], conv_stages=1).members[0].converged


def test_overfitting_is_flagged_when_val_over_train_grows(tmp_path):
    path = stepped_log(tmp_path / "over.log", [1.0, 1.0], train=[1.0, 0.5])
    assert analyze_training_logs([path]).members[0].overfitting
    path = stepped_log(tmp_path / "fine.log", [1.0, 1.0], train=[1.0, 0.9])
    assert not analyze_training_logs([path]).members[0].overfitting


# ---------------------------------- analysis ----------------------------------


def test_unknown_monitor_and_empty_logs_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="monitor must be"):
        analyze_training_logs([write_log(tmp_path / "deepx.log", [1.0])], monitor="best")
    with pytest.raises(ValueError, match="No logs"):
        analyze_training_logs([])


def test_labels_are_the_path_component_that_tells_members_apart(tmp_path):
    logs = [write_log(tmp_path / f"train_{j}" / "deepx.log", [1.0, 0.5]) for j in (1, 2)]
    assert [m.label for m in analyze_training_logs(logs).members] == ["train_1", "train_2"]


def test_multi_job_member_label_counts_its_extra_logs(tmp_path):
    logs = [
        [write_log(tmp_path / f"train_{j}" / f"outputs_{n}" / "deepx.log", [1.0, 0.5]) for n in (1, 2)] for j in (1, 2)
    ]
    assert [m.label for m in analyze_training_logs(logs).members] == ["train_1 (+1)", "train_2 (+1)"]


def test_onset_is_found_where_progress_stops(tmp_path):
    path = write_log(tmp_path / "deepx.log", decay_then_noise())
    member = analyze_training_logs([path]).members[0]
    # a window of 50 epochs drops by 0.5 decades on the slope: progress is under 2 x noise close to its end
    assert 200 <= member.onset <= 260
    assert 0.01 < member.noise < 0.03


def test_larger_k_declares_the_onset_earlier(tmp_path):
    path = write_log(tmp_path / "deepx.log", decay_then_noise())
    strict = analyze_training_logs([path], k=1.0).members[0].onset
    lenient = analyze_training_logs([path], k=8.0).members[0].onset
    assert lenient < strict


def test_no_onset_while_the_loss_is_still_clearly_falling(tmp_path):
    path = write_log(tmp_path / "deepx.log", decay_then_noise(n_epochs=150, flat_from=1000, noise=0.002))
    analysis = analyze_training_logs([path])
    assert analysis.members[0].onset is None
    assert analysis.onset is None
    assert analysis.suggested is None


def test_short_segments_give_no_suggestion(tmp_path):
    path = write_log(tmp_path / "deepx.log", decay_then_noise(n_epochs=60, flat_from=5))
    analysis = analyze_training_logs([path])
    assert analysis.segments_too_short
    assert analysis.suggestions == []


def test_suggested_patience_is_the_smallest_reducing_at_or_after_the_onset(tmp_path):
    path = write_log(tmp_path / "deepx.log", decay_then_noise())
    analysis = analyze_training_logs([path])
    suggestion = analysis.suggested
    assert suggestion.rtol == 0.05
    assert suggestion.first_reduction >= analysis.onset
    assert suggestion.patience in analysis.patience_grid
    index = analysis.patience_grid.index(suggestion.patience)
    if index > 0:
        assert analysis.replayed_first_reductions[0.05][analysis.patience_grid[index - 1]] < analysis.onset


def test_cooldown_without_reductions_is_half_the_patience(tmp_path):
    suggestion = analyze_training_logs([write_log(tmp_path / "deepx.log", decay_then_noise())]).suggested
    assert not suggestion.cooldown_measured
    assert suggestion.cooldown == suggestion.patience // 2


def test_suggestion_is_converted_to_scheduler_checks(tmp_path):
    """patience in train.toml counts checks: with 2 epochs per check it is half the patience in epochs."""
    log = write_log(tmp_path / "deepx.log", decay_then_noise(), accumulation=80, n_train=40)
    analysis = analyze_training_logs([log])
    suggestion = analysis.suggested
    patience_in_epochs = 2 * suggestion.patience
    assert analysis.replayed_first_reductions[0.05][patience_in_epochs] == suggestion.first_reduction
    assert suggestion.cooldown == round((patience_in_epochs // 2) / 2)


def test_minimum_length_counts_the_remaining_reductions(tmp_path):
    analysis = analyze_training_logs([write_log(tmp_path / "deepx.log", decay_then_noise())], min_scale=1e-4)
    assert analysis.n_reductions_to_end == 14  # 0.5**14 = 6.1e-5 is the first below 1e-4
    best = analysis.suggested
    assert best.min_total_epochs == pytest.approx(best.first_reduction + 13 * (best.patience + best.cooldown))


def test_accumulation_that_is_not_whole_epochs_warns(tmp_path):
    path = write_log(tmp_path / "deepx.log", decay_then_noise(), accumulation=384, n_train=160)
    with pytest.warns(UserWarning, match="approximating with one check per 2 epoch"):
        member = analyze_training_logs([path]).members[0]
    assert member.epochs_per_replay_check == 2


def test_check_loss_averages_the_epochs_of_each_check(tmp_path):
    path = write_log(tmp_path / "deepx.log", [1.0, 3.0, 5.0, 7.0, 9.0], accumulation=80, n_train=40)
    member = analyze_training_logs([path], monitor="train").members[0]
    np.testing.assert_allclose(member.check_loss, [2.0, 6.0])
    np.testing.assert_array_equal(member.check_epoch, [2, 4])


def test_monitor_defaults_to_the_validation_loss(tmp_path):
    """The loss that reproduced all 10 reductions of a real log, where train and mean diverged from the 8th on."""
    path = write_log(tmp_path / "deepx.log", np.ones(4), val=3 * np.ones(4))
    analysis = analyze_training_logs([path])
    assert analysis.monitor == "val"
    np.testing.assert_allclose(analysis.members[0].check_loss, 3.0)


@pytest.mark.parametrize(("monitor", "expected"), [("train", 1.0), ("val", 3.0), ("mean", 2.0)])
def test_monitor_chooses_the_loss(tmp_path, monitor, expected):
    path = write_log(tmp_path / "deepx.log", np.ones(4), val=3 * np.ones(4))
    np.testing.assert_allclose(analyze_training_logs([path], monitor=monitor).members[0].check_loss, expected)


# ---------------------------------- replay against the logged scale ----------------------------------


def test_replay_mismatch_finds_the_monitor_behind_the_log_with_one_epoch_per_check(tmp_path):
    train = np.ones(40)
    val = np.r_[np.linspace(4, 1, 20), np.ones(20)]
    scale = replay_reduce_on_plateau(val, patience=5, cooldown=2, rtol=0.05)
    assert scale.min() < 1
    path = write_log(tmp_path / "deepx.log", train, val=val, scale=scale, patience=5, cooldown=2)
    mismatch = analyze_training_logs([path]).members[0].replay_mismatch["job 1"]
    assert mismatch["val"] is None
    assert mismatch["train"] is not None


def test_replay_mismatch_is_empty_without_reductions(tmp_path):
    assert analyze_training_logs([write_log(tmp_path / "deepx.log", np.ones(10))]).members[0].replay_mismatch == {}


@pytest.mark.xfail(
    strict=True,
    reason="_replay_mismatch compares with the first epoch of each check, but the logged scale changes on its "
    "last (real log: 160 structures, batch 2, 2 epochs per check; val reproduces all 10 reductions)",
)
def test_replay_mismatch_with_several_epochs_per_check_matches_the_real_log_behaviour(tmp_path):
    """In real logs the scale changes on the last epoch of a check: a replay reduced from check i shows at its end."""
    val = np.r_[np.linspace(4, 1, 20), np.ones(60)]
    checks = val.reshape(-1, 2).mean(axis=1)
    replayed = replay_reduce_on_plateau(checks, patience=5, cooldown=2, rtol=0.05)
    assert replayed.min() < 1
    scale = np.repeat(replayed, 2)
    scale[2::2] = scale[1:-1:2]  # the first epoch of a check still shows the scale of the previous one
    path = write_log(
        tmp_path / "deepx.log", np.ones(80), val=val, scale=scale, patience=5, cooldown=2, accumulation=80, n_train=40
    )
    assert analyze_training_logs([path]).members[0].replay_mismatch["job 1"]["val"] is None


# ---------------------------------- report ----------------------------------


def test_report_of_a_run_without_reductions(tmp_path):
    report = format_report(analyze_training_logs([write_log(tmp_path / "deepx.log", decay_then_noise())]))
    assert "1 members | train 40 structures, batch 1 -> 40 steps/epoch" in report
    assert "not converged (no LR reduction yet)" in report
    assert "=> suggested (rtol 0.05)" in report
    assert "still training" in report


def test_report_of_converged_members_says_training_can_stop(tmp_path):
    path = stepped_log(tmp_path / "deepx.log", [1.0, 0.5, 0.499, 0.4985, 0.4984])
    report = format_report(analyze_training_logs([path]))
    assert "converged since epoch 20 (flat stages" in report
    assert "all members converged: training can be stopped" in report


def test_report_explains_short_segments(tmp_path):
    report = format_report(analyze_training_logs([write_log(tmp_path / "deepx.log", decay_then_noise(60, 5))]))
    assert "patience scan not meaningful" in report


def test_report_expresses_the_length_in_jobs_when_walltime_is_given(tmp_path):
    path = write_log(tmp_path / "deepx.log", decay_then_noise())
    assert "jobs of 24 h" in format_report(analyze_training_logs([path], walltime_h=24.0))
