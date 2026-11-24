"""
Training log analysis, on synthetic logs written in the exact line format of real DeepH-pack 1.0.6 logs.

The expected values are worked out by hand from the module docstring's description of the scheduler.
"""

import numpy as np
import pytest
from synthetic_deeph_logs import KEPT_OPTIMIZER, NEW_OPTIMIZER, decay_then_noise, write_log

from hamilflow.training.log_analysis import (
    analyze_training_logs,
    format_report,
    lr_stages,
    read_training_curve,
    replay_reduce_on_plateau,
)

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


# ---------------------------------- chained jobs ----------------------------------


RUN_1 = "2026-10-22_10-30-28"  # the run dir name of job 1 in the chained fixtures


def test_continuation_line_is_parsed(tmp_path):
    scratch = write_log(tmp_path / RUN_1 / "deepx.log", [1.0, 0.5])  # best val on the last epoch: nothing cut
    continued = write_log(tmp_path / "2.log", [1.0], continued_from=f"./outputs_r1/{RUN_1}")
    first, second = read_training_curve([scratch, continued]).jobs
    assert (first.continued_from, first.load_model_type, first.new_optimizer, first.new_training_data) == (
        None, None, None, None)
    assert second.continued_from == f"./outputs_r1/{RUN_1}"
    assert second.n_abandoned == 0
    assert (second.load_model_type, second.new_optimizer, second.new_training_data) == ("best", True, False)
    assert (first.start, first.n_epochs, second.start, second.n_epochs) == (0, 2, 2, 1)
    assert second.settings["patience"] == 120


@pytest.mark.parametrize(
    ("text", "new_optimizer", "new_data"),
    [
        # the real lines (DeepH-pack 1.0.6, cluster check 2026-11-24)
        (NEW_OPTIMIZER, True, False),
        (KEPT_OPTIMIZER, False, False),
        # not seen in a real log: guessed wordings for new training data and other variants
        ("We are using the new training dataset, and applying new optimizer.", True, True),
        ("We are using the origin training dataset, and using the origin optimizer.", False, False),
        ("We are using the origin training dataset, and not applying new optimizer.", False, False),
        ("", False, None),
    ],
)
def test_continuation_variants_are_read_loosely(tmp_path, text, new_optimizer, new_data):
    path = write_log(tmp_path / "2.log", [1.0], continued_from="/run", load_model_type="latest", continuation=text)
    job = read_training_curve(path).jobs[0]
    assert (job.load_model_type, job.new_optimizer, job.new_training_data) == ("latest", new_optimizer, new_data)


def chained_with_new_optimizer(tmp_path, lr0_2="0.00025", **second):
    """Job 1 reduces 1 -> 0.5 -> 0.25 with its best val on its last epoch; job 2 loads it and restarts the
    optimizer at lr0_1 x 0.25, its scale back at 1."""
    val_1 = [8.0, 8.0, 4.0, 4.0, 2.0, 1.9]
    val_2 = [1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.125]
    first = write_log(tmp_path / RUN_1 / "deepx.log", np.ones(6), val=val_1, scale=[1, 1, 0.5, 0.5, 0.25, 0.25])
    second = write_log(tmp_path / "2.log", np.ones(7), val=val_2, scale=[1, 1, 0.5, 0.5, 0.25, 0.25, 0.125],
                       lr0=lr0_2, continued_from=f"./outputs_r1/{RUN_1}", **second)
    return [first, second]


def test_new_optimizer_scale_is_normalized_to_the_first_initial_lr(tmp_path):
    curve = read_training_curve(chained_with_new_optimizer(tmp_path))
    np.testing.assert_allclose(curve.raw_scale, [1, 1, 0.5, 0.5, 0.25, 0.25, 1, 1, 0.5, 0.5, 0.25, 0.25, 0.125])
    np.testing.assert_allclose(curve.scale, [1, 1, 0.5, 0.5, 0.25, 0.25] + [0.25, 0.25, 0.125, 0.125, 1 / 16,
                                                                           1 / 16, 1 / 32])
    assert [job.lr_ratio for job in curve.jobs] == [1.0, 0.25]
    assert curve.lr0 == 0.00025  # the settings are those of the last log
    assert curve.current_learning_rate == pytest.approx(0.00025 * 0.125)
    assert curve.current_learning_rate == pytest.approx(0.001 * curve.scale[-1])


def test_new_optimizer_join_is_neither_a_reduction_nor_a_new_stage(tmp_path):
    member = analyze_training_logs([chained_with_new_optimizer(tmp_path)]).members[0]
    assert member.reductions == [3, 5, 9, 11, 13]  # none at the join, epoch 7
    stages = member.stages
    assert [(s.scale, s.start, s.end, s.jobs, s.restart) for s in stages] == [
        (1.0, 1, 2, 1, False), (0.5, 3, 4, 1, False), (0.25, 5, 8, 2, False), (0.125, 9, 10, 1, False),
        (1 / 16, 11, 12, 1, False), (1 / 32, 13, 13, 1, False)]
    assert stages[2].best_val == 1.0  # the stage spans both jobs
    assert stages[3].gain == pytest.approx(1.0)  # 1.0 / 0.5 - 1, against the stage across the join


def test_done_scale_is_reached_on_the_normalized_scale(tmp_path):
    # the raw scale of job 2 never goes below 0.125; normalized it is 1/16 from epoch 11
    member = analyze_training_logs([chained_with_new_optimizer(tmp_path)], done_scale=1 / 16).members[0]
    assert member.converged_since == 11
    assert member.converged_reason == "scale <= 0.0625"


def test_without_normalization_the_join_would_look_like_a_reset(tmp_path):
    """Same logs with the old lr0 in job 2: its scale is then a reset to the initial LR, a new stage."""
    member = analyze_training_logs([chained_with_new_optimizer(tmp_path, lr0_2="0.001")]).members[0]
    assert [(s.scale, s.start, s.restart) for s in member.stages][2:4] == [(0.25, 5, False), (1.0, 7, True)]
    assert member.reductions == [3, 5, 9, 11, 13]


def test_lr0_printed_with_limited_precision_still_continues_the_stage(tmp_path):
    curve = read_training_curve(chained_with_new_optimizer(tmp_path, lr0_2="2.5001e-04"))
    assert curve.jobs[1].lr_ratio == 0.25  # snapped onto the end of job 1
    assert [(s.start, s.end) for s in lr_stages(curve)][2] == (5, 8)


def test_restart_at_a_lower_lr_is_a_new_stage_not_a_reduction(tmp_path):
    """A restart from scratch scale at half the LR (e.g. after a divergence) starts a new stage."""
    first = write_log(tmp_path / "1.log", np.ones(3), scale=[1, 1, 1])
    second = write_log(tmp_path / "2.log", np.ones(3), scale=[1, 1, 1], lr0="0.0005", continued_from="/run",
                       load_model_type="latest")
    member = analyze_training_logs([[first, second]]).members[0]
    assert [(s.scale, s.start, s.restart) for s in member.stages] == [(1.0, 1, False), (0.5, 4, True)]
    assert member.reductions == []


def test_per_job_replay_uses_the_raw_scale_and_the_settings_of_each_job(tmp_path):
    """Job 1 at patience 5, job 2 (new optimizer at half the LR) at patience 3: each replays on its own."""
    val_1 = np.r_[np.linspace(4, 2, 10), np.full(20, 2.0)]
    scale_1 = replay_reduce_on_plateau(val_1, patience=5, cooldown=2, rtol=0.05)
    val_2 = np.r_[np.linspace(1.9, 1.0, 10), np.full(20, 1.0)]
    scale_2 = replay_reduce_on_plateau(val_2, patience=3, cooldown=1, rtol=0.05)
    assert scale_1.min() < 1 and scale_2.min() < 1
    lr0_2 = 0.001 * float(scale_1[-1])
    first = write_log(tmp_path / "1.log", np.ones(30), val=val_1, scale=scale_1, patience=5, cooldown=2,
                      accumulation=40)
    second = write_log(tmp_path / "2.log", np.ones(30), val=val_2, scale=scale_2, patience=3, cooldown=1,
                       lr0=str(lr0_2), continued_from="/run", load_model_type="latest")
    member = analyze_training_logs([[first, second]]).members[0]
    assert member.replay_mismatch["job 1"]["val"] is None
    assert member.replay_mismatch["job 2"]["val"] is None
    # the join continues the stage of job 1's last scale
    assert not any(s.restart for s in member.stages)


def test_min_scale_floor_of_a_job_is_on_its_raw_scale(tmp_path):
    """min_scale is relative to the first lr0: for job 2 at lr0 x 0.25, the floor of its raw scale is 4 x min_scale."""
    first = write_log(tmp_path / "1.log", np.ones(2), scale=[1.0, 0.25])
    val = np.ones(12)
    scale = replay_reduce_on_plateau(val, patience=1, cooldown=0, rtol=0.05, min_scale=0.4)
    assert scale.min() == pytest.approx(0.4)
    second = write_log(tmp_path / "2.log", np.ones(12), val=val, scale=scale, patience=1, cooldown=0,
                       lr0="0.00025", continued_from="/run", load_model_type="latest")
    member = analyze_training_logs([[first, second]], min_scale=0.1).members[0]
    assert member.replay_mismatch["job 2"]["val"] is None


def drop_lr0(path):
    path.write_text(path.read_text().replace("initial learning rate `0.001`", ""))
    return path


def test_missing_lr0_in_a_chain_warns_and_keeps_the_scale(tmp_path):
    first = write_log(tmp_path / "1.log", np.ones(2), scale=[1.0, 0.5])
    second = drop_lr0(write_log(tmp_path / "2.log", np.ones(2), scale=[0.5, 0.5]))
    with pytest.warns(UserWarning, match="No initial learning rate") as record:
        curve = read_training_curve([first, second])
    assert len(record) == 1 and str(second) in str(record[0].message)
    np.testing.assert_allclose(curve.scale, [1.0, 0.5, 0.5, 0.5])
    # its scale is taken as relative to the first lr0, and so is the current LR
    assert curve.lr0 is None
    assert curve.current_learning_rate == pytest.approx(0.001 * 0.5)


def test_missing_lr0_in_the_first_log_warns_once_naming_it(tmp_path):
    first = drop_lr0(write_log(tmp_path / "1.log", np.ones(2), scale=[1.0, 0.5]))
    second = write_log(tmp_path / "2.log", np.ones(2), scale=[0.5, 0.5])
    third = write_log(tmp_path / "3.log", np.ones(2), scale=[0.5, 0.5])
    with pytest.warns(UserWarning, match="No initial learning rate") as record:
        curve = read_training_curve([first, second, third])
    assert len(record) == 1 and str(first) in str(record[0].message)
    assert all(job.lr_ratio == 1.0 for job in curve.jobs)
    assert curve.current_learning_rate == pytest.approx(0.001 * 0.5)  # from the last log's own lr0


def test_snapped_lr_ratio_is_a_plain_float(tmp_path):
    curve = read_training_curve(chained_with_new_optimizer(tmp_path, lr0_2="2.5001e-04"))
    assert type(curve.jobs[1].lr_ratio) is float


def test_report_lists_the_restarted_jobs_and_normalized_scales(tmp_path):
    report = format_report(analyze_training_logs([chained_with_new_optimizer(tmp_path)]))
    assert "lr0 0.001 (first job), 0.00025 (last job)" in report
    assert "job 2 from epoch 7: new optimizer, loads best, lr0 0.00025 = 0.25 x first lr0" in report
    assert "current scale 3.1e-02 (of each member's first lr0)" in report


def test_report_header_shows_only_the_first_lr0_when_it_does_not_change(tmp_path):
    report = format_report(analyze_training_logs([write_log(tmp_path / "deepx.log", [1.0, 0.5])]))
    assert "lr0 0.001 (first job)" in report
    assert "last job" not in report


def test_report_has_no_job_lines_for_plain_resumes(tmp_path):
    first = write_log(tmp_path / "1.log", np.ones(2), scale=[1.0, 0.5])
    second = write_log(tmp_path / "2.log", np.ones(2), scale=[0.5, 0.5], continued_from="/run",
                       load_model_type="latest", continuation=KEPT_OPTIMIZER, first_epoch=3)
    assert "job 2 from epoch" not in format_report(analyze_training_logs([[first, second]]))


# ---------------------------------- resume from best: the dead branch is cut ----------------------------------


def three_stage_job_1(tmp_path):
    """Job 1: best val 4.0 at epoch 20 (scale 0.5), then 10 more epochs at scale 0.25 that never improve."""
    val = np.r_[np.linspace(10, 5, 10), np.linspace(4.9, 4.0, 10), np.full(10, 4.05)]
    scale = np.r_[np.ones(10), np.full(10, 0.5), np.full(10, 0.25)]
    return write_log(tmp_path / RUN_1 / "deepx.log", np.ones(30), val=val, scale=scale)


# the reviewer's repro: the logs are in a folder that is not named after the run dir, hence the warning
@pytest.mark.filterwarnings("ignore:.*cutting the previous log")
def test_resume_from_best_redo_stage_is_not_a_flat_stage(tmp_path):
    v1 = np.r_[np.linspace(10, 5, 10), np.linspace(4.9, 4.0, 10), np.full(10, 4.05)]
    s1 = np.r_[np.ones(10), np.full(10, .5), np.full(10, .25)]
    v2, s2 = np.r_[np.full(10, 3.99), np.full(5, 3.97)], np.r_[np.full(10, .5), np.full(5, .25)]
    p1 = write_log(tmp_path / "1.log", np.ones(30), val=v1, scale=s1)
    p2 = write_log(tmp_path / "2.log", np.ones(15), val=v2, scale=s2, continued_from="/run1",
                   load_model_type="best", continuation="We are using the origin training dataset.")
    assert not analyze_training_logs([[p1, p2]]).members[0].converged


def test_resume_from_best_with_the_optimizer_kept_follows_the_lineage(tmp_path):
    first = three_stage_job_1(tmp_path)
    val_2, scale_2 = np.r_[np.full(10, 3.99), np.full(5, 3.97)], np.r_[np.full(10, 0.5), np.full(5, 0.25)]
    second = write_log(tmp_path / "2.log", np.ones(15), val=val_2, scale=scale_2, continued_from=f"/x/{RUN_1}",
                       continuation="We are using the origin training dataset.")
    member = analyze_training_logs([[first, second]]).members[0]
    curve = member.curve
    assert [(job.start, job.n_epochs, job.n_abandoned) for job in curve.jobs] == [(0, 20, 0), (20, 15, 10)]
    np.testing.assert_array_equal(curve.epoch, np.arange(1, 36))
    np.testing.assert_array_equal(curve.joins, [20])
    assert curve.val[19] == 4.0 and curve.val[20] == 3.99
    assert [(s.scale, s.start, s.end, s.jobs, s.restart) for s in member.stages] == [
        (1.0, 1, 10, 1, False), (0.5, 11, 30, 2, False), (0.25, 31, 35, 1, False)]
    assert member.reductions == [11, 31]  # the reduction to 0.25 counted once


def test_resume_from_best_with_a_new_optimizer_at_the_carried_lr(tmp_path):
    """Decision 7: the LR carried over is the last one of job 1 (0.25), not the one at its best epoch (0.5)."""
    first = three_stage_job_1(tmp_path)
    second = write_log(tmp_path / "2.log", np.ones(10), val=np.full(10, 3.9), lr0="0.00025",
                       continued_from=f"/x/{RUN_1}")
    member = analyze_training_logs([[first, second]]).members[0]
    assert member.curve.jobs[1].n_abandoned == 10
    assert member.curve.jobs[1].lr_ratio == 0.25
    assert [(s.scale, s.start, s.end, s.restart) for s in member.stages] == [
        (1.0, 1, 10, False), (0.5, 11, 20, False), (0.25, 21, 30, True)]
    assert member.reductions == [11]  # the carried-over LR is not a reduction
    report = format_report(analyze_training_logs([[first, second]]))
    assert "job 2 from epoch 21: new optimizer, loads best, last 10 epochs of job 1 left out (dead branch)" in report


def test_resume_from_latest_cuts_nothing(tmp_path):
    first = three_stage_job_1(tmp_path)
    second = write_log(tmp_path / "2.log", np.ones(5), val=np.full(5, 3.9), scale=np.full(5, 0.25),
                       continued_from=f"/x/{RUN_1}", load_model_type="latest", continuation=KEPT_OPTIMIZER,
                       first_epoch=31)  # loaded latest epoch 30
    curve = read_training_curve([first, second])
    assert [(job.n_epochs, job.n_abandoned) for job in curve.jobs] == [(30, 0), (5, 0)]
    assert len(curve.val) == 35


def test_best_on_the_last_epoch_cuts_nothing(tmp_path):
    first = write_log(tmp_path / RUN_1 / "deepx.log", np.ones(4), val=[4.0, 3.0, 2.0, 1.0])
    second = write_log(tmp_path / "2.log", np.ones(2), continued_from=f"/x/{RUN_1}")
    curve = read_training_curve([first, second])
    assert [(job.n_epochs, job.n_abandoned) for job in curve.jobs] == [(4, 0), (2, 0)]
    report = format_report(analyze_training_logs([[first, second]]))
    assert "left out" not in report


def test_run_dir_name_mismatch_warns_but_still_cuts(tmp_path):
    first = three_stage_job_1(tmp_path)
    second = write_log(tmp_path / "2.log", np.ones(5), val=np.full(5, 3.9), scale=np.full(5, 0.5),
                       continued_from="/x/2026-01-01_00-00-00/")
    with pytest.warns(UserWarning, match=r"'2026-01-01_00-00-00', but the previous log is in '2026-10-22_10-30-28'"):
        curve = read_training_curve([first, second])
    assert curve.jobs[1].n_abandoned == 10


def test_matching_run_dir_name_does_not_warn(tmp_path, recwarn):
    first = three_stage_job_1(tmp_path)
    second = write_log(tmp_path / "2.log", np.ones(5), val=np.full(5, 3.9), scale=np.full(5, 0.5),
                       continued_from=f"/cluster/path/{RUN_1}/")  # only the name is compared
    read_training_curve([first, second])
    assert not [w for w in recwarn if "cutting" in str(w.message)]


def test_each_resume_from_best_cuts_its_own_previous_job(tmp_path):
    first = write_log(tmp_path / "r1" / "deepx.log", np.ones(5), val=[5.0, 4.0, 3.0, 3.5, 3.5])
    second = write_log(tmp_path / "r2" / "deepx.log", np.ones(4), val=[2.9, 2.0, 2.5, 2.5], continued_from="/r1")
    third = write_log(tmp_path / "r3" / "deepx.log", np.ones(3), val=[1.9, 1.8, 1.7], continued_from="/r2")
    curve = read_training_curve([first, second, third])
    assert [(job.start, job.n_epochs, job.n_abandoned) for job in curve.jobs] == [(0, 3, 0), (3, 2, 2), (5, 3, 2)]
    np.testing.assert_array_equal(curve.joins, [3, 5])
    np.testing.assert_allclose(curve.val, [5.0, 4.0, 3.0, 2.9, 2.0, 1.9, 1.8, 1.7])
    np.testing.assert_array_equal(curve.epoch, np.arange(1, 9))


# ---------------------------------- resume that redoes reductions: the dead branch is cut ------------------------


@pytest.mark.filterwarnings("ignore:.*does not say whether it kept the optimizer")
@pytest.mark.filterwarnings("ignore:.*do not continue those of")  # the scale fallback on purpose
def test_resume_from_latest_redo_after_a_late_reduction_is_not_a_flat_stage(tmp_path):
    v1 = np.r_[np.linspace(10, 5, 10), np.linspace(4.9, 4.0, 10), np.linspace(3.95, 3.8, 10), np.full(3, 3.79)]
    s1 = np.r_[np.ones(10), np.full(10, .5), np.full(10, .25), np.full(3, .125)]
    v2, s2 = np.r_[np.full(3, 3.79), np.full(5, 3.78)], np.r_[np.full(3, .25), np.full(5, .125)]
    p1 = write_log(tmp_path / "r1" / "deepx.log", np.ones(33), val=v1, scale=s1)
    p2 = write_log(tmp_path / "r2" / "deepx.log", np.ones(8), val=v2, scale=s2, continued_from="/x/r1",
                   load_model_type="latest", continuation="We are using the origin training dataset.")
    member = analyze_training_logs([[p1, p2]]).members[0]
    assert not member.converged
    assert member.reductions.count(37) == 0 or 31 not in member.reductions


@pytest.mark.filterwarnings("ignore:.*does not say whether it kept the optimizer")
def test_resume_from_latest_redo_cuts_the_trailing_lower_scale_epochs(tmp_path):
    v1 = np.r_[np.linspace(10, 5, 10), np.linspace(4.9, 4.0, 10), np.linspace(3.95, 3.8, 10), np.full(3, 3.79)]
    s1 = np.r_[np.ones(10), np.full(10, .5), np.full(10, .25), np.full(3, .125)]
    v2, s2 = np.r_[np.full(3, 3.79), np.full(5, 3.78)], np.r_[np.full(3, .25), np.full(5, .125)]
    p1 = write_log(tmp_path / "r1" / "deepx.log", np.ones(33), val=v1, scale=s1)
    p2 = write_log(tmp_path / "r2" / "deepx.log", np.ones(8), val=v2, scale=s2, continued_from="/x/r1",
                   load_model_type="latest", continuation=KEPT_OPTIMIZER, first_epoch=31)  # loaded latest epoch 30
    member = analyze_training_logs([[p1, p2]]).members[0]
    assert [(job.n_epochs, job.n_abandoned) for job in member.curve.jobs] == [(30, 0), (8, 3)]
    np.testing.assert_array_equal(member.curve.joins, [30])
    assert member.reductions == [11, 21, 34]
    assert [(s.scale, s.start, s.end, s.jobs) for s in member.stages] == [
        (1.0, 1, 10, 1), (0.5, 11, 20, 1), (0.25, 21, 33, 2), (0.125, 34, 38, 1)]
    report = format_report(analyze_training_logs([[p1, p2]]))
    assert "optimizer kept, loads latest, last 3 epochs of job 1 left out (dead branch)" in report


@pytest.mark.filterwarnings("ignore:.*does not say whether it kept the optimizer")
@pytest.mark.filterwarnings("ignore:.*do not continue those of")  # the scale fallback on purpose
def test_redo_cut_without_a_continuation_line_at_the_same_lr0(tmp_path):
    """No continuation line and the same lr0: taken as a plain resume with the optimizer kept."""
    first = write_log(tmp_path / "1.log", np.ones(4), scale=[1, 1, 0.5, 0.5])
    second = write_log(tmp_path / "2.log", np.ones(2), scale=[1, 1])
    curve = read_training_curve([first, second])
    assert curve.jobs[1].n_abandoned == 2
    np.testing.assert_allclose(curve.scale, [1, 1, 1, 1])


def test_scale_going_down_at_a_kept_optimizer_join_cuts_nothing(tmp_path):
    first = write_log(tmp_path / "r1" / "deepx.log", np.ones(4), scale=[1, 1, 0.5, 0.5])
    second = write_log(tmp_path / "r2" / "deepx.log", np.ones(2), scale=[0.25, 0.25], continued_from="/x/r1",
                       load_model_type="latest", continuation=KEPT_OPTIMIZER, first_epoch=5)
    curve = read_training_curve([first, second])
    assert [(job.n_epochs, job.n_abandoned) for job in curve.jobs] == [(4, 0), (2, 0)]


def test_new_optimizer_at_a_carried_lr_is_not_a_redo(tmp_path):
    """The scale of a new-optimizer job restarts at 1 (up in raw terms, not on the first lr0): nothing is cut."""
    first = write_log(tmp_path / "r1" / "deepx.log", np.ones(4), val=[4.0, 3.0, 2.0, 1.0], scale=[1, 1, 0.5, 0.25])
    second = write_log(tmp_path / "r2" / "deepx.log", np.ones(2), scale=[1, 1], lr0="0.00025",
                       continued_from="/x/r1", load_model_type="latest")
    curve = read_training_curve([first, second])
    assert [(job.n_epochs, job.n_abandoned) for job in curve.jobs] == [(4, 0), (2, 0)]
    np.testing.assert_allclose(curve.scale, [1, 1, 0.5, 0.25, 0.25, 0.25])


def test_new_optimizer_at_a_higher_lr_is_not_a_redo(tmp_path):
    """A new optimizer at a higher LR is a choice of the user, not a redo: a new stage, nothing cut."""
    first = write_log(tmp_path / "r1" / "deepx.log", np.ones(4), val=[4.0, 3.0, 2.0, 1.0], scale=[1, 1, 0.5, 0.25])
    second = write_log(tmp_path / "r2" / "deepx.log", np.ones(2), scale=[1, 1], continued_from="/x/r1",
                       load_model_type="latest")
    curve = read_training_curve([first, second])
    assert curve.jobs[1].n_abandoned == 0
    assert [(s.scale, s.restart) for s in lr_stages(curve)][-1] == (1.0, True)


@pytest.mark.filterwarnings("ignore:.*does not say whether it kept the optimizer")
@pytest.mark.filterwarnings("ignore:.*do not continue those of")  # the scale fallback on purpose
def test_best_cut_runs_before_the_redo_cut(tmp_path):
    """Best val at epoch 3 (scale 0.5); job 2 kept the optimizer but starts at 1.0: the redo cut then also
    removes the epochs at 0.5 before it (the scheduler state of the checkpoint is older still)."""
    first = write_log(tmp_path / "r1" / "deepx.log", np.ones(5), val=[4.0, 3.0, 2.0, 2.5, 2.5],
                      scale=[1, 1, 0.5, 0.5, 0.25])
    second = write_log(tmp_path / "r2" / "deepx.log", np.ones(2), scale=[1, 1], continued_from="/x/r1",
                       continuation="We are using the origin training dataset.")
    curve = read_training_curve([first, second])
    assert [(job.n_epochs, job.n_abandoned) for job in curve.jobs] == [(2, 0), (2, 3)]


# ---------------------------------- diverged epochs: nan / inf losses ----------------------------------


@pytest.mark.parametrize("bad", ["nan", "NaN", "inf", "-inf", "+Infinity"])
def test_nan_and_inf_losses_are_parsed(tmp_path, bad):
    path = write_log(tmp_path / "deepx.log", [3.0, 2.0, 1.0], val=[3.0, 2.0, 1.0])
    path.write_text(path.read_text().replace("Val-Loss 2.000000e+00", f"Val-Loss {bad}"))
    curve = read_training_curve(path)
    np.testing.assert_array_equal(curve.epoch, [1, 2, 3])
    assert curve.val[1] == float(bad) or (np.isnan(curve.val[1]) and np.isnan(float(bad)))


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_analysis_and_report_run_with_a_nan_epoch(tmp_path):
    train = decay_then_noise()
    val = train.copy()
    val[1] = np.nan
    first = write_log(tmp_path / "r1" / "deepx.log", train, val=val)
    assert "Val-Loss nan" in first.read_text()
    curve = read_training_curve(first)
    np.testing.assert_array_equal(curve.epoch, np.arange(1, 401))
    assert np.isnan(curve.val[1])
    analysis = analyze_training_logs([first])
    stage = analysis.members[0].stages[0]
    assert stage.best_val == pytest.approx(np.nanmin(val))
    assert np.isfinite(stage.cumulative_best_val)
    assert "1 members" in format_report(analysis)


def test_best_cut_ignores_nan_epochs(tmp_path):
    first = write_log(tmp_path / "r1" / "deepx.log", np.ones(4), val=[3.0, 1.0, np.nan, 2.0])
    second = write_log(tmp_path / "r2" / "deepx.log", np.ones(2), continued_from="/x/r1")
    curve = read_training_curve([first, second])
    assert [(job.n_epochs, job.n_abandoned) for job in curve.jobs] == [(2, 0), (2, 2)]


def test_all_nan_stage_has_nan_best_and_keeps_the_cumulative_best(tmp_path):
    path = write_log(tmp_path / "deepx.log", np.ones(4), val=[2.0, 1.0, np.nan, np.nan], scale=[1, 1, 0.5, 0.5])
    stages = lr_stages(read_training_curve(path))
    assert np.isnan(stages[1].best_val)
    assert stages[1].cumulative_best_val == 1.0
    assert np.isnan(stages[1].gain) or stages[1].gain == 0.0


@pytest.mark.filterwarnings("ignore:.*do not continue those of")  # the scale fallback on purpose
def test_redo_cut_on_an_inferred_kept_optimizer_warns(tmp_path):
    """No continuation line, same lr0, scale back at 1.0 after 1 -> 0.5 -> 0.25: cut, with a warning."""
    first = write_log(tmp_path / "1.log", np.ones(6), scale=[1, 1, 0.5, 0.5, 0.25, 0.25])
    second = write_log(tmp_path / "2.log", np.ones(2), scale=[1, 1])
    with pytest.warns(UserWarning, match="does not say whether it kept the optimizer"):
        curve = read_training_curve([first, second])
    assert curve.jobs[1].n_abandoned == 4


@pytest.mark.filterwarnings("ignore:.*do not continue those of")  # the scale fallback on purpose
def test_redo_cut_on_a_continuation_line_silent_about_the_optimizer_warns(tmp_path):
    first = write_log(tmp_path / "r1" / "deepx.log", np.ones(4), scale=[1, 1, 0.5, 0.5])
    second = write_log(tmp_path / "r2" / "deepx.log", np.ones(2), scale=[1, 1], continued_from="/x/r1",
                       load_model_type="latest", continuation="We are using the origin training dataset.")
    with pytest.warns(UserWarning, match="does not say whether it kept the optimizer"):
        assert read_training_curve([first, second]).jobs[1].n_abandoned == 2


def test_redo_cut_with_the_optimizer_stated_kept_does_not_warn(tmp_path, recwarn):
    first = write_log(tmp_path / "r1" / "deepx.log", np.ones(4), scale=[1, 1, 0.5, 0.5])
    second = write_log(tmp_path / "r2" / "deepx.log", np.ones(2), scale=[1, 1], continued_from="/x/r1",
                       load_model_type="latest", continuation=KEPT_OPTIMIZER, first_epoch=3)  # latest epoch 2
    assert read_training_curve([first, second]).jobs[1].n_abandoned == 2
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


# ---------------------------------- nan scale and nan checks ----------------------------------


def test_nan_scale_takes_the_last_finite_one(tmp_path):
    path = write_log(tmp_path / "deepx.log", np.ones(5), scale=[np.nan, 1.0, np.inf, 0.5, np.nan])
    np.testing.assert_allclose(read_training_curve(path).scale, [1.0, 1.0, 1.0, 0.5, 0.5])


def test_nan_scale_inside_a_stage_does_not_split_it_nor_converge(tmp_path):
    """The reviewer's case: a nan Scale at epochs 221-222 inside a 0.5 stage gave a false "flat stages"."""
    val = 10 ** (-0.01 * np.arange(300))
    scale = np.r_[np.ones(200), np.full(100, 0.5)]
    scale[220:222] = np.nan
    member = analyze_training_logs([write_log(tmp_path / "deepx.log", val, scale=scale)], conv_stages=1).members[0]
    assert [(s.scale, s.start, s.end) for s in member.stages] == [(1.0, 1, 200), (0.5, 201, 300)]
    assert member.reductions == [201]
    assert not member.converged


def test_a_nan_epoch_in_the_initial_segment_keeps_the_onset_and_noise(tmp_path):
    clean = analyze_training_logs([write_log(tmp_path / "a.log", decay_then_noise())]).members[0]
    val = decay_then_noise()
    val[150] = np.nan
    member = analyze_training_logs([write_log(tmp_path / "b.log", decay_then_noise(), val=val)]).members[0]
    assert member.onset is not None and abs(member.onset - clean.onset) <= 5
    assert np.isfinite(member.noise) and member.noise == pytest.approx(clean.noise, rel=0.2)
    assert np.isnan(member.check_loss[150])  # the replays keep it


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


def test_new_job_at_the_same_scale_stays_in_the_stage(tmp_path):
    first = write_log(tmp_path / "1.log", [1.0, 1.0, 1.0], val=[4.0, 3.0, 3.0], scale=[1.0, 0.5, 0.5])
    second = write_log(tmp_path / "2.log", [1.0, 1.0, 1.0], val=[2.9, 2.0, 2.0], scale=[0.5, 0.5, 0.25])
    stages = lr_stages(read_training_curve([first, second]))
    assert [(s.start, s.end, s.jobs, s.restart) for s in stages] == [(1, 1, 1, False), (2, 5, 2, False),
                                                                      (6, 6, 1, False)]
    assert stages[1].gain == pytest.approx(1.0)  # 4.0 / 2.0 - 1, over both jobs


def test_stage_starting_with_a_new_job_is_marked_restart(tmp_path):
    first = write_log(tmp_path / "1.log", [1.0, 1.0], scale=[1.0, 1.0])
    second = write_log(tmp_path / "2.log", [1.0, 1.0], scale=[0.5, 0.5])
    stages = lr_stages(read_training_curve([first, second]))
    assert [(s.start, s.end, s.restart, s.jobs) for s in stages] == [(1, 2, False, 1), (3, 4, True, 1)]


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


def test_replay_mismatch_with_several_epochs_per_check_reports_the_first_wrong_epoch(tmp_path):
    val = np.r_[np.linspace(4, 1, 20), np.ones(60)]
    checks = val.reshape(-1, 2).mean(axis=1)
    replayed = replay_reduce_on_plateau(checks, patience=5, cooldown=2, rtol=0.05)
    i_reduced = np.argmax(replayed < 1)
    scale = np.ones(80)  # the log never reduces, so the replay is first wrong at the end of check i_reduced
    path = write_log(
        tmp_path / "deepx.log", np.ones(80), val=val, scale=scale, patience=5, cooldown=2, accumulation=80, n_train=40
    )
    # a job without reductions is not compared: lower the very last epoch to make this one count
    scale[-1] = 0.5
    write_log(path, np.ones(80), val=val, scale=scale, patience=5, cooldown=2, accumulation=80, n_train=40)
    assert analyze_training_logs([path]).members[0].replay_mismatch["job 1"]["val"] == 2 * (i_reduced + 1)


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


# ---------------------------------- crash repeats: whole logs as dead branches ----------------------------------


def crash_repeat_chain(tmp_path, load_model_type, continuation, first_epoch=1):
    """r1 (best val at epoch 6 of 8, scale 1 -> 0.5 at epoch 5), r2 crashed after 5 epochs (continues r1),
    r3 started again as r2 was: it continues r1 too."""
    r1 = write_log(tmp_path / "r1" / "deepx.log", np.ones(8), val=[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 3.5, 3.5],
                   scale=[1, 1, 1, 1, 0.5, 0.5, 0.25, 0.25])
    r2 = write_log(tmp_path / "r2" / "deepx.log", np.ones(5), val=np.full(5, 3.2), scale=np.full(5, 0.5),
                   continued_from="/x/r1", load_model_type=load_model_type, continuation=continuation,
                   first_epoch=first_epoch)
    r3 = write_log(tmp_path / "r3" / "deepx.log", np.ones(4), val=np.full(4, 2.9), scale=np.full(4, 0.5),
                   continued_from="/x/r1", load_model_type=load_model_type, continuation=continuation,
                   first_epoch=first_epoch)
    return r1, r2, r3


def test_crash_repeat_from_best_drops_the_crashed_log_whole(tmp_path, recwarn):
    paths = crash_repeat_chain(tmp_path, "best", "We are using the origin training dataset.")
    curve = read_training_curve(paths)
    assert [(job.number, job.start, job.n_epochs, job.n_abandoned, job.dropped) for job in curve.jobs] == [
        (1, 0, 6, 0, []), (3, 6, 4, 7, [(1, 5)])]  # r1 cut after its best (2 epochs) + r2 (5 epochs)
    np.testing.assert_array_equal(curve.joins, [6])
    np.testing.assert_allclose(curve.val, [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.9, 2.9, 2.9, 2.9])
    np.testing.assert_array_equal(curve.epoch, np.arange(1, 11))
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]
    report = format_report(analyze_training_logs([paths]))
    assert "job 3 from epoch 7: optimizer kept, loads best, job 2 (5 epochs) left out: dead branch, " \
           "last 2 epochs of job 1 left out (dead branch)" in report


def test_crash_repeat_from_latest_drops_the_crashed_log_and_applies_the_redo_rule(tmp_path, recwarn):
    paths = crash_repeat_chain(tmp_path, "latest", KEPT_OPTIMIZER, first_epoch=7)  # both loaded latest epoch 6
    curve = read_training_curve(paths)
    # r3 restarts at 0.5: r1's trailing epochs at 0.25 are redone, r2 is a dead branch
    assert [(job.number, job.n_epochs, job.n_abandoned, job.dropped) for job in curve.jobs] == [
        (1, 6, 0, []), (3, 4, 7, [(1, 5)])]
    np.testing.assert_allclose(curve.scale, [1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]
    member = analyze_training_logs([paths]).members[0]
    assert member.reductions == [5]
    assert list(member.replay_mismatch) in ([], ["job 1"])  # numbered as given, not as kept


def test_a_normal_three_job_chain_drops_no_log(tmp_path):
    first = write_log(tmp_path / "r1" / "deepx.log", np.ones(5), val=[5.0, 4.0, 3.0, 3.5, 3.5])
    second = write_log(tmp_path / "r2" / "deepx.log", np.ones(4), val=[2.9, 2.0, 2.5, 2.5], continued_from="/r1")
    third = write_log(tmp_path / "r3" / "deepx.log", np.ones(3), val=[1.9, 1.8, 1.7], continued_from="/r2")
    curve = read_training_curve([first, second, third])
    assert [(job.number, job.n_abandoned, job.dropped) for job in curve.jobs] == [(1, 0, []), (2, 2, []), (3, 2, [])]
    np.testing.assert_array_equal(curve.joins, [3, 5])


def test_repeated_crash_repeats_drop_every_dead_branch(tmp_path, recwarn):
    """r2 loads best of r1 and crashes after 5 epochs; r3 starts again as r2 and crashes after 4; r4 starts
    again (3 epochs): r4 drops r3 and, through it, r2, and r1 is cut after its best (2 epochs)."""
    r1 = write_log(tmp_path / "r1" / "deepx.log", np.ones(5), val=[5.0, 4.0, 3.0, 3.5, 3.5])
    later = [write_log(tmp_path / f"r{n}" / "deepx.log", np.ones(k), val=np.full(k, 2.0), continued_from="/x/r1",
                       continuation=NEW_OPTIMIZER) for n, k in ((2, 5), (3, 4), (4, 3))]
    curve = read_training_curve([r1, *later])
    assert [(job.number, job.n_epochs, job.n_abandoned, job.dropped) for job in curve.jobs] == [
        (1, 3, 0, []), (4, 3, 11, [(1, 5), (2, 4)])]
    np.testing.assert_array_equal(curve.joins, [3])
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]
    report = format_report(analyze_training_logs([[r1, *later]]))
    assert "job 2 (5 epochs) left out: dead branch, job 3 (4 epochs) left out: dead branch, last 2 epochs of job 1" \
        in report


# ---------------------------------- real DeepH lines and continued epoch numbers (cluster check 2026-11-24) ----------


REAL_KEPT = ("[ 11.24-10:24:41 ] [model] We are loading the `latest` model inside `/x/outputs_a/2026-11-24_10-19-48`. "
             "We are using the origin training dataset, and NOT applying new optimizer.")
REAL_NEW = ("[ 11.24-10:30:02 ] [model] We are loading the `best` model inside `/x/outputs_a/2026-11-24_10-19-48`. "
            "We are using the origin training dataset, and applying new optimizer.")
REAL_SAVE = ("[ 11.24-10:19:52 ] [model] We will save the model into `/x/model`. The best model will be saved. "
             "The latest model (keep `3` each `2` epoch) will be saved.")


@pytest.mark.parametrize(("line", "load", "new_optimizer"), [(REAL_KEPT, "latest", False), (REAL_NEW, "best", True)])
def test_the_real_continuation_lines_are_parsed(tmp_path, line, load, new_optimizer):
    path = write_log(tmp_path / "deepx.log", [1.0, 0.5])
    lines = path.read_text().splitlines()
    path.write_text("\n".join([*lines[:4], REAL_SAVE, line, *lines[5:]]) + "\n")
    job = read_training_curve(path).jobs[0]
    assert (job.continued_from, job.load_model_type) == ("/x/outputs_a/2026-11-24_10-19-48", load)
    assert (job.new_optimizer, job.new_training_data) == (new_optimizer, False)
    assert job.settings["patience"] == 120  # the save line does not disturb the header


def test_kept_optimizer_resume_continues_the_epoch_numbers_run_b(tmp_path, recwarn):
    """Run B of the cluster check: A logs epochs 1-8, B loads latest epoch 8 and logs 9-13: nothing is dead."""
    a = write_log(tmp_path / "a" / "deepx.log", np.ones(8), scale=[1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5])
    b = write_log(tmp_path / "b" / "deepx.log", np.ones(5), scale=np.full(5, 0.5), continued_from="/x/a",
                  load_model_type="latest", continuation=KEPT_OPTIMIZER, first_epoch=9)
    curve = read_training_curve([a, b])
    assert [(job.n_epochs, job.n_abandoned) for job in curve.jobs] == [(8, 0), (5, 0)]
    np.testing.assert_array_equal(curve.raw_epoch, np.arange(1, 14))
    np.testing.assert_array_equal(curve.epoch, np.arange(1, 14))
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_kept_optimizer_resume_from_an_older_latest_drops_exactly_the_dead_epochs_run_d(tmp_path, recwarn):
    """Run D: A logs 1-8, D loads latest epoch 6 (load_model_epoch = 6) and logs 7-9: A's 7-8 are dead."""
    a = write_log(tmp_path / "a" / "deepx.log", np.ones(8), val=np.arange(8, 0, -1.0))
    d = write_log(tmp_path / "d" / "deepx.log", np.ones(3), val=[2.5, 2.4, 2.3], continued_from="/x/a",
                  load_model_type="latest", continuation=KEPT_OPTIMIZER, first_epoch=7)
    curve = read_training_curve([a, d])
    assert [(job.n_epochs, job.n_abandoned) for job in curve.jobs] == [(6, 0), (3, 2)]
    np.testing.assert_array_equal(curve.raw_epoch, np.arange(1, 10))
    np.testing.assert_allclose(curve.val, [8, 7, 6, 5, 4, 3, 2.5, 2.4, 2.3])
    np.testing.assert_array_equal(curve.joins, [6])
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_latest_redo_with_real_continued_numbering_is_not_a_flat_stage(tmp_path, recwarn):
    """The reviewer's latest-redo case as DeepH logs it: latest saved at epoch 30, the run went on to epoch 33
    (a reduction at 31), the resume logs from epoch 31 again."""
    v1 = np.r_[np.linspace(10, 5, 10), np.linspace(4.9, 4.0, 10), np.linspace(3.95, 3.8, 10), np.full(3, 3.79)]
    s1 = np.r_[np.ones(10), np.full(10, .5), np.full(10, .25), np.full(3, .125)]
    v2, s2 = np.r_[np.full(3, 3.79), np.full(5, 3.78)], np.r_[np.full(3, .25), np.full(5, .125)]
    p1 = write_log(tmp_path / "r1" / "deepx.log", np.ones(33), val=v1, scale=s1)
    p2 = write_log(tmp_path / "r2" / "deepx.log", np.ones(8), val=v2, scale=s2, continued_from="/x/r1",
                   load_model_type="latest", continuation=KEPT_OPTIMIZER, first_epoch=31)
    member = analyze_training_logs([[p1, p2]]).members[0]
    assert not member.converged
    assert [(job.n_epochs, job.n_abandoned) for job in member.curve.jobs] == [(30, 0), (8, 3)]
    assert member.reductions == [11, 21, 34]
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_numbering_that_does_not_continue_falls_back_on_the_scale_with_a_warning(tmp_path):
    first = write_log(tmp_path / "r1" / "deepx.log", np.ones(4), scale=[1, 1, 0.5, 0.5])
    second = write_log(tmp_path / "r2" / "deepx.log", np.ones(2), scale=[1, 1], continued_from="/x/r1",
                       load_model_type="latest", continuation=KEPT_OPTIMIZER)  # restarts at Epoch 1
    with pytest.warns(UserWarning, match="do not continue those of"):
        assert read_training_curve([first, second]).jobs[1].n_abandoned == 2
