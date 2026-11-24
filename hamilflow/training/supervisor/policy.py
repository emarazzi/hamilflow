"""Decision logic of the training supervisor: pure functions of the members' logs, no jobflow, no I/O
beyond reading the logs.

- ``live_check``: the rules the watchdog runs on one member while it trains (W1 divergence, W2
  converged / overfitting, W3 mistuned scheduler).
- ``round_decision``: after a round, one ``MemberAction`` per member, with the TOML overrides it
  implies, in the format ``ensemble.write_ensemble_train_tomls`` takes (``continue_from`` and
  ``fold_overrides`` per fold).

Continuation: a plain resume loads ``resume_from`` (``latest``) with the
optimizer kept and the member's current scheduler settings. A retune loads ``best`` with a new
optimizer whose initial LR is the LR in effect at the end of the run
(``TrainingCurve.current_learning_rate``); a divergence restart loads ``best`` too, at the LR of
the last job's best-val epoch x ``lr_drop_on_divergence`` (the scheduler kept reducing on the bad
epochs, so the last LR is not the one of the model loaded). Either way
``min_learning_rate_scale = min_scale / scale_carried``, so that training still ends at the same
absolute LR. A member already at that floor cannot continue: it is done or given up.

``min_scale`` is the user's ``process.train.scheduler.min_learning_rate_scale`` of the shared train
overrides, relative to the first lr0: ``resolve_min_scale``.

``history`` is plain JSON (step 5 keeps it in state.json)::

    {"start": {"<fold>": {"continue_from": ..., "fold_overrides": ...}},   # optional: round 1's start
     "rounds": [{"round": 1, "members": {"<fold>": {"action": "resume", "cause": "walltime", ...}},
                 "shared_overrides": {...}}, ...]}

with one entry per finished round, appended by ``append_history``; ``MemberAction.to_dict`` gives a
member's record. Restart counts come from the ``cause`` and ``action`` of the records; crashes count
only while consecutive.
"""
import copy
import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import numpy as np

from ..ensemble import deep_merge
from ..log_analysis import MemberAnalysis, TrainingAnalysis, analyze_training_logs, has_epochs
from ..write_train_infer_toml import SchedulerConfig

ACTION_POLICIES = ("resume", "retune", "stop_converged", "stop_overfit", "divergence", "finished_not_converged")
MODES = ("auto", "ask", "off")
# agreed with the user (2026-11-23): only the routine actions run on their own
DEFAULT_ACTIONS = {
    "resume": "auto",
    "stop_converged": "auto",
    "retune": "ask",
    "divergence": "ask",
    "stop_overfit": "ask",
    "finished_not_converged": "ask",
}
EXIT_REASONS = ("finished", "walltime", "stopped", "live_rule", "crashed")
_LOAD_MODEL_TYPES = ("best", "latest")


@dataclass
class SupervisorConfig:
    """Policy, thresholds and limits of the supervisor. Reference scale: ~200 s/epoch, ~850 epochs
    in a 2-day job. JSON round trip with ``to_dict`` / ``from_dict``."""

    # policy key -> "auto" | "ask" | "off"; keys left out take DEFAULT_ACTIONS
    actions: dict[str, str] = field(default_factory=dict)

    # passed to analyze_training_logs
    monitor: str = "val"
    conv_tol: float = 0.02
    conv_stages: int = 2
    # a member counts as converged ("scale <= done_scale") once its LR scale, relative to its first lr0,
    # is at or below this, whatever DeepH's own floor (min_learning_rate_scale) is: with done_scale
    # above min_scale the supervisor stops members before DeepH would
    done_scale: float = 1e-3
    overfit: float = 1.3
    # the target min_learning_rate_scale relative to the member's first lr0. None (default): the one of
    # the shared train overrides (resolve_min_scale); set it only to state the same value explicitly
    min_scale: float | None = None

    # watchdog. 300 s is ~1.5 epochs: reacting faster gains nothing, parsing the log is cheap anyway
    poll_s: float = 300.0
    walltime: float | None = None  # [s]; None: from SLURM (SLURM_JOB_END_TIME)
    # covers the stop (SIGINT .. SIGKILL), the checkpoint flush and the return (no SLURM signal reaches the job)
    walltime_margin: float = 900.0
    # W3 needs an onset, which needs window (50) + smoothing (21) epochs at least; 100 epochs ~ 6 h
    min_epochs_before_rules: int = 100
    # W1: val above 10 x the best so far is far beyond the epoch-to-epoch noise (~30% on real logs)
    # and beyond the spike of an optimizer reset (~3x); 5 epochs in a row (~17 min) rules out a blip
    divergence_factor: float = 10.0
    divergence_epochs: int = 5
    lr_drop_on_divergence: float = 0.5  # the LR of a divergence restart: carried-over LR x this
    # W3 "patience too large": still at the initial LR at onset + retune_slack x suggested patience
    retune_slack: float = 1.5
    # retune when the member's patience is off the pooled suggestion by more than this fraction (the
    # suggestion moves in steps of 5 epochs and with the noise: small differences are not worth an
    # optimizer reset), or when the suggested rtol is another one
    retune_threshold: float = 0.5

    # limits. 10 rounds of 2-day jobs are ~8500 epochs, about the default max_epoch (10000)
    max_rounds: int = 10
    max_divergence_restarts: int = 2
    max_crash_restarts: int = 2

    # continuation: plain resumes load latest, saved every latest_interval epochs
    resume_from: str = "latest"
    latest_interval: int = 10

    # how the watchdog runs and stops deeph-train: a template, "{toml}" is replaced by the
    # train.toml path, e.g. ["bash", "-c", "source /deeph-venv/bin/activate && exec deeph-train {toml}"]
    train_command: list[str] = field(default_factory=lambda: ["deeph-train", "{toml}"])
    # SIGINT -> stop_grace -> SIGTERM -> kill_grace -> SIGKILL. 240 s lets an epoch (~200 s) finish;
    # 30 s is SLURM's own KillWait. Both plus a checkpoint flush fit in walltime_margin (900 s)
    stop_grace: float = 240.0
    kill_grace: float = 30.0
    # a checkpoint modified less than this ago may be mid-write: not trusted, not interrupted
    checkpoint_settle_s: float = 5.0
    # another member's STOP_ROUND before this run's first latest checkpoint: train on until it lands, at
    # most 2 x latest_interval epoch times, or this long while the epoch time is not known yet (graph
    # loading and JIT took ~6 min, an epoch ~200 s: 10 epochs plus that is about 40 min)
    round_stop_max_wait: float = 3600.0

    def __post_init__(self):
        unknown = sorted(set(self.actions) - set(ACTION_POLICIES))
        if unknown:
            raise ValueError(f"actions: unknown {unknown}, expected some of {ACTION_POLICIES}")
        self.actions = {**DEFAULT_ACTIONS, **self.actions}
        bad = {key: mode for key, mode in self.actions.items() if mode not in MODES}
        if bad:
            raise ValueError(f"actions: modes must be one of {MODES}, got {bad}")
        if self.monitor not in ("train", "val", "mean"):
            raise ValueError(f"monitor must be train, val or mean, got {self.monitor!r}")
        if self.resume_from not in _LOAD_MODEL_TYPES:
            raise ValueError(f"resume_from must be one of {_LOAD_MODEL_TYPES}, got {self.resume_from!r}")
        positive = ("conv_tol", "done_scale", "poll_s", "divergence_factor", "retune_slack",
                    "retune_threshold", "overfit")
        for name in positive:
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)!r}")
        for name in ("conv_stages", "divergence_epochs", "max_rounds", "latest_interval"):
            if not (isinstance(getattr(self, name), int) and getattr(self, name) >= 1):
                raise ValueError(f"{name} must be an integer >= 1, got {getattr(self, name)!r}")
        for name in ("min_epochs_before_rules", "max_divergence_restarts", "max_crash_restarts"):
            if not (isinstance(getattr(self, name), int) and getattr(self, name) >= 0):
                raise ValueError(f"{name} must be an integer >= 0, got {getattr(self, name)!r}")
        if not 0 < self.lr_drop_on_divergence < 1:
            raise ValueError(f"lr_drop_on_divergence must be in (0, 1), got {self.lr_drop_on_divergence!r}")
        if self.min_scale is not None and not 0 < self.min_scale < 1:
            raise ValueError(f"min_scale must be in (0, 1) or None, got {self.min_scale!r}")
        if not self.done_scale < 1:
            raise ValueError("done_scale must be < 1")
        if self.walltime is not None and not self.walltime > self.walltime_margin:
            raise ValueError(f"walltime ({self.walltime} s) must be larger than walltime_margin "
                             f"({self.walltime_margin} s)")
        if not (isinstance(self.train_command, list) and self.train_command
                and all(isinstance(part, str) for part in self.train_command)):
            raise ValueError(f"train_command must be a non-empty list of strings, got {self.train_command!r}")
        if not any("{toml}" in part for part in self.train_command):
            raise ValueError(f"train_command must contain {{toml}} (the train.toml path): {self.train_command!r}")
        for name in ("stop_grace", "kill_grace", "checkpoint_settle_s", "round_stop_max_wait"):
            if not getattr(self, name) >= 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)!r}")
        if self.stop_grace + self.kill_grace >= self.walltime_margin > 0:
            raise ValueError(f"stop_grace + kill_grace ({self.stop_grace + self.kill_grace} s) must fit in "
                             f"walltime_margin ({self.walltime_margin} s)")
        if self.walltime_margin < 0:
            raise ValueError(f"walltime_margin must be >= 0, got {self.walltime_margin!r}")

    def analysis_kwargs(self) -> dict:
        """For analyze_training_logs; an unresolved min_scale (only used there to replay) takes DeepH's
        default."""
        min_scale = SchedulerConfig.min_learning_rate_scale if self.min_scale is None else self.min_scale
        return {"monitor": self.monitor, "conv_tol": self.conv_tol, "conv_stages": self.conv_stages,
                "done_scale": self.done_scale, "overfit": self.overfit, "min_scale": min_scale}

    def shared_overrides(self) -> dict:
        """What the supervisor sets in every member's train.toml."""
        return {"data": {"model_save": {"latest": True, "best": True, "latest_interval": self.latest_interval}}}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping) -> "SupervisorConfig":
        names = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - names)
        if unknown:
            raise ValueError(f"SupervisorConfig: unknown keys {unknown}")
        return cls(**data)


@dataclass
class ExitInfo:
    """How one member's train job ended (produced by the watchdog, step 4). Plain types only, so it
    passes between jobflow jobs as JSON.

    ``reason``: ``finished`` (deeph-train exited by itself: min scale or ``max_epoch``),
    ``walltime`` (stopped by the watchdog before the job's time limit), ``stopped`` (a STOP file),
    ``live_rule`` (stopped on a ``live_check`` rule, see ``live_status``), ``crashed`` (exited on its
    own with an error). A job that never returned (OOM, node failure) gives no ExitInfo at all: None.
    """

    reason: str
    return_code: int | None = None
    run_dir: str | None = None
    last_epoch: int | None = None
    last_checkpoint_epoch: int | None = None
    live_status: str | None = None  # a MemberStatus value, with reason "live_rule" (or an ask alert)
    live_reason: str | None = None
    # the highest latest checkpoint on disk, trusted or not: above last_checkpoint_epoch, the resume pins
    # load_model_epoch to the trusted one
    newest_checkpoint_epoch: int | None = None

    def __post_init__(self):
        if self.reason not in EXIT_REASONS:
            raise ValueError(f"ExitInfo.reason must be one of {EXIT_REASONS}, got {self.reason!r}")
        if self.live_status is not None:
            MemberStatus(self.live_status)  # raises on an unknown status

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping) -> "ExitInfo":
        return cls(**data)


def resolve_min_scale(cfg: SupervisorConfig, shared_overrides: Mapping | None) -> SupervisorConfig:
    """``cfg`` with ``min_scale`` set from the shared train overrides
    (``process.train.scheduler.min_learning_rate_scale``, else DeepH's default), so the supervisor never
    writes another floor than the user's. Both given and different: an error."""
    scheduler = (shared_overrides or {}).get("process", {}).get("train", {}).get("scheduler", {})
    from_toml = scheduler.get("min_learning_rate_scale")
    if cfg.min_scale is not None and from_toml is not None and not math.isclose(cfg.min_scale, from_toml):
        raise ValueError(f"SupervisorConfig.min_scale = {cfg.min_scale} but the train overrides set "
                         f"min_learning_rate_scale = {from_toml}: leave min_scale None")
    if cfg.min_scale is not None:
        return cfg
    value = from_toml if from_toml is not None else SchedulerConfig.min_learning_rate_scale
    return dataclasses.replace(cfg, min_scale=float(value))


class MemberStatus(StrEnum):
    RUNNING = "running"
    CONVERGED = "converged"
    OVERFIT = "overfit"
    DIVERGED = "diverged"
    MISTUNED = "mistuned"
    FINISHED_NOT_CONVERGED = "finished_not_converged"  # exited at min scale or max_epoch, not converged
    STOPPED_WALLTIME = "stopped_walltime"
    CRASHED = "crashed"  # no ExitInfo, or an error exit not caused by the watchdog


# ---------------------------------- live rules ----------------------------------


class LiveCheck(NamedTuple):
    """``hard_stop``: the watchdog stops the run whatever the policy (a non-finite loss: nothing more is
    learnt); otherwise a stop on ``status`` follows the policy of that action (``ask``: alert only).
    Restarting after a hard stop is still gated by the ``divergence`` policy."""

    status: "MemberStatus"
    reason: str
    hard_stop: bool = False


def _non_finite(member: MemberAnalysis) -> str | None:
    """A nan/inf loss in the member's last job (an earlier one was already restarted from)."""
    curve = member.curve
    start = curve.jobs[-1].start if curve.jobs else 0
    bad = ~np.isfinite(curve.train[start:]) | ~np.isfinite(curve.val[start:])
    if bad.any():
        return f"non-finite loss at epoch {int(curve.epoch[start + np.argmax(bad)])}"
    return None


def _divergence(member: MemberAnalysis, cfg: SupervisorConfig) -> str | None:
    """W1 on the last job of the member: a nan/inf loss, or val above ``divergence_factor`` x the best
    val so far over the last ``divergence_epochs`` epochs."""
    reason = _non_finite(member)
    if reason:
        return reason
    curve = member.curve
    start = curve.jobs[-1].start if curve.jobs else 0
    val = curve.val[start:]
    n = cfg.divergence_epochs
    if len(curve.val) <= n:
        return None
    before = curve.val[:-n]
    best = np.nanmin(before) if np.isfinite(before).any() else np.nan
    tail = curve.val[-n:]
    if np.isfinite(best) and len(val) >= n and (tail > cfg.divergence_factor * best).all():
        return (f"val above {cfg.divergence_factor:g} x the best ({best:.2e}) for the last {n} epochs "
                f"(now {tail[-1]:.2e})")
    return None


def _w3_applies(member: MemberAnalysis) -> bool:
    """W3 judges the scheduler settings the run started with: only for a run started from scratch
    whose later jobs all kept them and the optimizer (after a retune or a new-optimizer restart it
    would judge a run no longer in its initial-LR segment, or settings no longer in use)."""
    jobs = member.curve.jobs
    if not jobs or jobs[0].continued_from is not None:
        return False
    if any(job.new_optimizer for job in jobs[1:]):
        return False
    keys = ("patience", "cooldown", "rtol", "factor")
    return all(job.settings[key] == jobs[0].settings[key] for job in jobs[1:] for key in keys)


def _left_initial_lr(curve) -> int | None:
    """Index of the first epoch below the initial (normalized) LR, None if still at it. Unlike
    ``MemberAnalysis.reductions`` this includes a lower LR at a join."""
    below = np.nonzero(curve.scale < curve.scale[0] * (1 - 1e-12))[0]
    return int(below[0]) if len(below) else None


def _mistuned(member: MemberAnalysis, analysis: TrainingAnalysis, cfg: SupervisorConfig) -> str | None:
    """W3: patience too large (still at the initial LR well after the onset) or too small (first
    reduction while the loss was still clearly improving: no onset before it)."""
    curve = member.curve
    if len(curve.epoch) < cfg.min_epochs_before_rules or not _w3_applies(member):
        return None
    left = _left_initial_lr(curve)
    if left is not None:
        if analysis.segments_too_short or member.onset is not None:
            return None
        return (f"patience too small: first reduction at epoch {int(curve.epoch[left])} while the loss was still "
                "clearly improving (no noise-dominated onset before it)")
    suggested = analysis.suggested
    if member.onset is None or suggested is None:
        return None
    limit = member.onset + cfg.retune_slack * suggested.patience * curve.epochs_per_check
    if curve.epoch[-1] > limit:
        return (f"patience too large: still at the initial LR at epoch {int(curve.epoch[-1])}, past the onset "
                f"{member.onset} + {cfg.retune_slack:g} x suggested patience {suggested.patience}")
    return None


def _member_status(member: MemberAnalysis, analysis: TrainingAnalysis, cfg: SupervisorConfig) -> LiveCheck:
    reason = _non_finite(member)
    if reason:
        return LiveCheck(MemberStatus.DIVERGED, reason, hard_stop=True)
    reason = _divergence(member, cfg)
    if reason:
        return LiveCheck(MemberStatus.DIVERGED, reason)
    if member.converged:
        note = " (overfitting)" if member.overfitting else ""
        return LiveCheck(MemberStatus.CONVERGED, f"converged since epoch {member.converged_since} "
                                                 f"({member.converged_reason}){note}")
    if member.overfitting:
        return LiveCheck(MemberStatus.OVERFIT, f"val/train grew from {member.stages[0].val_over_train:.2f} to "
                                               f"{member.stages[-1].val_over_train:.2f}")
    reason = _mistuned(member, analysis, cfg)
    if reason:
        return LiveCheck(MemberStatus.MISTUNED, reason)
    return LiveCheck(MemberStatus.RUNNING, "")


def live_check(member_logs: Sequence, cfg: SupervisorConfig, shared_overrides: Mapping | None = None) -> LiveCheck:
    """The watchdog's rules on one member's chained logs (in order): W1 divergence, W2 converged /
    overfitting, W3 mistuned scheduler. ``RUNNING`` when none fires.

    Contract: the **last** of ``member_logs`` is the log of the run being watched. While it does not
    exist or has no epoch line yet (the first epoch takes minutes), the result is ``RUNNING`` without
    looking at the earlier logs: they belong to runs already decided on, and judging them again would
    stop a restart for its predecessor's divergence or a retune for its old settings. Earlier logs
    without epochs are skipped. ``LiveCheck.hard_stop`` marks a non-finite loss, which stops the run
    whatever the policy."""
    cfg = resolve_min_scale(cfg, shared_overrides)
    member_logs = list(member_logs)
    if not member_logs or not has_epochs(member_logs[-1]):
        return LiveCheck(MemberStatus.RUNNING, "no epoch in the current run yet")
    member_logs = [p for p in member_logs if has_epochs(p)]
    analysis = analyze_training_logs([member_logs], **cfg.analysis_kwargs())
    return _member_status(analysis.members[0], analysis, cfg)


# ---------------------------------- round decision ----------------------------------


@dataclass
class MemberAction:
    """What to do with one member after a round. ``continue_from`` and ``fold_overrides`` are this
    fold's entries for ``write_ensemble_train_tomls`` (only for the actions that train again)."""

    fold: int
    action: str  # done | resume | retune | restart_lower_lr | give_up
    cause: str  # what led to it: crash, converged, overfit, diverged, mistuned, retune, finished, walltime, ...
    reason: str
    status: str  # a MemberStatus value
    policy: str | None = None  # the policy key that gated it; None for the limits
    mode: str = "auto"
    needs_approval: bool = False
    continue_from: dict | None = None
    fold_overrides: dict = field(default_factory=dict)
    overfitting: bool = False
    # overrides of cfg.actions for this member's watchdog in the next round, e.g. {"retune": "off"}
    # while its own W3 fires but the pooled analysis cannot act on it (no STOP_ROUND every round)
    live_actions: dict = field(default_factory=dict)

    @property
    def trains(self) -> bool:
        return self.action in ("resume", "retune", "restart_lower_lr")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RoundDecision:
    round: int
    actions: dict[int, MemberAction]
    shared_overrides: dict
    analysis: TrainingAnalysis | None = field(default=None, repr=False)  # not serialized

    @property
    def pending_folds(self) -> list[int]:
        return [j for j, a in sorted(self.actions.items()) if a.trains]

    @property
    def finished(self) -> bool:
        return not self.pending_folds

    @property
    def needs_approval(self) -> bool:
        return any(a.needs_approval for a in self.actions.values())

    @property
    def continue_from(self) -> dict[int, dict | None]:
        return {j: self.actions[j].continue_from for j in self.pending_folds}

    @property
    def fold_overrides(self) -> dict[int, dict]:
        return {j: self.actions[j].fold_overrides for j in self.pending_folds}

    @property
    def live_actions(self) -> dict[int, dict]:
        """For ``run_supervised(..., live_actions=...)`` of each member trained next."""
        return {j: self.actions[j].live_actions for j in self.pending_folds}

    def to_dict(self) -> dict:
        return {"round": self.round, "members": {str(j): a.to_dict() for j, a in sorted(self.actions.items())},
                "shared_overrides": self.shared_overrides}


def new_history() -> dict:
    return {"rounds": []}


def append_history(history: Mapping, decision: RoundDecision) -> dict:
    """A copy of ``history`` with ``decision`` added as the latest round."""
    rounds = [*history.get("rounds", []), decision.to_dict()]
    return {**history, "rounds": rounds}


def _count(history: Mapping, fold: int, cause: str, action: str) -> int:
    return sum(
        1 for r in history.get("rounds", [])
        if (m := r["members"].get(str(fold))) and m["cause"] == cause and m["action"] == action
    )


def _consecutive_crashes(history: Mapping, fold: int) -> int:
    """Crash restarts of the member in the rounds just before this one: a round that ended without a
    crash resets the count."""
    n = 0
    for r in reversed(history.get("rounds", [])):
        record = r["members"].get(str(fold))
        if record is None:
            continue
        if record["cause"] != "crash":
            break
        n += 1
    return n


def _last_training_record(history: Mapping, fold: int) -> dict:
    """``continue_from`` and ``fold_overrides`` the member's latest round was started with: the latest
    record that trained, else round 1's start (``history["start"]``, from scratch when absent)."""
    for r in reversed(history.get("rounds", [])):
        record = r["members"].get(str(fold))
        if record is not None and record["action"] in ("resume", "retune", "restart_lower_lr"):
            return record
    return history.get("start", {}).get(str(fold), {"continue_from": None, "fold_overrides": {}})


def _last_record(history: Mapping, fold: int) -> dict | None:
    for r in reversed(history.get("rounds", [])):
        if str(fold) in r["members"]:
            return r["members"][str(fold)]
    return None


def _scheduler_overrides(curve, init_lr: float, min_scale: float, patience=None, cooldown=None, rtol=None,
                         new_optimizer: bool = False) -> dict:
    scheduler = {
        "min_learning_rate_scale": float(min_scale),
        "patience": int(curve.patience if patience is None else patience),
        "cooldown": int(curve.cooldown if cooldown is None else cooldown),
        "rtol": float(curve.rtol if rtol is None else rtol),
    }
    return {"process": {"train": {
        "optimizer": {"init_learning_rate": float(init_lr)},
        "scheduler": scheduler,
        "continued": {"new_optimizer": new_optimizer},
    }}}


def _lr0_first(curve) -> float | None:
    return curve.jobs[0].lr0 if curve.jobs and curve.jobs[0].lr0 is not None else curve.lr0


def _lr_at(curve, i: int) -> float | None:
    """The LR in effect at epoch index ``i`` (in the last job: its lr0 x its logged scale)."""
    if curve.lr0 is not None:
        return curve.lr0 * float(curve.raw_scale[i])
    lr0_first = _lr0_first(curve)
    return None if lr0_first is None else lr0_first * float(curve.scale[i])


def carry_over(curve, cfg: SupervisorConfig, drop: float = 1.0, at: str = "last") -> tuple[float, float] | None:
    """Decision 4: ``(init_learning_rate, min_learning_rate_scale)`` of a new-optimizer restart, from the
    LR x ``drop`` at ``at``: ``"last"``, the end of the run (a retune: the decay done is kept), or
    ``"best"``, the last job's best-val epoch, the checkpoint a divergence restart loads (the scheduler
    kept reducing on the diverged epochs after it). None when the member is already at the floor (the
    min scale would be >= 1) or the logs have no initial LR. ``cfg.min_scale`` must be resolved."""
    if at == "last":
        lr = curve.current_learning_rate
    elif at == "best":
        start = curve.jobs[-1].start if curve.jobs else 0
        val = curve.val[start:]
        if not np.isfinite(val).any():
            return None
        lr = _lr_at(curve, start + int(np.nanargmin(np.where(np.isfinite(val), val, np.nan))))
    else:
        raise ValueError(f"at must be 'last' or 'best', got {at!r}")
    if cfg.min_scale is None:
        raise ValueError("carry_over needs a resolved min_scale: use resolve_min_scale(cfg, shared_overrides)")
    lr0_first = _lr0_first(curve)
    if lr is None or lr0_first is None:
        return None
    lr *= drop
    scale_carried = lr / lr0_first
    min_scale = cfg.min_scale / scale_carried
    if min_scale >= 1 - 1e-9:
        return None
    return lr, min_scale


def _resume_overrides(curve, cfg: SupervisorConfig) -> dict:
    """A plain resume keeps the optimizer and the member's current settings: its last lr0, the min
    scale relative to that lr0, patience, cooldown and rtol (a retuned member keeps its retune)."""
    if curve.lr0 is None:
        return {"process": {"train": {"continued": {"new_optimizer": False}}}}
    ratio = curve.jobs[-1].lr_ratio if curve.jobs else 1.0
    return _scheduler_overrides(curve, curve.lr0, cfg.min_scale / ratio)


def _gate(action: MemberAction, cfg: SupervisorConfig, resume: MemberAction | None) -> MemberAction:
    """Apply the policy mode of ``action.policy``: ``auto`` keeps it, ``ask`` keeps it and flags it,
    ``off`` replaces it by what not acting means: the plain ``resume`` for the stop actions and a
    retune (itself gated by the resume policy), done for the resume itself and for a finished run,
    which cannot train further, and give_up for a divergence restart (resuming a diverged run is
    pointless). The limits (``policy`` None) are not gated."""
    if action.policy is None:
        return action
    mode = cfg.actions[action.policy]
    action.mode = mode
    if mode == "ask":
        action.needs_approval = True
    elif mode == "off":
        if action.policy == "divergence":  # resuming a diverged run is pointless
            return MemberAction(action.fold, "give_up", action.cause, f"divergence is off: {action.reason}",
                                action.status, action.policy, mode, overfitting=action.overfitting)
        if action.policy in ("resume", "finished_not_converged") or resume is None:
            return MemberAction(action.fold, "done", action.cause, f"{action.policy} is off: {action.reason}",
                                action.status, action.policy, mode, overfitting=action.overfitting)
        resume.reason = f"{action.policy} is off: {action.reason}"
        resume.status, resume.overfitting = action.status, action.overfitting
        return _gate(resume, cfg, None)
    return action


def _differs(member: MemberAnalysis, analysis: TrainingAnalysis, cfg: SupervisorConfig) -> str | None:
    """The pooled suggestion is off the member's current settings by more than ``retune_threshold``."""
    s = analysis.suggested
    if s is None or analysis.segments_too_short:
        return None
    curve = member.curve
    off = abs(curve.patience - s.patience) / s.patience
    if off > cfg.retune_threshold or not math.isclose(curve.rtol, s.rtol):
        return (f"pooled suggestion patience {s.patience}, cooldown {s.cooldown}, rtol {s.rtol} vs current "
                f"patience {curve.patience}, cooldown {curve.cooldown}, rtol {curve.rtol}")
    return None


def round_decision(
    member_logs: Mapping[int, Sequence],
    exit_infos: Mapping[int, ExitInfo | Mapping | None],
    cfg: SupervisorConfig,
    history: Mapping | None = None,
    shared_overrides: Mapping | None = None,
    round_: int | None = None,
) -> RoundDecision:
    """One ``MemberAction`` per member after round ``round_`` (the job's round; checked against the
    history, which must hold exactly the rounds before it; default: the one after the history).

    ``member_logs``: fold -> that member's chained logs, oldest first, for **all** members (the patience
    suggestion pools them); logs that do not exist or have no epoch line yet are left out.
    ``exit_infos``: fold -> ``ExitInfo`` (or its dict) of the members trained this round; None for a job
    that never returned. A member not in it was not trained this round: its last action stands.
    ``history``: the previous rounds (module docstring). Rules, in order: crash, converged, overfit,
    diverged, mistuned or suggestion differs, finished without converging, resume; then the policy gate
    and the limits (``max_rounds``, restart caps, LR floor). ``shared_overrides``: the train overrides
    shared by all members, for the user's ``min_learning_rate_scale`` (``resolve_min_scale``)."""
    cfg = resolve_min_scale(cfg, shared_overrides)
    history = history if history is not None else new_history()
    n_before = len(history.get("rounds", []))
    if round_ is None:
        round_ = n_before + 1
    elif round_ != n_before + 1:
        raise ValueError(f"round {round_}: the history holds {n_before} rounds, it must hold the "
                         f"{round_ - 1} before it")
    logs = {j: [p for p in member_logs[j] if has_epochs(p)] for j in sorted(member_logs)}
    unknown = sorted(set(exit_infos) - set(logs))
    if unknown:
        raise ValueError(f"exit_infos has folds {unknown} that are not in member_logs")
    with_logs = [j for j in logs if logs[j]]
    analysis = analyze_training_logs([logs[j] for j in with_logs], **cfg.analysis_kwargs()) if with_logs else None
    members = dict(zip(with_logs, analysis.members, strict=True)) if analysis else {}
    last_round = round_ >= cfg.max_rounds

    actions = {}
    for j, fold_logs in logs.items():
        if j not in exit_infos:
            last = _last_record(history, j)
            action = last["action"] if last and last["action"] in ("done", "give_up") else "done"
            actions[j] = MemberAction(j, action, "earlier", "not trained this round", MemberStatus.RUNNING.value)
            if last:
                actions[j].status, actions[j].overfitting = last["status"], last.get("overfitting", False)
            continue
        info = exit_infos[j]
        info = ExitInfo.from_dict(info) if isinstance(info, Mapping) else info
        action = _decide(j, info, members.get(j), analysis, fold_logs, cfg, history)
        action.live_actions = _w3_suppression(action, fold_logs, cfg)
        if action.live_actions:
            action.reason += "; its watchdog will not stop the round for W3 again (the pool cannot act on it)"
        if last_round and action.trains:
            action = MemberAction(j, "give_up", "max_rounds", f"max_rounds ({cfg.max_rounds}) reached; would have "
                                  f"been {action.action}: {action.reason}", action.status, None,
                                  overfitting=action.overfitting)
        actions[j] = action
    return RoundDecision(round_, actions, cfg.shared_overrides(), analysis)


def _crashed_run_has_checkpoint(info, member, logs, history, j, cfg) -> bool:
    """Whether the crashed run saved a ``latest`` checkpoint: from ``ExitInfo.last_checkpoint_epoch``,
    else (no ExitInfo) from its log. DeepH saves a ``latest`` at a run's first epoch and at the multiples
    of ``latest_interval`` of the (continued) epoch numbers, and logs an epoch only after its checkpoint
    is saved (DeepH-pack 1.0.6): so a logged epoch means a saved checkpoint. The crashed run's log is the last one
    unless it is the run the round started from (it logged nothing)."""
    if member is None or not logs:
        return False
    if info is not None:
        return bool(info.last_checkpoint_epoch)
    started_from = (_last_training_record(history, j).get("continue_from") or {}).get("previous_output_dir")
    if started_from is not None and Path(started_from).resolve() == Path(logs[-1]).parent.resolve():
        return False
    return member.curve.jobs[-1].n_epochs >= 1


def _pin_checkpoint(action: MemberAction, info: ExitInfo | None) -> None:
    """Resume from the trusted latest checkpoint when a newer, untrusted one (possibly cut mid-write)
    is on disk: ``continued.load_model_epoch``; otherwise DeepH's -1 (the newest) is right."""
    if (info is None or not info.last_checkpoint_epoch or not info.newest_checkpoint_epoch
            or info.newest_checkpoint_epoch <= info.last_checkpoint_epoch
            or (action.continue_from or {}).get("load_model_type") != "latest"):
        return
    action.fold_overrides = deep_merge(action.fold_overrides, {"process": {"train": {"continued": {
        "load_model_epoch": int(info.last_checkpoint_epoch)}}}})
    action.reason += (f"; pinned to latest epoch {info.last_checkpoint_epoch} (epoch {info.newest_checkpoint_epoch}"
                      " on disk is not trusted)")


def _w3_suppression(action: MemberAction, logs, cfg: SupervisorConfig) -> dict:
    """``{"retune": "off"}`` for a member that trains again without a retune while its own W3 (what its
    watchdog judges: this member alone) still fires: the pooled analysis could not act on it, so the
    watchdog must not stop the round for it again. Recomputed every round, so it ends with the retune,
    or when W3 no longer fires (e.g. the member left its initial LR after the onset)."""
    if not action.trains or action.action == "retune" or not logs or cfg.actions["retune"] != "auto":
        return {}  # only "auto" stops the round on W3; "ask" just alerts once per run
    solo = analyze_training_logs([logs], **cfg.analysis_kwargs())
    return {"retune": "off"} if _mistuned(solo.members[0], solo, cfg) else {}


def _decide(j, info: ExitInfo | None, member: MemberAnalysis | None, analysis, logs, cfg, history) -> MemberAction:
    run_dir = str(Path(logs[-1]).parent) if logs else None
    if info is not None and info.run_dir is not None:
        run_dir = info.run_dir

    def resume(cause, reason, status):
        if member is None:  # no epoch logged yet: start again from scratch
            return MemberAction(j, "resume", cause, reason, status, "resume",
                                fold_overrides={"process": {"train": {"continued": {"new_optimizer": False}}}})
        return MemberAction(j, "resume", cause, reason, status, "resume",
                            continue_from={"previous_output_dir": run_dir, "load_model_type": cfg.resume_from},
                            fold_overrides=_resume_overrides(member.curve, cfg))

    # 1. crash
    if info is None or info.reason == "crashed":
        n = _consecutive_crashes(history, j)
        what = "no exit info (the job never returned)" if info is None else f"exit code {info.return_code}"
        if n >= cfg.max_crash_restarts:
            return MemberAction(j, "give_up", "crash", f"crashed ({what}) after {n} consecutive crash restarts",
                                MemberStatus.CRASHED.value)
        restart = f"crash restart {n + 1}/{cfg.max_crash_restarts}"
        if not _crashed_run_has_checkpoint(info, member, logs, history, j, cfg):
            # nothing saved by the crashed run: start it again exactly as it was started
            previous = _last_training_record(history, j)
            return _gate(MemberAction(
                j, "resume", "crash", f"crashed ({what}) before its first latest checkpoint; {restart}: "
                f"started again as before", MemberStatus.CRASHED.value, "resume",
                continue_from=copy.deepcopy(previous["continue_from"]),
                fold_overrides=copy.deepcopy(previous["fold_overrides"])), cfg, None)
        # the crashed run's own settings (its log's header), from its last checkpoint
        action = resume("crash", f"crashed ({what}); {restart} from latest", MemberStatus.CRASHED.value)
        action.continue_from["load_model_type"] = "latest"
        _pin_checkpoint(action, info)
        return _gate(action, cfg, None)

    status, reason, _ = _member_status(member, analysis, cfg) if member is not None else (MemberStatus.RUNNING, "", 0)
    if info.live_status == MemberStatus.DIVERGED and status != MemberStatus.DIVERGED:
        status, reason = MemberStatus.DIVERGED, f"watchdog: {info.live_reason or 'diverged'}"
    overfitting = bool(member is not None and member.overfitting)
    plain = resume("walltime" if info.reason == "walltime" else info.reason, f"{info.reason}: resume from "
                   f"{cfg.resume_from}", MemberStatus.STOPPED_WALLTIME.value if info.reason == "walltime"
                   else MemberStatus.RUNNING.value)
    plain.overfitting = overfitting
    if not info.last_checkpoint_epoch and (member is None or cfg.resume_from == "latest"):
        # stopped (another member's STOP_ROUND, the walltime) before this run saved a latest checkpoint:
        # start it again as it was started, which is no crash
        previous = _last_training_record(history, j)
        plain.continue_from = copy.deepcopy(previous["continue_from"])
        plain.fold_overrides = copy.deepcopy(previous["fold_overrides"])
        plain.reason += "; no latest checkpoint in this run: started again as before"
    else:
        _pin_checkpoint(plain, info)

    def gated(action):
        action.overfitting = overfitting
        return _gate(action, cfg, plain)

    # 2. converged, 3. overfit
    if status == MemberStatus.CONVERGED:
        return gated(MemberAction(j, "done", "converged", reason, status.value, "stop_converged"))
    if status == MemberStatus.OVERFIT:
        return gated(MemberAction(j, "done", "overfit", f"overfitting: {reason} (best-val checkpoint kept)",
                                  status.value, "stop_overfit"))
    # 4. diverged
    if status == MemberStatus.DIVERGED:
        n = _count(history, j, "diverged", "restart_lower_lr")
        if n >= cfg.max_divergence_restarts:
            return MemberAction(j, "give_up", "diverged", f"{reason}; after {n} divergence restarts",
                                status.value, overfitting=overfitting)
        carried = carry_over(member.curve, cfg, cfg.lr_drop_on_divergence, at="best") if member is not None else None
        if carried is None:
            return MemberAction(j, "give_up", "diverged", f"{reason}; no LR to restart from (at the floor, or no "
                                "initial LR in the logs)", status.value, overfitting=overfitting)
        lr, min_scale = carried
        return gated(MemberAction(
            j, "restart_lower_lr", "diverged",
            f"{reason}; restart {n + 1}/{cfg.max_divergence_restarts} from best at LR {lr:.3g}", status.value,
            "divergence", continue_from={"previous_output_dir": run_dir, "load_model_type": "best"},
            fold_overrides=_scheduler_overrides(member.curve, lr, min_scale, new_optimizer=True)))
    # 5. mistuned, or the pooled suggestion differs
    if member is not None:
        differs = _differs(member, analysis, cfg)
        mistuned = status == MemberStatus.MISTUNED or info.live_status == MemberStatus.MISTUNED
        if (mistuned or differs) and analysis.suggested is not None and not analysis.segments_too_short:
            s = analysis.suggested
            if status == MemberStatus.MISTUNED:
                why = reason
            elif mistuned:
                why = f"watchdog: {info.live_reason or 'mistuned'}"
            else:
                why = differs
            curve = member.curve
            same = (s.patience == curve.patience and s.cooldown == curve.cooldown and math.isclose(s.rtol, curve.rtol))
            if same:  # nothing to change: a retune would only reset the optimizer
                plain.reason += f" ({why}; the suggestion is the current settings, no retune)"
                return _gate(plain, cfg, None)
            carried = carry_over(member.curve, cfg)
            if carried is not None:
                lr, min_scale = carried
                return gated(MemberAction(
                    j, "retune", "mistuned" if mistuned else "retune",
                    f"{why}; retune to patience {s.patience}, cooldown {s.cooldown}, rtol {s.rtol} from best at LR "
                    f"{lr:.3g}", MemberStatus.MISTUNED.value, "retune",
                    continue_from={"previous_output_dir": run_dir, "load_model_type": "best"},
                    fold_overrides=_scheduler_overrides(member.curve, lr, min_scale, s.patience, s.cooldown,
                                                        s.rtol, new_optimizer=True)))
    # 6. finished (min scale or max_epoch), or at the LR floor, without converging
    at_floor = member is not None and member.curve.scale[-1] <= cfg.min_scale * (1 + 1e-9)
    if info.reason == "finished" or at_floor:
        where = "at the LR floor" if at_floor else "deeph-train exited by itself (min scale or max_epoch)"
        return gated(MemberAction(j, "give_up", "finished", f"{where} without converging",
                                  MemberStatus.FINISHED_NOT_CONVERGED.value, "finished_not_converged"))
    # 7. resume
    return _gate(plain, cfg, None)
