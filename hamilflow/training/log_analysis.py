"""LR reductions, convergence and ``patience`` / ``cooldown`` suggestions from DeepH training logs.

For ``type = "reduce_on_plateau"``. The scheduler is replayed as ``optax.contrib.reduce_on_plateau``
(DeepH-pack user guide, "Process: Train: Scheduler"): the monitored loss is averaged over
``accum_size`` optimizer steps, an average counts as an improvement only if it is below
``(1 - rtol) * best``, after ``patience`` averages without improvement the LR scale is multiplied
by ``factor``, then the counter is frozen for ``cooldown`` averages; training stops when the scale
reaches ``min_learning_rate_scale``. With ``accum_size = -1`` the accumulation is one epoch of
batches, so patience and cooldown are in epochs.

Monitored loss: with ``only_use_train_loss = false`` the scheduler "considers both train and
validation loss"; how they are combined is not documented, so ``monitor`` chooses train / val /
mean. The default is val: on a real log with 10 reductions, only val reproduced all of them. On logs
with actual reductions, ``MemberAnalysis.replay_mismatch`` shows which choice reproduces the logged
LR scale.

Suggestion logic, on the part of each log run at the initial LR (the replay of other parameters
is only meaningful up to the first actual reduction):

1. noise: std of log10(monitored loss) around a running median (window ``smooth``);
2. noise-dominated onset: first epoch where the smoothed loss drops by less than ``k`` x noise
   over the previous ``window`` epochs, i.e. where progress at this LR is buried in the
   fluctuations and lowering the LR is worth it;
3. patience: smallest one in the grid whose replayed first reduction falls (median over members)
   at or after the onset, for each rtol;
4. cooldown: median post-reduction transient measured on actual reductions in the logs when there
   are any, otherwise patience // 2 (heuristic).

Chained jobs: a member resumed over several jobs is read as one curve. A job continued with
``new_optimizer = true`` logs its scale relative to its own ``initial learning rate`` (it restarts
at 1.0), so ``TrainingCurve.scale`` is normalized to the first log's initial LR,
``scale x lr0_job / lr0_first``, and the stages, reductions and convergence use it; the scale as
logged stays in ``raw_scale`` for the per-job replay of the scheduler. ``min_scale`` and
``done_scale`` are relative to the first log's initial LR too.

Convergence: for every LR stage of each member, the best val loss reached and its gain over the
best of the previous stages. A member is converged when its last ``conv_stages`` completed
reduced-LR stages each improved the best val loss by less than ``conv_tol``, or when the LR scale
is down to ``done_scale`` (updates negligible); it is flagged as overfitting when val/train grew by
more than ``overfit`` x since the first stage. Killing the run at that point is safe with
``data.model_save.best = true``, which keeps the best-val checkpoint.
"""
import itertools
import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

# a number, or nan / inf as Python prints them (any case, signed): a diverged epoch must stay a row
_NUMBER = r"([\d.eE+-]+|[+-]?(?i:nan|inf(?:inity)?))"
_EPOCH_RE = re.compile(
    rf"Epoch (\d+) \| Time ([\d.]+) s \| Train-Loss {_NUMBER} \| Val-Loss {_NUMBER} \| Scale {_NUMBER}"
)
_HEADER_RE = {
    "n_train": r"Train size: `(\d+)`",
    "batch_size": r"Batch size: `(\d+)`",
    "lr0": r"initial learning rate `([\d.eE+-]+)`",
    "factor": r"factor `([\d.eE+-]+)`",
    "patience": r"patience `(\d+)`",
    "rtol": r"rtol `([\d.eE+-]+)`",
    "cooldown": r"cooldown `(\d+)`",
    "accumulation_size": r"accumulation size `(\d+)`",
}
# ``[model] We are loading the `best` model inside `<run dir>`. We are using the origin training dataset, and
# applying new optimizer.`` (DeepH-pack 1.0.6, new_optimizer = true). The other variants (new training data,
# optimizer kept) were not seen in a real log: the wording after the run dir is matched loosely.
_CONTINUED_RE = re.compile(r"We are loading the `(\w+)` model inside `([^`]*)`\.?([^\n]*)")
# case-insensitive: the real line with the optimizer kept says "... and NOT applying new optimizer."
_NEW_OPTIMIZER_RE = re.compile(r"\bnew optimizer", re.IGNORECASE)
_NOT_NEW_OPTIMIZER_RE = re.compile(
    r"\b(?:not|without|no)\b[^.,]*\bnew optimizer|\b(?:origin|original|old|previous) optimizer", re.IGNORECASE
)
_TRAINING_DATA_RE = re.compile(r"\b(origin|original|new) training data", re.IGNORECASE)
_MONITORS = ("train", "val", "mean")
_SETTINGS = ("n_train", "batch_size", "factor", "patience", "rtol", "cooldown", "accumulation_size")
# relative difference below which a job's first normalized scale continues the previous job's last one (the
# initial learning rate is printed with limited precision in the log)
_JOIN_RTOL = 1e-3


@dataclass
class JobLog:
    """One log of a member (one job): its scheduler settings and how it was started."""

    path: str
    start: int  # index of its first epoch in the curve
    n_epochs: int
    lr0: float | None
    settings: dict  # header values of this log, None when missing (keys: _SETTINGS)
    continued_from: str | None = None  # run dir it was loaded from; None for a run started from scratch
    load_model_type: str | None = None  # "best" / "latest"
    new_optimizer: bool | None = None  # None without a continuation line
    new_training_data: bool | None = None  # None when the line does not say
    # raw (logged) scale x lr_ratio = scale relative to the first log's lr0
    lr_ratio: float = 1.0
    # epochs before this job cut from the curve, the dead branch this job's model does not descend
    # from: after the previous log's best val when this job loaded `best`, the trailing epochs at a
    # lower scale than this job starts with when it kept the optimizer, and whole logs (``dropped``)
    # between the log it continues and itself (see _cut_abandoned)
    n_abandoned: int = 0
    index: int = 0  # position of the log in the paths given; logs dropped whole leave gaps
    dropped: list = field(default_factory=list)  # (index, n_epochs) of the logs before it dropped whole

    @property
    def number(self) -> int:
        """1-based job number, as in the report: the position among the logs given."""
        return self.index + 1

    @property
    def stop(self) -> int:
        return self.start + self.n_epochs


@dataclass
class TrainingCurve:
    """One ensemble member, possibly over several jobs; the settings are those of the last log.

    ``scale`` is relative to the first log's initial LR (continuous across a ``new_optimizer``
    restart), ``raw_scale`` is the scale as logged, relative to each job's own initial LR."""

    name: str
    epoch: np.ndarray
    time: np.ndarray
    train: np.ndarray
    val: np.ndarray
    scale: np.ndarray
    joins: np.ndarray  # index of the first epoch of each later log
    n_train: int
    batch_size: int
    factor: float
    patience: int
    rtol: float
    cooldown: int
    accumulation_size: int  # optimizer steps per scheduler check
    lr0: float | None
    raw_scale: np.ndarray | None = None  # as logged; None: same as scale
    jobs: list[JobLog] = field(default_factory=list)
    raw_epoch: np.ndarray | None = None  # epoch numbers as logged (a kept optimizer continues them); None: epoch

    def __post_init__(self):
        if self.raw_scale is None:
            self.raw_scale = self.scale
        if self.raw_epoch is None:
            self.raw_epoch = self.epoch

    @property
    def current_learning_rate(self) -> float | None:
        """LR in effect at the last epoch: the last log's lr0 x its last logged scale. Without lr0 in
        the last log, its scale is taken as relative to the first log's lr0, as for ``scale``."""
        if self.lr0 is not None:
            return self.lr0 * float(self.raw_scale[-1])
        lr0_first = self.jobs[0].lr0 if self.jobs else None
        return None if lr0_first is None else lr0_first * float(self.scale[-1])

    @property
    def steps_per_epoch(self) -> int:
        return max(1, round(self.n_train / self.batch_size))

    @property
    def epochs_per_check(self) -> float:
        return self.accumulation_size / self.steps_per_epoch

    def monitored(self, monitor: str) -> np.ndarray:
        return {"train": self.train, "val": self.val, "mean": 0.5 * (self.train + self.val)}[monitor]


@dataclass
class LRStage:
    """A stretch of epochs at one LR scale. A new job that continues at the same scale does not
    start a new stage (``jobs`` counts the jobs the stage spans): splitting there would judge the
    convergence on fragments, whose gains are each a part of the stage's gain."""

    scale: float
    start: int
    end: int
    n_epochs: int
    best_val: float
    cumulative_best_val: float
    gain: float  # relative improvement over the best of the previous stages; nan on the first one
    val_over_train: float  # median over the last 10 epochs of the stage
    running: bool
    restart: bool  # the stage starts with a new job
    jobs: int = 1  # number of jobs the stage spans


@dataclass
class MemberAnalysis:
    curve: TrainingCurve
    label: str
    stages: list[LRStage]
    reductions: list[int]  # epochs of the first check at each lower LR scale
    # per log: monitor -> None if the replay reproduces the logged scale, else the first wrong epoch.
    # Only logs with reductions are included.
    replay_mismatch: dict[str, dict[str, int | None]]
    noise: float  # std of log10(monitored loss), in the initial-LR segment
    onset: int | None  # noise-dominated onset epoch
    replayed_first_reduction: int | None  # with the settings in the log
    converged_since: int | None
    converged_reason: str | None
    overfitting: bool
    # scheduler checks of the initial-LR segment, what the patience scan replays
    check_loss: np.ndarray
    check_epoch: np.ndarray
    epochs_per_replay_check: int
    # its initial-LR segment is too short to judge (shorter than 2 x max(window, smooth)): left out of
    # the pooled patience scan, where "never reduces" would count it as "later than the end of its log"
    segment_too_short: bool = False

    @property
    def converged(self) -> bool:
        return self.converged_since is not None


@dataclass
class PatienceSuggestion:
    """``patience`` and ``cooldown`` are the values for train.toml, in scheduler checks. Both are
    None when no patience works at this ``rtol``, with the reason in ``note``."""

    rtol: float
    patience: int | None = None
    cooldown: int | None = None
    cooldown_measured: bool = False
    first_reduction: float | None = None  # median over members
    first_reduction_range: tuple[int, int] | None = None
    n_never: int = 0  # members with no replayed reduction within their log
    min_total_epochs: float | None = None  # first reduction + later stages at cooldown + patience each
    note: str = ""


@dataclass
class TrainingAnalysis:
    members: list[MemberAnalysis]
    monitor: str
    onset: float | None  # median over members
    segments_too_short: bool
    patience_grid: list[int]  # in epochs
    replayed_first_reductions: dict[float, dict[int, float]]  # rtol -> patience [epochs] -> median epoch or inf
    suggestions: list[PatienceSuggestion]
    suggested: PatienceSuggestion | None  # at the rtol of the logs if possible
    n_reductions_to_end: int
    median_epoch_time: float
    walltime_h: float | None
    conv_tol: float
    conv_stages: int
    done_scale: float
    min_scale: float

    @property
    def all_converged(self) -> bool:
        return all(m.converged for m in self.members)


def _read_log(path) -> tuple[np.ndarray, dict]:
    text = Path(path).read_text()
    rows = np.array([[float(x) for x in m.groups()] for m in _EPOCH_RE.finditer(text)])
    if not len(rows):
        raise ValueError(f"No epoch lines in {path}")
    _carry_scale_forward(rows[:, 4])
    header = {}
    for key, pattern in _HEADER_RE.items():
        match = re.search(pattern, text)
        header[key] = float(match.group(1)) if match else None
    continued = _parse_continuation(text)
    header["optimizer_stated"] = continued.pop("optimizer_stated")
    header["continued"] = continued
    return rows, header


def has_epochs(path) -> bool:
    """Whether ``path`` is a log with at least one epoch line (False for a missing file)."""
    path = Path(path)
    return path.is_file() and _EPOCH_RE.search(path.read_text()) is not None


def _carry_scale_forward(scale: np.ndarray) -> None:
    """In place: a nan/inf scale (a diverged epoch) takes the last finite one, 1.0 before any, so
    that it neither splits a stage nor looks like a reduction."""
    last = 1.0
    for i, value in enumerate(scale):
        if np.isfinite(value):
            last = value
        else:
            scale[i] = last


def _parse_continuation(text: str) -> dict:
    """``continued_from``, ``load_model_type``, ``new_optimizer``, ``new_training_data`` from the
    continuation line; all None for a run started from scratch."""
    match = _CONTINUED_RE.search(text)
    if match is None:
        return {"continued_from": None, "load_model_type": None, "new_optimizer": None, "new_training_data": None,
                "optimizer_stated": False}
    load_type, run_dir, rest = match.groups()
    data = _TRAINING_DATA_RE.search(rest)
    new, not_new = bool(_NEW_OPTIMIZER_RE.search(rest)), bool(_NOT_NEW_OPTIMIZER_RE.search(rest))
    return {
        "continued_from": run_dir,
        "load_model_type": load_type,
        "new_optimizer": new and not not_new,
        "new_training_data": None if data is None else data.group(1).lower() == "new",
        # whether the line says anything about the optimizer: if not, "kept" is only inferred
        "optimizer_stated": new or not_new,
    }


def _kept_optimizer(header: dict, previous: dict) -> bool:
    """Whether a job continued the previous job's optimizer (and scheduler) state: its continuation
    line says so, or, without a continuation line, its lr0 is the previous one's (or unknown)."""
    new_optimizer = header["continued"]["new_optimizer"]
    if new_optimizer is not None:
        return not new_optimizer
    return header["lr0"] is None or previous["lr0"] is None or header["lr0"] == previous["lr0"]


def _cut_abandoned(paths: list, logs: list[tuple[np.ndarray, dict]]) -> tuple[list[int], list, list[int], list]:
    """Drop the dead branches before each job, the epochs the model of the new job does not descend
    from. Keeping them would show a redone reduction and a spurious flat stage.

    0. The job's continuation line names the run dir of an earlier log than the previous one (a run
       that crashed and was started again as before): the logs in between are dead branches and are
       left out whole; the rules below then apply to the named log.
    1. The job loaded the `best` model: the epochs of the log it continues after its best val
       (nanargmin within that log) are dead.
    2. The job kept the optimizer and its first scale (on the first log's lr0) is above the last one
       of the log it continues: it loaded a checkpoint saved before reductions it now redoes
       (``latest``, saved every ``latest_interval`` epochs, or ``best``), so that log's trailing
       epochs at a scale below its first one are dead. Model-free: works whatever checkpoint was loaded.

    Run dirs are compared by name only (paths differ between the cluster and a local copy). Returns
    the indices of the logs kept, those logs (cut), and per log kept the number of epochs left out
    before it (all rules together) and the ``(index, n_epochs)`` of the logs dropped whole before it."""
    rows = [r for r, _ in logs]
    headers = [header for _, header in logs]
    lr0_first = headers[0]["lr0"]

    def ratio(header):
        return 1.0 if lr0_first is None or header["lr0"] is None else header["lr0"] / lr0_first

    n_logged = [len(r) for r in rows]
    kept = [0]
    n_abandoned, dropped = {0: 0}, {0: []}
    for j in range(1, len(logs)):
        continued = headers[j]["continued"]
        run_name = None if continued["continued_from"] is None else Path(continued["continued_from"].rstrip("/")).name
        m = kept[-1]  # the log this job continues: by default the previous one kept
        if run_name is not None and run_name != Path(paths[m]).parent.name:
            match = [k for k in kept[:-1] if Path(paths[k]).parent.name == run_name]
            if match:
                m = match[-1]
        # a log dropped whole takes along the logs it had dropped itself
        dropped[j] = sorted(entry for k in kept[kept.index(m) + 1 :] for entry in [(k, n_logged[k]), *dropped[k]])
        kept = kept[: kept.index(m) + 1]
        previous = rows[m]
        if continued["load_model_type"] == "best" and not np.isnan(previous[:, 3]).all():
            if run_name != Path(paths[m]).parent.name:
                warnings.warn(
                    f"{paths[j]} loads the best model of run {run_name!r}, but the previous log is in "
                    f"{Path(paths[m]).parent.name!r}: cutting the previous log after its best val anyway",
                    stacklevel=3,
                )
            previous = previous[: int(np.nanargmin(previous[:, 3])) + 1]
        if _kept_optimizer(headers[j], headers[m]):
            before = len(previous)
            first_epoch = rows[j][0, 0]
            if first_epoch > rows[m][0, 0]:
                # DeepH continues the epoch numbers when the optimizer is kept (it loaded the checkpoint of
                # epoch first_epoch - 1): the previous log's epochs from first_epoch on are dead, exactly
                previous = previous[previous[:, 0] < first_epoch]
                how = f"its epochs from {int(first_epoch)} on (the new log continues from there) are cut"
            else:
                # the numbering does not continue (an older DeepH?): fall back on the scale, a resume that
                # starts at a higher LR than the previous log ended with redoes reductions
                first = rows[j][0, 4] * ratio(headers[j])
                scale = previous[:, 4] * ratio(headers[m])
                if first > scale[-1] * (1 + _JOIN_RTOL):
                    keep = len(scale)
                    while keep > 1 and scale[keep - 1] < first * (1 - _JOIN_RTOL):
                        keep -= 1
                    previous = previous[:keep]
                    warnings.warn(
                        f"the epoch numbers of {paths[j]} do not continue those of {paths[m]} although the optimizer "
                        "was kept: the previous log's trailing epochs at a lower LR than the new one starts with "
                        "are cut (scale heuristic)",
                        stacklevel=3,
                    )
                how = "the previous log's trailing epochs at a lower LR are cut"
            if len(previous) < before and not headers[j]["optimizer_stated"]:
                warnings.warn(
                    f"{paths[j]} does not say whether it kept the optimizer: taken as a resume of {paths[m]}; {how}",
                    stacklevel=3,
                )
        # all the logged epochs of the dropped logs, and those of the log continued that are gone (also
        # by the cuts of the dropped logs, which are now part of this dead branch)
        n_abandoned[j] = n_logged[m] - len(previous) + sum(n for _, n in dropped[j])
        rows[m] = previous
        kept.append(j)
    return kept, [(rows[k], headers[k]) for k in kept], [n_abandoned[k] for k in kept], [dropped[k] for k in kept]


def _job_logs(paths: list, logs: list[tuple[np.ndarray, dict]], n_abandoned: list[int], indices: list[int],
              dropped: list) -> list[JobLog]:
    """Per log: settings, continuation and the factor that puts its scale on the first log's lr0. A
    job whose normalized scale starts within ``_JOIN_RTOL`` of where the previous one ended is
    snapped onto it, so the limited precision of the printed lr0 does not split a stage."""
    lr0_first = logs[0][1]["lr0"]
    if lr0_first is None and len(paths) > 1:
        warnings.warn(f"No initial learning rate in {paths[0]}: the scales of all the logs are taken as "
                      "relative to the same initial LR", stacklevel=3)
    jobs, start, previous_last = [], 0, None
    for path, (rows, header), cut, index, gone in zip(paths, logs, n_abandoned, indices, dropped, strict=True):
        if lr0_first is None:
            ratio = 1.0
        elif header["lr0"] is None:
            warnings.warn(f"No initial learning rate in {path}: its scale is taken as relative to the "
                          "first log's initial LR", stacklevel=3)
            ratio = 1.0
        else:
            ratio = header["lr0"] / lr0_first
        raw_first, raw_last = rows[0, 4], rows[-1, 4]
        if previous_last is not None and raw_first > 0 and np.isclose(raw_first * ratio, previous_last,
                                                                      rtol=_JOIN_RTOL, atol=0):
            ratio = previous_last / raw_first
        jobs.append(JobLog(
            path=str(path), start=start, n_epochs=len(rows), lr0=header["lr0"],
            settings={key: header[key] for key in _SETTINGS}, lr_ratio=float(ratio), n_abandoned=cut,
            index=index, dropped=gone, **header["continued"],
        ))
        start, previous_last = start + len(rows), float(raw_last * ratio)
    return jobs


def read_training_curve(paths) -> TrainingCurve:
    """Read one member from its log, or from the logs of its consecutive jobs (a sequence of paths,
    in order, for ``[process.train.continued]``); epochs are renumbered consecutively, and the
    scale is put on the first log's initial LR (see the module docstring). Before a job that
    loaded the `best` model, the previous log's epochs after its best val are left out
    (``JobLog.n_abandoned``), so the curve follows the lineage of the model; a log that is a dead
    branch as a whole (a crashed run started again as before) is left out of ``jobs`` and listed
    in the next job's ``dropped``."""
    paths = [paths] if isinstance(paths, str | Path) else list(paths)
    logs = [_read_log(p) for p in paths]
    header = logs[-1][1]
    missing = [k for k in ("n_train", "factor", "patience", "rtol", "cooldown") if header[k] is None]
    if missing:
        raise ValueError(f"No {', '.join(missing)} in {paths[-1]}: is it a reduce_on_plateau training log?")
    indices, logs, n_abandoned, dropped = _cut_abandoned(paths, logs)
    raw_epoch, time, train, val, raw_scale = np.concatenate([rows for rows, _ in logs]).T
    jobs = _job_logs([paths[k] for k in indices], logs, n_abandoned, indices, dropped)
    scale = np.concatenate([raw_scale[job.start : job.stop] * job.lr_ratio for job in jobs])
    name = str(paths[0]) if len(paths) == 1 else f"{paths[0]} (+{len(paths) - 1})"
    batch_size = int(header["batch_size"] or 1)
    steps_per_epoch = max(1, round(header["n_train"] / batch_size))
    return TrainingCurve(
        name=name,
        epoch=np.arange(1, len(train) + 1),
        time=time,
        train=train,
        val=val,
        scale=scale,
        raw_scale=raw_scale,
        raw_epoch=raw_epoch.astype(int),
        jobs=jobs,
        joins=np.array([job.start for job in jobs[1:]], dtype=int),
        n_train=int(header["n_train"]),
        batch_size=batch_size,
        factor=header["factor"],
        patience=int(header["patience"]),
        rtol=header["rtol"],
        cooldown=int(header["cooldown"]),
        accumulation_size=int(header["accumulation_size"] or steps_per_epoch),
        lr0=header["lr0"],
    )


def _member_labels(specs: list[list[str]]) -> list[str]:
    """Short unique names: the first path component that differs between all members (e.g.
    ``train_1`` ... ``train_5``), else the full path of each member's first log."""
    parts = [str(s[0]).split("/") for s in specs]
    for j in range(min(len(p) for p in parts)):
        col = [p[j] for p in parts]
        if len(set(col)) == len(col):
            return [c + (f" (+{len(s) - 1})" if len(s) > 1 else "") for c, s in zip(col, specs, strict=True)]
    return [str(s[0]) for s in specs]


def _reduction_indices(curve: TrainingCurve) -> np.ndarray:
    """Indices of the first epoch at each lower LR scale (normalized); a lower scale at the start of a
    new job is a change of the settings, not a reduction by the scheduler."""
    indices = np.nonzero(np.diff(curve.scale) < 0)[0] + 1
    return indices[~np.isin(indices, curve.joins)]


def _nanmin(x: np.ndarray) -> float:
    """Min ignoring nan; nan (no warning) when all are nan."""
    x = x[~np.isnan(x)]
    return float(x.min()) if len(x) else float("nan")


def _nanmedian(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    return float(np.median(x)) if len(x) else float("nan")


def lr_stages(curve: TrainingCurve) -> list[LRStage]:
    """Split the curve at every change of the normalized LR scale (reductions, or a new job started at
    another LR), with per-stage statistics on the val loss. A new job at the same LR stays in the stage
    (``jobs`` > 1), also when it restarts the optimizer with its initial LR set to the carried-over LR."""
    scale = curve.scale
    changes = np.nonzero(np.diff(scale) != 0)[0] + 1
    edges = sorted({0, len(scale)} | set(changes.tolist()))
    joins = set(curve.joins.tolist())
    stages, cumulative = [], np.inf
    for a, b in itertools.pairwise(edges):
        best = _nanmin(curve.val[a:b])  # a nan epoch (diverged) does not count
        previous, cumulative = cumulative, float(np.fmin(cumulative, best))
        tail = slice(max(a, b - 10), b)
        stages.append(
            LRStage(
                scale=float(scale[a]),
                start=int(curve.epoch[a]),
                end=int(curve.epoch[b - 1]),
                n_epochs=b - a,
                best_val=float(best),
                cumulative_best_val=float(cumulative),
                gain=float(previous / cumulative - 1) if np.isfinite(previous) else float("nan"),
                val_over_train=_nanmedian(curve.val[tail] / curve.train[tail]),
                running=b == len(scale),
                restart=a in joins,
                jobs=1 + sum(a < j < b for j in joins),
            )
        )
    return stages


def replay_reduce_on_plateau(loss, patience: int, cooldown: int, rtol: float, factor: float = 0.5,
                             atol: float = 0.0, min_scale: float = 1e-4) -> np.ndarray:
    """``optax.contrib.reduce_on_plateau`` on one already-accumulated value per check.

    Returns the LR scale in effect during each check, as the ``Scale`` printed on the log's line for
    the last epoch of that check: a reduction decided at the end of check i shows up there at check i + 1.
    """
    best, plateau, cool, scale = np.inf, 0, 0, 1.0
    out = np.empty(len(loss))
    for i, value in enumerate(loss):
        out[i] = scale
        improved = value < (1 - rtol) * best - atol
        if improved:
            best = value
        count = 0 if improved else plateau + 1
        if cool > 0:
            plateau, cool = 0, cool - 1
        elif count == patience:
            plateau, scale, cool = 0, max(scale * factor, min_scale), cooldown
        else:
            plateau = count
    return out


def _epochs_per_replay_check(curve: TrainingCurve) -> int:
    """Epochs per replayed check. The log has one loss per epoch, so a check shorter than an epoch,
    or not a whole number of epochs, can only be approximated."""
    epc = curve.epochs_per_check
    k = max(1, round(epc))
    if abs(epc - k) > 1e-9:
        warnings.warn(
            f"{curve.name}: accumulation size {curve.accumulation_size} is not a multiple of "
            f"{curve.steps_per_epoch} steps/epoch; approximating with one check per {k} epoch(s)",
            stacklevel=3,
        )
    return k


def _checks(curve: TrainingCurve, monitor: str, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Monitored values, one per replayed check, and the epoch each check ends on."""
    loss = curve.monitored(monitor)
    n = len(loss) // k
    return loss[: n * k].reshape(n, k).mean(1), curve.epoch[: n * k].reshape(n, k)[:, -1]


def _running_median(x: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    return np.array([np.median(x[max(0, i - half) : i + half + 1]) for i in range(len(x))])


def _first_reduction(scale: np.ndarray) -> int | None:
    indices = np.nonzero(scale < scale[0] * (1 - 1e-12))[0] if len(scale) else []
    return int(indices[0]) if len(indices) else None


def _job_curve(curve: TrainingCurve, job: JobLog) -> TrainingCurve:
    """One job of ``curve`` on its own: its raw (logged) scale and its own scheduler settings (falling
    back to the last log's when its header lacks one)."""
    a, b = job.start, job.stop
    settings = {key: value for key, value in job.settings.items() if value is not None}
    for key in ("n_train", "batch_size", "patience", "cooldown", "accumulation_size"):
        if key in settings:
            settings[key] = int(settings[key])
    if "accumulation_size" not in settings and "n_train" in settings:
        settings["accumulation_size"] = max(1, round(settings["n_train"] / settings.get("batch_size", 1)))
    return replace(curve, epoch=curve.epoch[a:b], time=curve.time[a:b], train=curve.train[a:b],
                   val=curve.val[a:b], scale=curve.raw_scale[a:b], raw_scale=curve.raw_scale[a:b],
                   raw_epoch=curve.raw_epoch[a:b],
                   joins=np.array([], dtype=int), jobs=[job], lr0=job.lr0, **settings)


def _replay_mismatch(curve: TrainingCurve, k: int, min_scale: float) -> dict[str, int | None]:
    """Per monitor, the first epoch where the replay differs from the logged scale (None: never).
    ``curve`` is one job, with its raw scale and ``min_scale`` in the same units. A job only starts a
    fresh scheduler with ``new_optimizer = true``: a job continuing the previous optimizer state is
    expected to mismatch."""
    mismatch = {}
    for monitor in _MONITORS:
        values, epochs = _checks(curve, monitor, k)
        # the logged scale changes on the last epoch of a check
        logged = curve.scale[: len(values) * k].reshape(-1, k)[:, -1]
        patience = max(1, round(curve.patience * curve.epochs_per_check / k))
        cooldown = round(curve.cooldown * curve.epochs_per_check / k)
        replayed = logged[0] * replay_reduce_on_plateau(
            values, patience, cooldown, curve.rtol, curve.factor, min_scale=min_scale / logged[0]
        )
        same = np.isclose(replayed, logged)
        mismatch[monitor] = None if same.all() else int(epochs[np.argmin(same)])
    return mismatch


def _convergence(curve, stages, conv_tol, conv_stages, done_scale, overfit):
    """(epoch since when the member is converged or None, reason or None, overfitting)."""
    done = [s for s in stages if not s.running and s.scale < stages[0].scale]
    flat_since = next(
        (done[j].end for j in range(conv_stages - 1, len(done))
         if all(s.gain < conv_tol for s in done[j - conv_stages + 1 : j + 1])),
        None,
    )
    low = np.nonzero(curve.scale <= done_scale)[0]
    low_since = int(curve.epoch[low[0]]) if len(low) else None
    since, reason = None, None
    if flat_since is not None and (low_since is None or flat_since <= low_since):
        since, reason = flat_since, "flat stages"
    elif low_since is not None:
        since, reason = low_since, f"scale <= {done_scale:g}"
    return since, reason, stages[-1].val_over_train > overfit * stages[0].val_over_train


def _post_reduction_transients(members: list[MemberAnalysis], monitor: str) -> list[int]:
    """Epochs for the monitored loss to complete 90% of its drop after each actual reduction."""
    transients = []
    for member in members:
        indices = _reduction_indices(member.curve)
        log_loss = np.log10(member.curve.monitored(monitor))
        for j, i0 in enumerate(indices):
            i1 = min(indices[j + 1] if j + 1 < len(indices) else len(log_loss), i0 + 150)
            if i1 - i0 < 20:
                continue
            smooth = _running_median(log_loss[i0:i1], 7)
            total = smooth[0] - smooth.min()
            if total > 0:
                transients.append(int(np.argmax(smooth[0] - smooth >= 0.9 * total)))
    return transients


def _median_first(epochs: list[int | None]) -> float:
    # a member that never reduces counts as "later than the end of its log"
    if all(e is None for e in epochs):
        return np.inf
    return float(np.median([np.inf if e is None else e for e in epochs]))


def analyze_training_logs(
    logs: Sequence,
    *,
    monitor: str = "val",
    rtols: Sequence[float] | None = None,
    patience_grid: Sequence[int] = tuple(range(5, 201, 5)),
    smooth: int = 21,
    window: int = 50,
    k: float = 2.0,
    min_scale: float = 1e-4,
    conv_tol: float = 0.02,
    conv_stages: int = 2,
    done_scale: float = 1e-3,
    overfit: float = 1.3,
    walltime_h: float | None = None,
) -> TrainingAnalysis:
    """Analyze the LR reductions and the convergence of an ensemble and suggest patience/cooldown.

    ``logs`` has one entry per member: the path of its log, or a sequence of paths for a member
    trained over several jobs. ``rtols`` are the values to scan (default: the one in the logs, plus
    0.01, 0.02, 0.05, 0.1). ``patience_grid`` is in epochs, ``smooth`` and ``window`` too; ``k`` sets
    the onset, at progress over ``window`` < ``k`` x noise. ``min_scale`` is
    ``min_learning_rate_scale`` (not in the logs) relative to the first log's initial LR, only used to
    replay and to estimate the length of the whole training; ``done_scale`` is relative to it too.
    ``walltime_h`` adds that length in jobs to the report. See the module docstring for the method;
    ``format_report`` turns the result into text.
    """
    if monitor not in _MONITORS:
        raise ValueError(f"monitor must be one of {_MONITORS}, got {monitor!r}")
    if not logs:
        raise ValueError("No logs given")
    specs = [[p] if isinstance(p, str | Path) else list(p) for p in logs]
    curves = [read_training_curve(s) for s in specs]
    reference = curves[0]
    patience_grid = sorted(patience_grid)
    rtols = sorted(set(rtols) if rtols is not None else {reference.rtol, 0.01, 0.02, 0.05, 0.1})

    members, too_short = [], True
    for curve, label in zip(curves, _member_labels(specs), strict=True):
        checks_k = _epochs_per_replay_check(curve)
        values, epochs = _checks(curve, monitor, checks_k)
        cut = _first_reduction(curve.scale)
        n_const = (cut if cut is not None else len(curve.train)) // checks_k
        member_short = n_const < 2 * max(window, smooth) // checks_k
        too_short &= member_short

        # a diverged (nan/inf) check is left out of the noise and onset, which a single one would void;
        # the replays keep it, as the scheduler does (it is no improvement)
        finite = np.isfinite(values[:n_const]) & (values[:n_const] > 0)
        log_values, finite_epochs = np.log10(values[:n_const][finite]), epochs[:n_const][finite]
        n_finite = len(log_values)
        noise, onset = float("nan"), None
        if n_finite >= 2:
            smoothed = _running_median(log_values, max(1, smooth // checks_k))
            noise = float(np.std((log_values - smoothed)[n_finite // 2 :]))
            w = max(1, window // checks_k)
            progress = np.full(n_finite, np.inf)
            progress[w:] = smoothed[:-w] - smoothed[w:]
            burn = max(w, smooth // checks_k)  # skip the steep initial transient and the edge
            below = np.nonzero((progress < k * noise) & (np.arange(n_finite) >= burn))[0]
            onset = int(finite_epochs[below[0]]) if len(below) else None

        patience = max(1, round(curve.patience * curve.epochs_per_check / checks_k))
        cooldown = round(curve.cooldown * curve.epochs_per_check / checks_k)
        replayed = _first_reduction(
            replay_reduce_on_plateau(values, patience, cooldown, curve.rtol, curve.factor, min_scale=min_scale)
        )

        # per job on the raw scale: the scheduler of a job counts from that job's own initial LR
        replay_mismatch = {}
        for job_log in curve.jobs:
            if _first_reduction(curve.raw_scale[job_log.start : job_log.stop]) is None:
                continue
            job = _job_curve(curve, job_log)
            job_k = checks_k if job.epochs_per_check == curve.epochs_per_check else _epochs_per_replay_check(job)
            replay_mismatch[f"job {job_log.number}"] = _replay_mismatch(job, job_k, min_scale / job_log.lr_ratio)

        stages = lr_stages(curve)
        since, reason, overfitting = _convergence(curve, stages, conv_tol, conv_stages, done_scale, overfit)
        members.append(
            MemberAnalysis(
                curve=curve,
                label=label,
                stages=stages,
                reductions=[int(curve.epoch[i]) for i in _reduction_indices(curve)],
                replay_mismatch=replay_mismatch,
                noise=noise,
                onset=onset,
                replayed_first_reduction=None if replayed is None else int(epochs[replayed]),
                converged_since=since,
                converged_reason=reason,
                overfitting=overfitting,
                check_loss=values[:n_const],
                check_epoch=epochs[:n_const],
                epochs_per_replay_check=checks_k,
                segment_too_short=member_short,
            )
        )

    # the pooled scan judges the members whose initial-LR segment is long enough (at least one, else
    # segments_too_short); a short one (e.g. stopped early by another member's STOP_ROUND) would never
    # reduce within its segment and, with an even count, push the median to "never"
    scanned = [m for m in members if not m.segment_too_short] or members
    onsets = [m.onset for m in members if m.onset is not None]
    target = float(np.median(onsets)) if onsets else None
    first: dict[tuple[float, int], list[int | None]] = {}
    for rtol in rtols:
        for patience in patience_grid:
            first[rtol, patience] = []
            for m in scanned:
                checks = max(1, patience // m.epochs_per_replay_check)
                f = _first_reduction(replay_reduce_on_plateau(m.check_loss, checks, 0, rtol, reference.factor))
                first[rtol, patience].append(None if f is None else int(m.check_epoch[f]))

    n_reductions = int(np.ceil(np.log(min_scale) / np.log(reference.factor) - 1e-9))
    transients = _post_reduction_transients(members, monitor)
    median_noise = float(np.nanmedian([m.noise for m in members])) if onsets else float("nan")
    suggestions: list[PatienceSuggestion] = []
    for rtol in rtols if target is not None and not too_short else []:
        ok = [p for p in patience_grid if _median_first(first[rtol, p]) >= target]
        if not ok:
            suggestions.append(PatienceSuggestion(
                rtol, note=f"even patience {patience_grid[0]} reduces before the onset; use a smaller rtol"))
            continue
        patience = ok[0]
        first_epoch = _median_first(first[rtol, patience])
        if np.isinf(first_epoch):
            note = (
                f"never reduces within the logs: rtol too large compared to the noise ({10**median_noise - 1:.1%})"
                if patience == patience_grid[0]
                else f"patience {patience} never reduces within the logs, smaller ones reduce before the onset"
            )
            suggestions.append(PatienceSuggestion(rtol, note=note))
            continue
        cooldown = int(np.median(transients)) if transients else patience // 2
        reached = [e for e in first[rtol, patience] if e is not None]
        # patience and cooldown above are in epochs, train.toml wants scheduler checks
        epc = reference.epochs_per_check
        suggestions.append(
            PatienceSuggestion(
                rtol=rtol,
                patience=max(1, round(patience / epc)),
                cooldown=round(cooldown / epc),
                cooldown_measured=bool(transients),
                first_reduction=first_epoch,
                first_reduction_range=(min(reached), max(reached)),
                n_never=len(scanned) - len(reached),
                min_total_epochs=first_epoch + (n_reductions - 1) * (patience + cooldown),
            )
        )
    usable = [s for s in suggestions if s.patience is not None]
    suggested = next((s for s in usable if s.rtol == reference.rtol), usable[0] if usable else None)

    return TrainingAnalysis(
        members=members,
        monitor=monitor,
        onset=target,
        segments_too_short=too_short,
        patience_grid=list(patience_grid),
        replayed_first_reductions={
            rtol: {p: _median_first(first[rtol, p]) for p in patience_grid} for rtol in rtols
        },
        suggestions=suggestions,
        suggested=suggested,
        n_reductions_to_end=n_reductions,
        median_epoch_time=float(np.median(np.concatenate([c.time for c in curves]))),
        walltime_h=walltime_h,
        conv_tol=conv_tol,
        conv_stages=conv_stages,
        done_scale=done_scale,
        min_scale=min_scale,
    )


def _format_mismatch(mismatch: dict[str, dict[str, int | None]]) -> list[str]:
    lines = []
    for job, by_monitor in mismatch.items():
        match = [m for m, epoch in by_monitor.items() if epoch is None]
        if match:
            lines.append(f"  {job}: replay reproduces the logged scale with monitor {', '.join(match)}")
        else:
            agree = "; ".join(f"{m} until epoch {epoch - 1}" for m, epoch in by_monitor.items())
            lines.append(f"  {job}: replay reproduces the logged scale with no monitor (agrees: {agree})")
    return lines


def _format_jobs(curve: TrainingCurve) -> list[str]:
    """One line per continued job, when the chain restarts the optimizer or changes the initial LR:
    what the normalized scales of the report mean."""
    if not any(job.new_optimizer or job.lr_ratio != 1.0 or job.n_abandoned for job in curve.jobs[1:]):
        return []
    lines = []
    for previous, job in itertools.pairwise(curve.jobs):
        how = "new optimizer" if job.new_optimizer else "optimizer kept"
        if job.new_optimizer is None:
            how = "no continuation line"
        tail = job.n_abandoned - sum(n for _, n in job.dropped)
        cut = "".join(f", job {k + 1} ({n} epochs) left out: dead branch" for k, n in job.dropped)
        cut += f", last {tail} epochs of job {previous.number} left out (dead branch)" if tail else ""
        lines.append(f"  job {job.number} from epoch {job.start + 1}: {how}"
                     f"{f', loads {job.load_model_type}' if job.load_model_type else ''}{cut}, "
                     f"lr0 {job.lr0} = {job.lr_ratio:.4g} x first lr0 (logged scale x {job.lr_ratio:.4g})")
    return lines


def _format_lr0(curve: TrainingCurve) -> str:
    """The first job's lr0, which the scales of the report are relative to, and the last one's if it differs."""
    first = curve.jobs[0].lr0 if curve.jobs else curve.lr0
    last = f", {curve.lr0} (last job)" if curve.lr0 != first else ""
    return f"lr0 {first} (first job){last}"


def format_report(analysis: TrainingAnalysis) -> str:
    """Human-readable text of an ``analyze_training_logs`` result. Scales are relative to each member's
    first initial LR (as logged unless a job changed it, which the report lists)."""
    members = analysis.members
    r0 = members[0].curve
    lines = [
        (f"{len(members)} members | train {r0.n_train} structures, batch {r0.batch_size} -> "
         f"{r0.steps_per_epoch} steps/epoch | current: patience {r0.patience}, cooldown {r0.cooldown}, "
         f"rtol {r0.rtol}, factor {r0.factor}, accumulation {r0.accumulation_size}, {_format_lr0(r0)}"),
        f"monitored loss: {analysis.monitor}",
        "",
        "---- LR reductions ----",
        (f"{'member':14s} {'epochs':>6s} {'s/ep':>6s} {'best val':>9s} {'noise':>6s} {'onset':>5s} "
         f"{'reductions':>10s} {'1st actual':>10s} {'1st replay':>10s}"),
    ]
    for m in members:
        c = m.curve
        replayed = "none" if m.replayed_first_reduction is None else m.replayed_first_reduction
        lines.append(
            f"{m.label:14s} {len(c.train):6d} {np.median(c.time):6.0f} {_nanmin(c.val):9.2e} "
            f"{10**m.noise - 1:6.1%} {m.onset!s:>5s} {len(m.reductions):10d} "
            f"{m.reductions[0] if m.reductions else 'none':>10} {replayed:>10}"
        )
        lines.extend(f"{m.label}{line}" for line in _format_jobs(c))
        if m.replay_mismatch:
            lines.extend(f"{m.label}{line}" for line in _format_mismatch(m.replay_mismatch))

    lines += [
        "",
        (f"---- convergence (best val loss per LR stage; converged when the last {analysis.conv_stages} completed "
         f"reduced-LR stages gained < {analysis.conv_tol:.0%}, or scale <= {analysis.done_scale:g}) ----"),
    ]
    for m in members:
        lines.append(m.label)
        lines.append(
            f"  {'scale':>9s} {'epochs':>11s} {'n':>4s} {'best val':>9s} {'cum. best':>9s} "
            f"{'gain':>7s} {'val/train':>9s}"
        )
        for s in m.stages:
            gain = "" if np.isnan(s.gain) else f"{s.gain:.1%}"
            lines.append(
                f"  {s.scale:9.2e} {s.start:5d}-{s.end:<5d} {s.n_epochs:4d} {s.best_val:9.2e} "
                f"{s.cumulative_best_val:9.2e} {gain:>7s} {s.val_over_train:9.2f}"
                f"{'  (running)' if s.running else ''}{'  (restart)' if s.restart else ''}"
                f"{f'  ({s.jobs} jobs)' if s.jobs > 1 else ''}"
            )
        if m.converged:
            verdict = (f"converged since epoch {m.converged_since} ({m.converged_reason}; "
                       f"best val {m.stages[-1].cumulative_best_val:.2e})")
        elif not m.reductions:
            verdict = "not converged (no LR reduction yet)"
        else:
            verdict = f"not converged ({len(m.reductions)} LR reductions)"
        if m.overfitting:
            verdict += (f" | overfitting? val/train {m.stages[0].val_over_train:.2f} -> "
                        f"{m.stages[-1].val_over_train:.2f}")
        lines.append(f"  => {verdict}")
    reductions = [len(m.reductions) for m in members]
    lines.append("")
    lines.append(f"ensemble: reductions per member {reductions}, current scale "
                 f"{', '.join(f'{m.curve.scale[-1]:.1e}' for m in members)} (of each member's first lr0)")
    if max(reductions) - min(reductions) > 1 and not analysis.all_converged:
        lines.append("  note: members are at different LR stages (normal while they run, e.g. started at different "
                     "times); stop them all at a common stage, otherwise the ensemble spread mixes training stages")
    if analysis.all_converged:
        lines.append("  => all members converged: training can be stopped (best-val checkpoints are kept)")
    else:
        lines.append(f"  => still training: {', '.join(m.label for m in members if not m.converged)}")

    lines += ["", "---- patience and cooldown ----",
              f"noise-dominated onset (median over members): {analysis.onset}"]
    if analysis.segments_too_short:
        lines.append("  -> the initial-LR segments are too short (e.g. a continued run that reduced the LR early): "
                     "patience scan not meaningful, use logs of runs started from scratch")
    elif analysis.onset is None:
        lines.append("  -> no member reached the noise-dominated regime: the loss is still clearly improving, "
                     "keep the LR (or train longer before tuning)")
    else:
        rtols = list(analysis.replayed_first_reductions)
        lines.append("")
        lines.append("replayed first reduction epoch (median over members; '-' = never within the logs)")
        lines.append(f"{'pat. [ep]':>9s} " + " ".join(f"{'rtol ' + str(rt):>11s}" for rt in rtols))
        for p in analysis.patience_grid:
            cells = [analysis.replayed_first_reductions[rt][p] for rt in rtols]
            lines.append(f"{p:9d} " + " ".join(f"{'-' if np.isinf(c) else f'{c:.0f}':>11s}" for c in cells))
            if all(np.isinf(cells)):
                lines.append(f"{'...':>9s} (never for larger patience)")
                break
        lines.append("")
        for s in analysis.suggestions:
            if s.patience is None:
                lines.append(f"rtol {s.rtol}: {s.note}")
                continue
            source = "measured transient" if s.cooldown_measured else "patience // 2, no reductions to measure"
            total = f">= {s.min_total_epochs:.0f} epochs"
            if analysis.walltime_h is not None:
                per_job = analysis.walltime_h * 3600 / analysis.median_epoch_time
                total += f" (~{s.min_total_epochs / per_job:.1f} jobs of {analysis.walltime_h:g} h)"
            lines.append(
                f"rtol {s.rtol}: patience {s.patience}, cooldown {s.cooldown} ({source}) | first reduction at epoch "
                f"~{s.first_reduction:.0f} (members: {s.first_reduction_range[0]}-{s.first_reduction_range[1]}"
                f"{f', {s.n_never} never' if s.n_never else ''}) | {total} for {analysis.n_reductions_to_end} "
                f"reductions down to scale {analysis.min_scale:g}"
            )
    best = analysis.suggested
    if best is not None:
        lines += [
            "",
            (f"=> suggested (rtol {best.rtol}): patience = {best.patience}, cooldown = {best.cooldown}  "
             f"[in scheduler checks = {r0.accumulation_size} optimizer steps each]"),
            ("   user guide reference (for typical datasets): patience 120 epochs well converged, <= 60 quick; "
             "cooldown 20-50 epochs"),
        ]
    return "\n".join(lines)
