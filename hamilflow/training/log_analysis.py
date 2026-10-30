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
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

_EPOCH_RE = re.compile(
    r"Epoch (\d+) \| Time ([\d.]+) s \| Train-Loss ([\d.eE+-]+) \| Val-Loss ([\d.eE+-]+) \| Scale ([\d.eE+-]+)"
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
_MONITORS = ("train", "val", "mean")


@dataclass
class TrainingCurve:
    """One ensemble member, possibly over several jobs; the settings are those of the last log."""

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
    """A stretch of epochs at one LR scale; a new job also starts a new stage."""

    scale: float
    start: int
    end: int
    n_epochs: int
    best_val: float
    cumulative_best_val: float
    gain: float  # relative improvement over the best of the previous stages; nan on the first one
    val_over_train: float  # median over the last 10 epochs of the stage
    running: bool
    restart: bool


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
    header = {}
    for key, pattern in _HEADER_RE.items():
        match = re.search(pattern, text)
        header[key] = float(match.group(1)) if match else None
    return rows, header


def read_training_curve(paths) -> TrainingCurve:
    """Read one member from its log, or from the logs of its consecutive jobs (a sequence of paths,
    in order, for ``[process.train.continued]``); epochs are renumbered consecutively."""
    paths = [paths] if isinstance(paths, str | Path) else list(paths)
    logs = [_read_log(p) for p in paths]
    header = logs[-1][1]
    missing = [k for k in ("n_train", "factor", "patience", "rtol", "cooldown") if header[k] is None]
    if missing:
        raise ValueError(f"No {', '.join(missing)} in {paths[-1]}: is it a reduce_on_plateau training log?")
    _, time, train, val, scale = np.concatenate([rows for rows, _ in logs]).T
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
        joins=np.cumsum([len(rows) for rows, _ in logs])[:-1],
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
    """Indices of the first epoch at each lower LR scale; a new job's scale is not a reduction."""
    indices = np.nonzero(np.diff(curve.scale) < 0)[0] + 1
    return indices[~np.isin(indices, curve.joins)]


def lr_stages(curve: TrainingCurve) -> list[LRStage]:
    """Split the curve at every change of the LR scale (reductions, or a reset when continuing with
    ``new_optimizer = true``) and at every new job, with per-stage statistics on the val loss."""
    scale = curve.scale
    changes = np.nonzero(np.diff(scale) != 0)[0] + 1
    edges = sorted({0, len(scale)} | set(changes.tolist()) | set(curve.joins.tolist()))
    stages, cumulative = [], np.inf
    for a, b in itertools.pairwise(edges):
        best = curve.val[a:b].min()
        previous, cumulative = cumulative, min(cumulative, best)
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
                val_over_train=float(np.median(curve.val[tail] / curve.train[tail])),
                running=b == len(scale),
                restart=a in curve.joins,
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


def _replay_mismatch(curve: TrainingCurve, k: int, min_scale: float) -> dict[str, int | None]:
    """Per monitor, the first epoch where the replay differs from the logged scale (None: never).
    A job only starts a fresh scheduler with ``new_optimizer = true``: a job continuing the previous
    optimizer state is expected to mismatch."""
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
    ``min_learning_rate_scale`` (not in the logs), only used to replay and to estimate the length of
    the whole training. ``walltime_h`` adds that length in jobs to the report. See the module
    docstring for the method; ``format_report`` turns the result into text.
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
        too_short &= n_const < 2 * max(window, smooth) // checks_k

        log_values = np.log10(values[:n_const])
        noise, onset = float("nan"), None
        if n_const >= 2:
            smoothed = _running_median(log_values, max(1, smooth // checks_k))
            noise = float(np.std((log_values - smoothed)[n_const // 2 :]))
            w = max(1, window // checks_k)
            progress = np.full(n_const, np.inf)
            progress[w:] = smoothed[:-w] - smoothed[w:]
            burn = max(w, smooth // checks_k)  # skip the steep initial transient and the edge
            below = np.nonzero((progress < k * noise) & (np.arange(n_const) >= burn))[0]
            onset = int(epochs[below[0]]) if len(below) else None

        patience = max(1, round(curve.patience * curve.epochs_per_check / checks_k))
        cooldown = round(curve.cooldown * curve.epochs_per_check / checks_k)
        replayed = _first_reduction(
            replay_reduce_on_plateau(values, patience, cooldown, curve.rtol, curve.factor, min_scale=min_scale)
        )

        replay_mismatch = {}
        bounds = [0, *curve.joins.tolist(), len(curve.scale)]
        for j, (a, b) in enumerate(itertools.pairwise(bounds)):
            job_scale = curve.scale[a:b]
            if _first_reduction(job_scale) is None:
                continue
            job = replace(curve, epoch=curve.epoch[a:b], time=curve.time[a:b], train=curve.train[a:b],
                          val=curve.val[a:b], scale=job_scale, joins=np.array([], dtype=int))
            replay_mismatch[f"job {j + 1}"] = _replay_mismatch(job, checks_k, min_scale)

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
            )
        )

    onsets = [m.onset for m in members if m.onset is not None]
    target = float(np.median(onsets)) if onsets else None
    first: dict[tuple[float, int], list[int | None]] = {}
    for rtol in rtols:
        for patience in patience_grid:
            first[rtol, patience] = []
            for m in members:
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
                n_never=len(members) - len(reached),
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


def format_report(analysis: TrainingAnalysis) -> str:
    """Human-readable text of an ``analyze_training_logs`` result."""
    members = analysis.members
    r0 = members[0].curve
    lines = [
        (f"{len(members)} members | train {r0.n_train} structures, batch {r0.batch_size} -> "
         f"{r0.steps_per_epoch} steps/epoch | current: patience {r0.patience}, cooldown {r0.cooldown}, "
         f"rtol {r0.rtol}, factor {r0.factor}, accumulation {r0.accumulation_size}, lr0 {r0.lr0}"),
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
            f"{m.label:14s} {len(c.train):6d} {np.median(c.time):6.0f} {c.val.min():9.2e} "
            f"{10**m.noise - 1:6.1%} {m.onset!s:>5s} {len(m.reductions):10d} "
            f"{m.reductions[0] if m.reductions else 'none':>10} {replayed:>10}"
        )
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
                 f"{', '.join(f'{m.curve.scale[-1]:.1e}' for m in members)}")
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
