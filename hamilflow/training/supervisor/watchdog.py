"""Run one ``deeph-train`` process and supervise it: the loop inside a member's train job.

``run_supervised`` starts the trainer in its own session, finds the run dir it creates, and every
``cfg.poll_s`` checks, in order: the control files of the round (``STOP_ROUND``, ``STOP_{fold}``),
the walltime (the internal timer is the only protection: no signal reaches the job's Python under
jobflow-remote), and ``policy.live_check`` on the member's chained logs. To stop, it signals the
trainer's process group only (SIGINT, then SIGTERM, then SIGKILL), so the job itself always ends
normally with an ``ExitInfo``. Trainer failures never raise: they are ``ExitInfo(reason="crashed")``.

Light on purpose: a poll reads the new part of the current log, stats a few checkpoint dirs (a
handful of entries each), and runs ``live_check`` on the member's logs (small text files).
"""
import dataclasses
import datetime
import os
import re
import signal
import socket
import subprocess
import time
import tomllib
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

from .policy import ExitInfo, LiveCheck, MemberStatus, SupervisorConfig, live_check, resolve_min_scale

_RUN_NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
_EPOCH_NUMBER_RE = re.compile(r"Epoch (\d+) \| Time ([\d.]+) s")
_CHECKPOINT_RE = re.compile(r"^epoch_(\d+)$")  # not the orbax tmp dirs epoch_N.orbax-checkpoint-tmp-K
_METADATA = "_CHECKPOINT_METADATA"  # orbax's marker of a finished checkpoint
# environment of the hamilflow venv that must not leak into the deeph venv
_DROPPED_ENV = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
# live status -> the policy key that decides whether the watchdog stops on it
_STATUS_POLICY = {
    MemberStatus.CONVERGED: "stop_converged",
    MemberStatus.OVERFIT: "stop_overfit",
    MemberStatus.DIVERGED: "divergence",
    MemberStatus.MISTUNED: "retune",
}
_MAX_SETTLE_WAIT = 60.0  # [s] the longest the stop waits for a checkpoint being written (or 10 x the settle time)


# ---------------------------------- helpers reused by step 5 and the CLI ----------------------------------


def find_new_run_dir(outputs_dir, existing: set[str] | frozenset[str] = frozenset()) -> Path | None:
    """The newest ``yyyy-mm-dd_hh-mm-ss`` folder of ``outputs_dir`` whose name is not in ``existing``
    (the run dirs there before the launch); None if there is none (yet)."""
    outputs_dir = Path(outputs_dir)
    if not outputs_dir.is_dir():
        return None
    runs = sorted(p.name for p in outputs_dir.iterdir()
                  if _RUN_NAME_RE.match(p.name) and p.name not in existing and p.is_dir())
    return outputs_dir / runs[-1] if runs else None


def run_dir_names(outputs_dir) -> set[str]:
    outputs_dir = Path(outputs_dir)
    return {p.name for p in outputs_dir.iterdir() if _RUN_NAME_RE.match(p.name)} if outputs_dir.is_dir() else set()


def _checkpoint_epochs(pytree_dir: Path) -> dict[int, Path]:
    if not pytree_dir.is_dir():
        return {}
    return {int(m.group(1)): p for p in pytree_dir.iterdir() if (m := _CHECKPOINT_RE.match(p.name))}


def _mtime(path: Path) -> float:
    """Latest modification time of ``path`` and of its direct entries (a checkpoint dir holds a few)."""
    times = [path.stat().st_mtime]
    if path.is_dir():
        times += [entry.stat().st_mtime for entry in os.scandir(path)]
    return max(times)


def last_checkpoint_epoch(run_dir, settle_s: float = 5.0, now: float | None = None,
                          kind: str = "latest", parts: Sequence[str] = ("params", "states")) -> int | None:
    """The highest epoch N with ``model/{part}/{kind}.pytree/epoch_N`` complete for every part of
    ``parts`` (DeepH user guide, 9.1): orbax writes a ``epoch_N.orbax-checkpoint-tmp-K`` dir and renames
    it to ``epoch_N`` with its ``_CHECKPOINT_METADATA`` once done (orbax, DeepH-pack 1.0.6), so the name and that file
    mark a complete checkpoint; and, as an extra guard, none modified in the last ``settle_s`` seconds
    before ``now``. None if there is none. A resume needs params and states (the default); inference
    loads params only (``parts=("params",)``): after a SIGINT the two can differ."""
    if run_dir is None:
        return None
    model = Path(run_dir) / "model"
    found = [_checkpoint_epochs(model / part / f"{kind}.pytree") for part in parts]
    now = time.time() if now is None else now
    for epoch in sorted(set.intersection(*(set(f) for f in found)), reverse=True):
        dirs = [f[epoch] for f in found]
        if not all((d / _METADATA).is_file() for d in dirs):
            continue
        try:
            if max(_mtime(d) for d in dirs) <= now - settle_s:
                return epoch
        except FileNotFoundError:  # removed meanwhile (DeepH keeps only latest_num of them)
            continue
    return None


def newest_checkpoint_epoch(run_dir, kind: str = "latest") -> int | None:
    """The highest ``epoch_N`` of ``{kind}.pytree`` in params or states, complete or not: what
    DeepH's ``load_model_epoch = -1`` could pick."""
    if run_dir is None:
        return None
    model = Path(run_dir) / "model"
    epochs = set(_checkpoint_epochs(model / "params" / f"{kind}.pytree"))
    epochs |= set(_checkpoint_epochs(model / "states" / f"{kind}.pytree"))
    return max(epochs) if epochs else None


def _newest_checkpoint_mtime(run_dir) -> float | None:
    """Latest modification among the checkpoint dirs of a run (a handful of entries)."""
    if run_dir is None:
        return None
    times = []
    for part in ("params", "states"):
        for kind in ("latest", "best"):
            pytree = Path(run_dir) / "model" / part / f"{kind}.pytree"
            try:
                times += [_mtime(p) for p in _checkpoint_epochs(pytree).values()]
                if pytree.is_dir():
                    times.append(pytree.stat().st_mtime)
            except FileNotFoundError:
                continue
    return max(times) if times else None


def _squeue_end_time(job_id: str) -> float | None:
    """One ``squeue -h -j <id> -o %e`` call (on the compute node, once per job): the end time, local."""
    try:
        out = subprocess.run(["squeue", "-h", "-j", job_id, "-o", "%e"], capture_output=True, text=True,
                             timeout=30, check=True).stdout.strip()
        return datetime.datetime.strptime(out, "%Y-%m-%dT%H:%M:%S").timestamp()
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def job_end_time(cfg: SupervisorConfig, start: float, env: Mapping[str, str] | None = None,
                 squeue: Callable[[str], float | None] = _squeue_end_time) -> tuple[float | None, str]:
    """(end time of the job, its source), in this order: ``SLURM_JOB_END_TIME``, a single
    ``squeue`` call, ``start + cfg.walltime``; (None, "none") without any (no walltime limit)."""
    env = os.environ if env is None else env
    value = env.get("SLURM_JOB_END_TIME")
    if value:
        try:
            return float(value), "SLURM_JOB_END_TIME"
        except ValueError:
            pass
    if env.get("SLURM_JOB_ID"):
        end = squeue(env["SLURM_JOB_ID"])
        if end is not None:
            return end, "squeue"
    if cfg.walltime is not None:
        return start + cfg.walltime, "cfg.walltime"
    return None, "none"


def clean_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """The trainer's environment: this one without the hamilflow venv's Python variables."""
    env = dict(os.environ if env is None else env)
    for key in _DROPPED_ENV:
        env.pop(key, None)
    return env


# ---------------------------------- the current log ----------------------------------


def _gpu_description(env: dict | None = None) -> str:
    """The GPU(s) this job got, for the journal (a mixed partition can hand out an 11 GB card): one
    ``nvidia-smi`` query, bounded, never raising."""
    visible = (env if env is not None else os.environ).get("CUDA_VISIBLE_DEVICES", "unset")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        out = "nvidia-smi not available"
    return f"{'; '.join(out.splitlines()) or 'none'} (CUDA_VISIBLE_DEVICES={visible})"


class _LogTail:
    """Reads the current deepx.log incrementally: the epochs seen and their times."""

    def __init__(self, path: Path):
        self.path, self.offset, self.partial = path, 0, ""
        self.epochs: list[int] = []
        self.times: list[float] = []

    def update(self) -> None:
        try:
            if self.path.stat().st_size < self.offset:  # truncated or replaced: read it again from the start
                self.offset, self.partial, self.epochs, self.times = 0, "", [], []
            with open(self.path, encoding="utf-8", errors="replace") as f:
                f.seek(self.offset)
                text = f.read()
                self.offset = f.tell()
        except FileNotFoundError:
            return
        lines = (self.partial + text).split("\n")
        self.partial = lines.pop()
        for line in lines:
            if m := _EPOCH_NUMBER_RE.search(line):
                self.epochs.append(int(m.group(1)))
                self.times.append(float(m.group(2)))

    @property
    def last_epoch(self) -> int | None:
        return self.epochs[-1] if self.epochs else None

    @property
    def epoch_time(self) -> float | None:
        """Median epoch time, without the run's first epoch (JIT compilation: ~120 s against ~25 s on the
        cluster, also on a resumed run). None until two later epochs are logged: an estimate from the
        first one alone made the walltime window ~5x too wide."""
        times = self.times[1:]
        return float(np.median(times)) if len(times) >= 2 else None


# ---------------------------------- the watchdog ----------------------------------


class _Journal:
    """``round_dir/watchdog_{fold}.log``: a compact trace of what the watchdog saw and did."""

    def __init__(self, path: Path, now: Callable[[], float]):
        self.path, self.now = path, now
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        stamp = datetime.datetime.fromtimestamp(self.now()).strftime("%Y-%m-%d %H:%M:%S")
        with open(self.path, "a") as f:
            f.write(f"[{stamp}] {message}\n")


def _outputs_dir(train_toml: Path, cwd: Path) -> Path:
    with open(train_toml, "rb") as f:
        outputs = Path(tomllib.load(f)["data"]["outputs_dir"])
    return outputs if outputs.is_absolute() else cwd / outputs


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _exited(proc: subprocess.Popen) -> bool:
    """Whether the trainer exited, without reaping it: while it is an unreaped zombie its pid, hence its
    process group id, cannot be reused, so the group can still be signalled safely."""
    if proc.returncode is not None:
        return True
    try:
        return os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
    except ChildProcessError:
        return True


def _wait_exit(proc: subprocess.Popen, timeout: float | None) -> bool:
    """Wait for the trainer to exit (not reaped), up to ``timeout`` seconds (None: no limit)."""
    end = None if timeout is None else time.monotonic() + timeout
    while not _exited(proc):
        if end is not None and time.monotonic() >= end:
            return False
        time.sleep(0.02)
    return True


def _group_survivors(pgid: int, leader: int) -> list[int]:
    """Live processes of process group ``pgid`` other than its (exited, unreaped) leader, from /proc
    (read once, at exit). Empty where /proc is not available."""
    survivors = []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return []
    for entry in entries:
        if not entry.isdigit() or int(entry) == leader:
            continue
        try:
            state, _ppid, pgrp = Path(f"/proc/{entry}/stat").read_text().rsplit(")", 1)[1].split()[:3]
        except (OSError, ValueError):
            continue
        if int(pgrp) == pgid and state != "Z":
            survivors.append(int(entry))
    return survivors


def _reap(proc: subprocess.Popen, journal) -> tuple[int, list[int]]:
    """Kill what is left of the trainer's process group (e.g. the trainer a wrapper left running), then
    reap the trainer. Its pid is still held, so the group id cannot be someone else's. Returns the exit
    code and the pids that were still alive in the group."""
    survivors = _group_survivors(proc.pid, proc.pid)
    if survivors:
        journal(f"processes {survivors} of the trainer's group were still running: killed")
    _signal_group(proc, signal.SIGKILL)
    return proc.wait(), survivors


def _stop(proc: subprocess.Popen, run_dir, cfg: SupervisorConfig, journal, now, sleep) -> None:
    """Stop the trainer's process group: wait (bounded) while a checkpoint is being written, then
    SIGINT, SIGTERM after ``stop_grace``, SIGKILL after ``kill_grace``. The trainer is not reaped here."""
    began, bound = now(), min(_MAX_SETTLE_WAIT, 10 * cfg.checkpoint_settle_s)
    while (mtime := _newest_checkpoint_mtime(run_dir)) is not None and now() - mtime < cfg.checkpoint_settle_s:
        if now() - began >= bound or _exited(proc):
            break
        sleep(max(0.05, min(cfg.checkpoint_settle_s, 1.0)))
    if (waited := now() - began) > 0.5:
        journal(f"waited {waited:.1f} s for a checkpoint being written")
    for sig, grace in ((signal.SIGINT, cfg.stop_grace), (signal.SIGTERM, cfg.kill_grace), (signal.SIGKILL, None)):
        if _exited(proc):
            return
        journal(f"sending {sig.name}")
        _signal_group(proc, sig)
        if _wait_exit(proc, grace):
            return


def _write_alert(round_dir: Path, fold: int, check: LiveCheck, journal) -> None:
    alerts = round_dir / "alerts"
    alerts.mkdir(parents=True, exist_ok=True)
    (alerts / f"member_{fold}.txt").write_text(
        f"status: {check.status.value}\nreason: {check.reason}\n"
        f"written: {datetime.datetime.now().isoformat(timespec='seconds')}\n"
        f"The policy for this is 'ask': training goes on. Touch {round_dir / f'STOP_{fold}'} (this member) or "
        f"{round_dir / 'STOP_ROUND'} (all members) to stop; the supervise step then decides.\n"
    )
    journal(f"alert: {check.status.value}: {check.reason}")


def run_supervised(
    train_toml,
    cwd,
    member_logs_before: Sequence,
    cfg: SupervisorConfig,
    round_dir,
    fold: int,
    shared_overrides: Mapping | None = None,
    *,
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    end_time: float | None | str = "auto",
    env: Mapping[str, str] | None = None,
    live_actions: Mapping[str, str] | None = None,
) -> ExitInfo:
    """Run ``cfg.train_command`` on ``train_toml`` from ``cwd`` (the fold dir) and supervise it until it
    exits or is stopped; return how it ended.

    ``member_logs_before``: the member's logs of the earlier rounds, oldest first (``live_check`` sees
    them chained with the current log). ``round_dir``: the round's folder, with the control files
    ``STOP_ROUND`` / ``STOP_{fold}``, the alerts and ``watchdog_{fold}.log``. ``end_time``: "auto" (looked up by
    ``job_end_time``), or a time stamp, or None (no walltime limit). ``live_actions``: this
    member's overrides of ``cfg.actions`` for the live rules only (``MemberAction.live_actions``, e.g.
    ``{"retune": "off"}`` when the pool cannot act on this member's W3). ``now``, ``sleep``, ``env``:
    injectable for tests.

    ``STOP_{fold}`` stops at once. ``STOP_ROUND`` written by another member or the user stops at once
    if this run has a trusted ``latest`` checkpoint; otherwise it waits for the first one, at most
    ``2 x latest_interval`` epoch times (``round_stop_max_wait`` before the first epoch), so the
    member is not sent back to its round's start. The walltime and hard stops come first.

    ``ExitInfo.reason``: ``walltime``, ``stopped`` (a control file), ``live_rule`` (``live_status`` /
    ``live_reason`` say which), whatever the exit code once the watchdog signalled; otherwise
    ``finished`` (exit 0 with a run dir) or ``crashed`` (another exit code, or no run dir). A stop
    before the run dir appeared keeps its reason, with ``run_dir`` None.
    """
    train_toml, cwd, round_dir = Path(train_toml).resolve(), Path(cwd), Path(round_dir)
    cfg = resolve_min_scale(cfg, shared_overrides)
    if live_actions:
        cfg = dataclasses.replace(cfg, actions={**cfg.actions, **live_actions})
    journal = _Journal(round_dir / f"watchdog_{fold}.log", now)
    outputs_dir = _outputs_dir(train_toml, cwd)
    existing = run_dir_names(outputs_dir)
    start = now()
    if end_time == "auto":
        end, source = job_end_time(cfg, start, env)
    else:
        end, source = end_time, "given" if end_time is not None else "none"
    if end is None:
        journal("no walltime limit known (no SLURM end time, no cfg.walltime): the watchdog will not stop for it")
    else:
        journal(f"job end time from {source}: in {end - start:.0f} s, stop {cfg.walltime_margin:g} s before")
        if source in ("SLURM_JOB_END_TIME", "squeue") and cfg.walltime is not None \
                and abs(end - (start + cfg.walltime)) > 600:
            journal(f"warning: the SLURM end time differs from start + cfg.walltime ({cfg.walltime:g} s) by "
                    f"{end - start - cfg.walltime:+.0f} s; using SLURM's")
    deadline = None if end is None else end - cfg.walltime_margin
    if live_actions:
        journal(f"live rules overridden for this member: {dict(live_actions)}")
    if deadline is not None and now() >= deadline:
        journal("the walltime margin is already reached: not starting the trainer")
        return ExitInfo("walltime", return_code=None)

    command = [part.replace("{toml}", str(train_toml)) for part in cfg.train_command]
    out_path = cwd / f"{train_toml.stem}.out"
    journal(f"node {socket.gethostname()}, GPU: {_gpu_description(env)}")
    journal(f"start: {command} in {cwd}, output in {out_path.name}")
    with open(out_path, "ab") as out:
        try:
            proc = subprocess.Popen(command, cwd=cwd, env=clean_env(env), stdout=out, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        except OSError as error:
            journal(f"could not start the trainer: {error}")
            return ExitInfo("crashed", return_code=None)
    try:
        return _supervise(proc, cfg, journal, outputs_dir, existing, deadline, list(member_logs_before),
                          round_dir, fold, shared_overrides, now, sleep)
    finally:
        if proc.returncode is None:  # only on an error in the watchdog itself: never leave the trainer behind
            journal("watchdog error: killing the trainer")
            _reap(proc, journal)


def _supervise(proc, cfg, journal, outputs_dir, existing, deadline, logs_before, round_dir, fold,
               shared_overrides, now, sleep) -> ExitInfo:
    run_dir, tail = None, None
    stop_reason, live = None, None
    alerted = None  # the status of the latest alert, to write it once per change
    window_checkpoint = "unset"  # latest checkpoint epoch when the walltime window was entered
    wrote_round_stop = False
    round_stop_since = None  # when another member's (or the user's) STOP_ROUND was first seen
    while True:
        if run_dir is None and (run_dir := find_new_run_dir(outputs_dir, existing)) is not None:
            tail = _LogTail(run_dir / "deepx.log")
            journal(f"run dir: {run_dir}")
        if _exited(proc):
            break
        if tail is not None:
            tail.update()
        t = now()
        wait_until = None  # a bound the next sleep must not overshoot

        # (a) control files: STOP_{fold} at once; another's STOP_ROUND once this run has a latest checkpoint
        if (round_dir / f"STOP_{fold}").exists():
            stop_reason = "stopped"
            journal(f"control file STOP_{fold}: stopping")
        elif (round_dir / "STOP_ROUND").exists() and not wrote_round_stop:
            if last_checkpoint_epoch(run_dir, cfg.checkpoint_settle_s, t) is not None:
                stop_reason = "stopped"
                journal("control file STOP_ROUND: stopping")
            else:
                epoch_time = None if tail is None else tail.epoch_time
                bound = cfg.round_stop_max_wait if epoch_time is None else min(
                    cfg.round_stop_max_wait, 2 * cfg.latest_interval * epoch_time)
                if round_stop_since is None:
                    round_stop_since = t
                    journal(f"control file STOP_ROUND: no latest checkpoint yet, training on until the first one "
                            f"(at most {bound:.0f} s)")
                if t - round_stop_since >= bound:
                    stop_reason = "stopped"
                    journal("control file STOP_ROUND: no latest checkpoint within the wait, stopping anyway")
                else:
                    wait_until = round_stop_since + bound
        # (b) walltime
        if stop_reason is None and deadline is not None:
            if t >= deadline:
                stop_reason = "walltime"
                journal("walltime margin reached: stopping")
            else:
                wait_until = deadline if wait_until is None else min(wait_until, deadline)
                if tail is not None and tail.epoch_time is not None and run_dir is not None:
                    window = cfg.latest_interval * tail.epoch_time
                    if t < deadline - window:
                        # not (or no longer, with a better epoch time) in the window: a checkpoint written
                        # before entering it must not count as "the next one"
                        window_checkpoint = "unset"
                    else:
                        checkpoint = last_checkpoint_epoch(run_dir, cfg.checkpoint_settle_s, t)
                        if window_checkpoint == "unset":
                            window_checkpoint = checkpoint
                            journal(f"within {window:.1f} s of the walltime deadline: stopping after the next "
                                    f"latest checkpoint (now {checkpoint})")
                        elif checkpoint is not None and (window_checkpoint is None or checkpoint > window_checkpoint):
                            stop_reason = "walltime"
                            journal(f"stopping right after the latest checkpoint of epoch {checkpoint}, "
                                    f"{deadline - t:.1f} s before the deadline")
        # (c) live rules (a hard stop also ends a STOP_ROUND wait)
        if stop_reason is None and tail is not None and tail.last_epoch is not None:
            try:
                check = live_check([*logs_before, run_dir / "deepx.log"], cfg, shared_overrides)
            except Exception:  # noqa: BLE001 - nothing in a log may kill the watchdog: journal it, train on
                journal("live_check failed:\n" + traceback.format_exc().rstrip())
                check = LiveCheck(MemberStatus.RUNNING, "")
            if check.hard_stop:
                stop_reason, live = "live_rule", check
                journal(f"hard stop: {check.status.value}: {check.reason}")
            elif check.status in _STATUS_POLICY:
                mode = cfg.actions[_STATUS_POLICY[check.status]]
                if mode == "auto":
                    stop_reason, live = "live_rule", check
                    journal(f"live rule ({_STATUS_POLICY[check.status]} auto): {check.status.value}: {check.reason}")
                    if check.status == MemberStatus.MISTUNED:
                        (round_dir / "STOP_ROUND").write_text(f"member {fold}: {check.reason}\n")
                        wrote_round_stop = True
                        journal("wrote STOP_ROUND: the round is re-decided on the pooled analysis")
                elif mode == "ask" and alerted != check.status:
                    _write_alert(round_dir, fold, check, journal)
                    alerted = check.status
            elif check.status == MemberStatus.RUNNING:
                alerted = None

        if stop_reason is not None:
            _stop(proc, run_dir, cfg, journal, now, sleep)
            break
        pause = cfg.poll_s if wait_until is None else min(cfg.poll_s, max(0.0, wait_until - now()))
        sleep(pause)

    _wait_exit(proc, None)
    exit_time = now()
    return_code, survivors = _reap(proc, journal)
    if run_dir is None:
        run_dir = find_new_run_dir(outputs_dir, existing)
    if run_dir is not None:
        tail = tail or _LogTail(run_dir / "deepx.log")
        tail.update()
    if stop_reason is not None:
        reason = stop_reason
    elif return_code == 0 and run_dir is not None and not survivors:
        reason = "finished"
    else:
        reason = "crashed"
        if return_code == 0 and survivors:
            journal("the command exited while the trainer it started was still running (killed): a crash")
    info = ExitInfo(
        reason=reason,
        return_code=return_code,
        run_dir=None if run_dir is None else str(run_dir),
        last_epoch=None if tail is None else tail.last_epoch,
        last_checkpoint_epoch=last_checkpoint_epoch(run_dir, cfg.checkpoint_settle_s, exit_time),
        live_status=None if live is None else live.status.value,
        live_reason=None if live is None else live.reason,
        newest_checkpoint_epoch=newest_checkpoint_epoch(run_dir),
    )
    if reason == "crashed" and run_dir is None:
        journal("the trainer exited before creating a run dir")
    journal(f"end: {reason}, exit code {return_code}, last epoch {info.last_epoch}, "
            f"last checkpoint {info.last_checkpoint_epoch} (newest on disk {info.newest_checkpoint_epoch})")
    return info
