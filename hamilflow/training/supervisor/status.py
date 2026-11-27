"""The compact view of a cycle, for ``hamilflow-train-supervisor status``.

A header (round N of M, the cycle's state), one row per member (what it is doing now, its run, the last
epoch's numbers, seconds per epoch, the time left before the watchdog's walltime stop, and the decision
that put it in this round), flags (alerts, STOP/FINISH files, a pending proposal), then commands for what
can be done next. Cheap, for the login node: it reads state.json, the ends of the current logs and the
watchdogs' journals (``round_N/watchdog_J.log``), never a whole log.
"""
import datetime
import re
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ..ensemble import fold_dir, train_outputs_dir
from . import jobs
from .watchdog import find_new_run_dir

_EPOCH_LINE = re.compile(r"Epoch (\d+) \| Time ([\d.]+) s \| Train-Loss (\S+) \| Val-Loss (\S+) \| Scale (\S+)")
_LR0 = re.compile(r"initial learning rate `([\d.eE+-]+)`")
_STAMP = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (.*)$")
_END_TIME = re.compile(r"job end time from \S+: in (\d+) s, stop ([\d.]+) s before")
_END = re.compile(r"^end: (\w+)")


def _read_ends(path: Path, size: int = 16384) -> tuple[str, str]:
    with open(path, "rb") as f:
        head = f.read(size).decode(errors="replace")
        f.seek(max(0, path.stat().st_size - size))
        tail = f.read().decode(errors="replace")
    return head, tail


def _log_numbers(log: Path) -> dict:
    """The last epoch's numbers and the pace, from the ends of ``log`` only."""
    if not log.is_file():
        return {}
    head, tail = _read_ends(log)
    lines = _EPOCH_LINE.findall(tail)
    if not lines:
        return {}
    epoch, _t, _train, val, scale = lines[-1]
    times = [float(t) for _, t, *_ in lines[1:]] or [float(lines[-1][1])]  # without a JIT epoch if possible
    lr0 = _LR0.search(head)
    return {"epoch": int(epoch), "val": float(val), "scale": float(scale),
            "lr": float(lr0.group(1)) * float(scale) if lr0 else None,
            "s_per_epoch": float(np.median(times[-10:]))}


def _lineage_epochs(logs: list, current: Path | None) -> int | None:
    """Epochs of the member's lineage up to now: its earlier runs chained with the current one, without
    their dead branches (DeepH restarts its count at 1 after a new-optimizer restart, e.g. a retune)."""
    import warnings

    from ..log_analysis import has_epochs, read_training_curve

    chain = [p for p in logs if Path(p).is_file() and has_epochs(p)]
    if current is not None and current.is_file() and has_epochs(current) and str(current) not in map(str, chain):
        chain.append(current)
    if not chain:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the analysis warns about inferred cuts; not status's business
            return len(read_training_curve(chain).val)
    except (ValueError, OSError):
        return None


def _journal(path: Path) -> dict:
    """From a watchdog journal: when the walltime stop is due, how the run ended, the last alert."""
    if not path.is_file():
        return {}
    info = {"started": True}
    for line in path.read_text(errors="replace").splitlines():
        match = _STAMP.match(line)
        if not match:
            continue
        stamp = datetime.datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
        text = match.group(2)
        if m := _END_TIME.search(text):
            info["stop_at"] = stamp + float(m.group(1)) - float(m.group(2))
        elif m := _END.match(text):
            info["ended"] = m.group(1)
        elif text.startswith("alert: "):
            info["alert"] = text[len("alert: "):]
        elif text.startswith(("control file", "stopping right after", "within ")):
            info["stopping"] = text
    return info


_CAUSE_LABELS = {"retune": "ensemble suggestion", "user": "set by the user", "stopped": "STOP file",
                 "crash": "crash", "earlier": "earlier", "removed_by_user": "removed by the user",
                 "finished": "ended without converging"}


def cause_label(record: Mapping) -> str:
    """The cause of a decision in words, for display (the stored cause names are unchanged): a retune the
    member's own live rule asked for says which way its patience was off; one that only follows the
    pooled suggestion says so."""
    cause = record.get("cause") or ""
    if cause == "mistuned":
        reason = record.get("reason") or ""
        if "patience too large" in reason:
            return "own patience too large"
        if "patience too small" in reason:
            return "own patience too small"
        return "own scheduler mistuned"
    return _CAUSE_LABELS.get(cause, cause)


def _last_record(state: Mapping, fold: str) -> dict | None:
    """The member's last real decision: not the "earlier" placeholders of the rounds after it stopped."""
    for r in reversed(state["history"]["rounds"]):
        record = r["members"].get(fold)
        if record is not None and record.get("cause") != "earlier":
            return record
    return None


_ENDED = {"done": "done", "give_up": "given up", "finished_by_user": "finished by user"}


def duration(seconds: float) -> str:
    """``48 min``, ``1 h 48 min`` (whole minutes, no decimal hours); ``<1 min``; ``now`` once past."""
    if seconds <= 0:
        return "now"
    minutes = round(seconds / 60)
    if minutes < 1:
        return "<1 min"
    hours, minutes = divmod(minutes, 60)
    return f"{minutes} min" if hours == 0 else f"{hours} h {minutes} min"


def build_status(cycle_dir, state: Mapping, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    cycle_dir = Path(cycle_dir)
    n = state["round"]
    here = jobs.round_dir(cycle_dir, n)
    proposal = None
    if state["status"] == "awaiting_approval" and (here / "proposal.toml").is_file():
        proposal = jobs.read_proposal(here / "proposal.toml")
    members = []
    for fold in sorted(state["members"], key=int):
        j = int(fold)
        record = _last_record(state, fold)
        trains_now = (fold_dir(cycle_dir, j) / jobs.train_toml_name(n)).is_file() and state["status"] == "running"
        run = None
        if trains_now:
            run = find_new_run_dir(train_outputs_dir(cycle_dir, j, round=n))
        if run is None:
            recorded = [r["run_dir"] for r in state["members"][fold]["runs"] if r["run_dir"]]
            run = Path(recorded[-1]) if recorded else None
        numbers = {"epoch": None, "val": None, "scale": None, "lr": None, "s_per_epoch": None,
                   "lineage_epoch": None}
        if run is not None:
            numbers.update(_log_numbers(Path(run) / "deepx.log"))
            if numbers["epoch"] is not None:
                numbers["lineage_epoch"] = _lineage_epochs(state["members"][fold]["logs"], Path(run) / "deepx.log")
        journal = _journal(here / f"watchdog_{fold}.log") if trains_now else {}
        if trains_now:
            if not journal:
                doing = "queued"
            elif "ended" in journal:
                doing = f"stopped ({journal['ended']})"
            elif numbers["epoch"] is None:
                doing = "starting"
            else:
                doing = "stopping" if "stopping" in journal else "training"
        elif proposal is not None:
            doing = "waiting for approval"
        elif record and state["status"] == "finished" and state.get("finish_requested") \
                and record["action"] in ("resume", "retune", "restart_lower_lr"):
            doing = "finished by user"
            record = {**record, "action": "finished", "cause": "user"}
        else:
            doing = _ENDED.get(record["action"], "not training") if record else "-"
        stop_in = None
        if trains_now and "stop_at" in journal and "ended" not in journal:
            stop_in = journal["stop_at"] - now
        decided = None if record is None else f"{record['action']} ({cause_label(record)})"
        proposed = None
        if proposal is not None:
            prop = proposal["members"].get(j)
            proposed = "not in proposal" if prop is None else (
                f"{prop['action']}{' *' if prop.get('needs_approval') else ''}")
        members.append({
            "fold": j, "doing": doing, "run_round": jobs_round_of(run), "decided": decided, "proposed": proposed,
            "stop_in_s": stop_in, "alert": journal.get("alert") or _alert_file(here, fold),
            "stop_file": state["status"] == "running" and (here / f"STOP_{fold}").exists(), **numbers,
        })
    return {
        "name": cycle_dir.resolve().name, "cycle_dir": str(cycle_dir), "round": n,
        "max_rounds": state["cfg"].get("max_rounds"), "status": state["status"],
        "finish_requested": state["status"] != "finished" and (bool(state.get("finish_requested"))
                                                             or jobs.finish_path(cycle_dir).exists()),
        "finish_file": jobs.finish_path(cycle_dir).exists() and state["status"] == "running",
        "stop_round": state["status"] == "running" and (here / "STOP_ROUND").exists(),
        "proposal": proposal is not None, "members": members,
    }


def jobs_round_of(run) -> int | None:
    match = re.search(r"outputs_r(\d+)", str(run or ""))
    return int(match.group(1)) if match else None


def _alert_file(here: Path, fold: str) -> str | None:
    path = here / "alerts" / f"member_{fold}.txt"
    if not path.is_file():
        return None
    lines = dict(line.split(": ", 1) for line in path.read_text().splitlines()[:2] if ": " in line)
    return f"{lines.get('status', '')}: {lines.get('reason', '')}".strip(": ")


def format_status(view: Mapping, cycle: str = "<cycle>", jobs_lines: list[str] | None = None) -> str:
    """``jobs_lines``: the ``--jobs`` block, printed before the next steps (which then skip ``--jobs``)."""
    of = f" of {view['max_rounds']}" if view.get("max_rounds") else ""
    state = view["status"].replace("_", " ")
    finish = " (finish requested: ends after this round)" if view["finish_requested"] else ""
    out = [f"{view['name']}: round {view['round']}{of}, {state}{finish}", ""]
    awaiting = view["proposal"]
    last_col = "proposed" if awaiting else "decision"
    out.append(f"  {'member':6s}  {'now':20s}  {'run':>3s}  {'epoch (lineage)':>15s}  {'val':>8s}  {'scale':>6s}  {'lr':>7s}  "
               f"{'s/ep':>4s}  {'stop in':>11s}  {last_col}")
    for m in view["members"]:
        def num(key, fmt, m=m):
            return "-" if m.get(key) is None else format(m[key], fmt)
        stop_in = "-" if m["stop_in_s"] is None else duration(m["stop_in_s"])
        run = "-" if m["run_round"] is None else f"r{m['run_round']}"
        last = (m["proposed"] if awaiting else m["decided"]) or "-"
        epoch = num("epoch", "d")
        if m.get("lineage_epoch") is not None and m["lineage_epoch"] != m.get("epoch"):
            epoch = f"{epoch} ({m['lineage_epoch']})"
        out.append(f"  {m['fold']:<6d}  {m['doing']:20s}  {run:>3s}  {epoch:>15s}  {num('val', '.2e'):>8s}  "
                   f"{num('scale', '.3g'):>6s}  {num('lr', '.1e'):>7s}  {num('s_per_epoch', '.0f'):>4s}  "
                   f"{stop_in:>11s}  {last}")
    flags = []
    for m in view["members"]:
        if m["alert"]:
            flags.append(f"! member {m['fold']}: {m['alert']}")
        if m["stop_file"]:
            flags.append(f"STOP_{m['fold']} is set: member {m['fold']} stops at its next poll")
    if view["stop_round"]:
        flags.append("STOP_ROUND is set: the round stops (members without a latest checkpoint first get one)")
    if view.get("finish_file"):
        flags.append("FINISH is set: the round's supervise job ends the cycle")
    if flags:
        out += [""] + [f"  {f}" for f in flags]
    steps = []
    if awaiting:
        steps += [("see the proposal", "review <cycle>"), ("approve it", "approve <cycle> [--set J.key=value]")]
    elif view["status"] == "running":
        steps += [("stop the round", "stop <cycle>"), ("stop one member", "stop <cycle> --member J"),
                  ("end after this round", "finish <cycle>")]
        if jobs_lines is None:
            steps.append(("job states", "status <cycle> --jobs"))
    elif view["status"] == "finished":
        steps += [("final models", "summary <cycle> [--infer-tomls overrides.json]")]
    if jobs_lines:
        out += [""] + jobs_lines
    if steps:
        width = max(len(label) for label, _ in steps) + 1
        out += ["", "next:"] + [f"  {label + ':':<{width + 1}}hamilflow-train-supervisor {cmd.replace('<cycle>', cycle)}"
                                for label, cmd in steps]
    return "\n".join(out)


# ---------------------------------- summary ----------------------------------

_VERDICTS = {
    ("done", "converged"): "converged",
    ("done", "overfit"): "stopped: overfitting",
    ("done", "user"): "done (set by the user)",
    ("done", "removed_by_user"): "done (removed from the proposal)",
    ("give_up", "max_rounds"): "max rounds reached",
    ("give_up", "crash"): "gave up after crashes",
    ("give_up", "diverged"): "gave up: diverged",
    ("give_up", "finished"): "ended without converging",
}
_SETTING_LABELS = (("init_learning_rate", "lr"), ("patience", "patience"), ("cooldown", "cooldown"),
                   ("rtol", "rtol"), ("min_learning_rate_scale", "min scale"))


def verdict(member: Mapping) -> str:
    if member["action"] == "finished_by_user":
        return "finished by the user"
    return _VERDICTS.get((member["action"], member["cause"]), f"{member['action']} ({member['cause']})")


def _settings_line(settings: list[dict]) -> str:
    """``r1 start: lr 0.001, patience 20, … · r2 retune: patience 15, cooldown 7 · r3 resume``: the first
    round in full, then only what changed."""
    parts, before = [], {}  # [first round, last round, action, changes]
    for entry in settings:
        changed = [f"{label} {_fmt(entry[key])}" for key, label in _SETTING_LABELS
                   if key in entry and before.get(key) != entry[key]]
        if not changed and parts and not parts[-1][3] and parts[-1][2] == entry["action"]:
            parts[-1][1] = entry["round"]  # r2-r6 resume: a run of rounds that changed nothing
        else:
            parts.append([entry["round"], entry["round"], entry["action"], changed])
        before = entry
    return " · ".join(f"r{a}{f'–r{b}' if b != a else ''} {action}" + (f": {', '.join(ch)}" if ch else "")
                      for a, b, action, ch in parts)


def _fmt(value) -> str:
    return f"{value:.3g}" if isinstance(value, float) else str(value)


def build_summary_view(cycle_dir, state: Mapping, final: Mapping) -> dict:
    """The summary (``jobs.summary``) plus, per member, the epochs and best val of its lineage."""
    from ..log_analysis import has_epochs, read_training_curve

    cycle_dir = Path(cycle_dir)
    members = []
    for fold, member in sorted(final["members"].items(), key=lambda kv: int(kv[0])):
        logs = [p for p in state["members"][fold]["logs"] if Path(p).is_file() and has_epochs(p)]
        epochs = best = None
        if logs:
            try:
                curve = read_training_curve(logs)
                epochs, best = len(curve.val), float(np.nanmin(curve.val))
            except (ValueError, OSError):
                pass
        run = member["final_run_dir"]
        shown = run
        if run is not None:
            try:
                shown = str(Path(run).relative_to(state["cycle_dir"]))
            except ValueError:
                pass
        members.append({"fold": int(fold), "verdict": verdict(member), "reason": member["reason"],
                        "rounds": len(member["settings"]), "epochs": epochs, "best_val": best,
                        "final_model": shown, "settings": _settings_line(member["settings"])})
    return {"name": cycle_dir.resolve().name, "status": final["status"], "round": final["round"],
            "max_rounds": state["cfg"].get("max_rounds"), "members": members}


def format_summary(view: Mapping, cycle: str = "<cycle>") -> str:
    of = f" of {view['max_rounds']}" if view.get("max_rounds") else ""
    if view["status"] == "finished":
        head = f"{view['name']}: finished after round {view['round']}{of}"
    else:
        head = f"{view['name']}: round {view['round']}{of}, {view['status'].replace('_', ' ')} (verdicts so far)"
    out = [head, ""]
    width = max([len(m["verdict"]) for m in view["members"]] + [7])
    out.append(f"  {'member':6s}  {'verdict':{width}s}  {'rounds':>6s}  {'epochs':>6s}  {'best val':>8s}  final model")
    for m in view["members"]:
        epochs = "-" if m["epochs"] is None else str(m["epochs"])
        best = "-" if m["best_val"] is None else f"{m['best_val']:.2e}"
        model = "none (no complete best checkpoint)" if m["final_model"] is None else f"{m['final_model']} (best)"
        out.append(f"  {m['fold']:<6d}  {m['verdict']:{width}s}  {m['rounds']:>6d}  {epochs:>6s}  {best:>8s}  {model}")
    out += ["", "settings per round (changes only after the first):"]
    out += [f"  {m['fold']}: {m['settings']}" for m in view["members"]]
    steps = [("infer TOMLs", "summary <cycle> --infer-tomls overrides.json"), ("as JSON", "summary <cycle> --json")]
    width = max(len(label) for label, _ in steps) + 1
    out += ["", "next:"] + [f"  {label + ':':<{width + 1}}hamilflow-train-supervisor {cmd.replace('<cycle>', cycle)}"
                            for label, cmd in steps]
    out.append(f"  {'next cycle:':<{width + 1}}make(new_cycle, continue_from=warm_start_from(\"{cycle}\"))  (python)")
    return "\n".join(out)
