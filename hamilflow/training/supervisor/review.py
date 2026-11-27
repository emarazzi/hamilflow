"""The compact view of a pending proposal, for ``hamilflow-train-supervisor review``.

Per member: the proposed action (``*`` = needs your approval), a one-line reason and the key numbers; then
only what changes for its next run (where it starts from, the optimizer, the settings that differ from its
current run); then the evidence, only for the members that need a decision; then ready-to-copy commands.
``review --full`` still prints the raw proposal and the whole analysis report.

Light, as everything the CLI runs on the login node: it parses the members' logs once (about a second for
5 members x 5 jobs x 850 epochs) and reads state.json and the proposal.
"""
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from ..ensemble import deep_merge
from ..log_analysis import analyze_training_logs, has_epochs
from . import jobs
from .policy import SupervisorConfig, resolve_min_scale
from .status import cause_label

TRAINS = ("resume", "retune", "restart_lower_lr")
_BARS = "▁▂▃▄▅▆▇█"
_ROUND_RE = re.compile(r"outputs_r(\d+)")
_SETTING_NAMES = {"init_learning_rate": "lr", "patience": "patience", "cooldown": "cooldown", "rtol": "rtol"}


def sparkline(values: Sequence[float], width: int = 24) -> str:
    """The val loss in ``width`` characters (the minimum of each bin, log scale): high bars = high loss."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) < 2:
        return ""
    bins = np.array_split(np.log10(v), min(width, len(v)))
    low = np.array([b.min() for b in bins])
    span = low.max() - low.min()
    if span <= 0:
        return _BARS[0] * len(low)
    return "".join(_BARS[round((x - low.min()) / span * (len(_BARS) - 1))] for x in low)


def _round_of(run_dir) -> int | None:
    match = _ROUND_RE.search(str(run_dir or ""))
    return int(match.group(1)) if match else None


def _exit_of(member: Mapping, run_dir) -> dict:
    for run in member.get("runs", []):
        if run.get("run_dir") == run_dir:
            return run.get("exit") or {}
    return {}


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def _source(prop: Mapping, member_state: Mapping) -> str:
    """Where the next run starts: ``from latest of r5 (epoch 120), optimizer kept``."""
    cont = prop.get("continue_from")
    continued = prop.get("fold_overrides", {}).get("process", {}).get("train", {}).get("continued", {})
    if not cont:
        return "from scratch"
    kind = cont.get("load_model_type", "latest")
    where = f"{kind} of r{_round_of(cont.get('previous_output_dir'))}" if _round_of(
        cont.get("previous_output_dir")) else f"{kind} of {cont.get('previous_output_dir')}"
    epoch = continued.get("load_model_epoch")
    if epoch is None and kind == "latest":
        epoch = _exit_of(member_state, cont.get("previous_output_dir")).get("last_checkpoint_epoch")
    pinned = " (pinned)" if continued.get("load_model_epoch") is not None else ""
    epoch_text = f" (epoch {epoch}{pinned})" if epoch else ""
    optimizer = "NEW optimizer" if continued.get("new_optimizer") else "optimizer kept"
    return f"from {where}{epoch_text}, {optimizer}"


def short_reason(prop: Mapping) -> str:
    """One line for the table: for a retune, why in words (its full reason lists every value, shown in the
    change line anyway), else the reason up to its first ';'."""
    if prop.get("action") == "retune" and prop.get("cause") == "mistuned":
        return f"{cause_label(prop)} (its live rule): retune to the pooled suggestion"
    if prop.get("action") == "retune" and prop.get("cause") == "retune":
        return "the ensemble suggestion differs from its current patience/cooldown: retune"
    return (prop.get("reason") or "").split(";")[0]


def _changes(current: Mapping, proposed: Mapping) -> list[str]:
    out = []
    for key, name in _SETTING_NAMES.items():
        if key in proposed and key in current and not np.isclose(float(current[key]), float(proposed[key])):
            out.append(f"{name} {_fmt(current[key])} → {_fmt(proposed[key])}")
    return out


def _current_settings(curve) -> dict:
    return {"init_learning_rate": curve.lr0, "patience": curve.patience, "cooldown": curve.cooldown,
            "rtol": curve.rtol} if curve is not None and curve.lr0 is not None else {}


def _evidence(prop: Mapping, member, analysis) -> list[str]:
    """The numbers behind a decision that needs approval."""
    lines = []
    cause = prop.get("cause")
    if member is None:
        return [prop.get("reason", "")]
    if cause == "overfit" or prop.get("status") == "overfit":
        lines.append("best val and val/train per LR stage:")
        for s in member.stages[-5:]:
            gain = "" if np.isnan(s.gain) else f"  gain {s.gain:6.1%}"
            lines.append(f"  scale {s.scale:8.3g}  epochs {s.start:4d}-{s.end:<4d}  best val {s.best_val:.3e}"
                         f"  val/train {s.val_over_train:.2f}{gain}{'  (running)' if s.running else ''}")
    if cause in ("mistuned", "retune"):
        suggestion = analysis.suggested if analysis is not None else None
        onset = "none" if member.onset is None else f"epoch {member.onset}"
        median = "" if analysis is None or analysis.onset is None else f" (ensemble median {analysis.onset:.0f})"
        lines.append(f"noise-dominated onset {onset}{median}; noise {10 ** member.noise - 1:.1%} of the loss"
                     if np.isfinite(member.noise) else f"noise-dominated onset {onset}{median}")
        lines.append(f"current patience {member.curve.patience}: first reduction "
                     f"{'none yet' if not member.reductions else f'at epoch {member.reductions[0]}'}")
        if suggestion is not None and suggestion.patience is not None:
            lines.append(f"suggested (rtol {suggestion.rtol}): patience {suggestion.patience}, cooldown "
                         f"{suggestion.cooldown} → first reduction ~epoch {suggestion.first_reduction:.0f}")
    if cause == "diverged":
        val = member.curve.val
        finite = val[np.isfinite(val)]
        last = "nan/inf" if not np.isfinite(val[-1]) else f"{val[-1]:.3e}"
        if len(finite):
            lines.append(f"last val {last} against best {finite.min():.3e}")
    if cause in ("finished", "converged"):
        for s in member.stages[-3:]:
            gain = "" if np.isnan(s.gain) else f"  gain {s.gain:.1%}"
            lines.append(f"  scale {s.scale:8.3g}  best val {s.best_val:.3e}{gain}")
    return lines


def build_review(cycle_dir, state: Mapping, proposal: Mapping) -> dict:
    """JSON-able: per member the proposed action, its numbers, what changes and the evidence."""
    cfg = resolve_min_scale(SupervisorConfig.from_dict(state["cfg"]), state["base_overrides"])
    base = deep_merge(state["base_overrides"], proposal.get("shared_overrides", {}))
    folds = sorted(int(j) for j in state["members"])
    logs = {j: [p for p in state["members"][str(j)]["logs"] if Path(p).is_file() and has_epochs(p)] for j in folds}
    with_logs = [j for j in folds if logs[j]]
    analysis = analyze_training_logs([logs[j] for j in with_logs], **cfg.analysis_kwargs()) if with_logs else None
    analyzed = dict(zip(with_logs, analysis.members, strict=True)) if analysis else {}

    members = []
    for j in folds:
        prop = proposal["members"].get(j)
        member = analyzed.get(j)
        curve = member.curve if member is not None else None
        view = {"fold": j, "action": "not in the proposal (done)" if prop is None else prop["action"],
                "needs_approval": bool(prop and prop.get("needs_approval")),
                "cause": None if prop is None else prop.get("cause"),
                "reason": "removed from the proposal" if prop is None else short_reason(prop),
                "epochs": None, "best_val": None, "scale": None, "reductions": None, "val_over_train": None,
                "spark": "", "start": None, "changes": [], "evidence": []}
        if member is not None:
            view.update({"epochs": len(curve.val), "best_val": float(np.nanmin(curve.val)),
                         "scale": float(curve.scale[-1]), "reductions": len(member.reductions),
                         "val_over_train": float(member.stages[-1].val_over_train), "spark": sparkline(curve.val)})
        if prop is not None and prop["action"] in TRAINS:
            view["start"] = _source(prop, state["members"][str(j)])
            proposed = jobs.round_settings(base, prop.get("fold_overrides"))
            view["changes"] = _changes(_current_settings(curve), proposed)
            if prop["action"] != "resume":  # a new optimizer: show the LR it starts at and its floor
                if not any(c.startswith("lr ") for c in view["changes"]):
                    view["changes"].append(f"lr {_fmt(proposed.get('init_learning_rate'))} (carried over)")
                view["changes"].append(f"min scale {_fmt(proposed.get('min_learning_rate_scale'))}")
        if prop is not None and prop.get("needs_approval"):
            view["evidence"] = _evidence(prop, member, analysis)
        members.append(view)
    return {"cycle_dir": str(cycle_dir), "after_round": proposal.get("after_round"), "round": proposal.get("round"),
            "members": members}


def _next_steps(view: Mapping) -> list[str]:
    steps = [("approve as is", "approve <cycle>")]
    for m in view["members"]:
        if not m["needs_approval"]:
            continue
        j = m["fold"]
        if m["action"] in ("done", "give_up"):
            steps.append((f"keep {j} training", f"approve <cycle> --set {j}.action=resume"))
        elif m["action"] in ("retune", "restart_lower_lr"):
            steps.append((f"other values for {j}", f"approve <cycle> --set {j}.patience=15 --set {j}.cooldown=7"))
            steps.append((f"no change for {j}", f"approve <cycle> --set {j}.action=resume"))
    steps += [("edit everything", "approve <cycle> --edit"), ("full details", "review <cycle> --full")]
    width = max(len(label) for label, _ in steps) + 1
    return [f"{label + ':':<{width + 1}}hamilflow-train-supervisor {command}" for label, command in steps]


def format_review(view: Mapping, cycle: str = "<cycle>") -> str:
    members = view["members"]
    n_ask = sum(m["needs_approval"] for m in members)
    title = (f"round {view['after_round']} → proposal for round {view['round']}: {len(members)} members, "
             f"{n_ask} need{'s' if n_ask == 1 else ''} approval (*)")
    out = [title, ""]
    out.append(f"  {'member':6s}  {'action':16s}  {'epochs':>6s}  {'best val':>8s}  {'scale':>6s}  {'red.':>4s}  "
               f"{'val/tr':>6s}  val loss")
    for m in members:
        def num(key, fmt, m=m):
            return "-" if m[key] is None else format(m[key], fmt)
        out.append(f"  {str(m['fold']) + (' *' if m['needs_approval'] else ''):6s}  {m['action']:16s}  "
                   f"{num('epochs', 'd'):>6s}  {num('best_val', '.2e'):>8s}  {num('scale', '.3g'):>6s}  "
                   f"{num('reductions', 'd'):>4s}  {num('val_over_train', '.2f'):>6s}  {m['spark']}")
    for m in members:
        out += ["", f"member {m['fold']}{' *' if m['needs_approval'] else ''}: {m['action']} — {m['reason']}"]
        if m["start"]:
            out.append(f"  starts {m['start']}")
            out.append(f"  {'; '.join(m['changes']) if m['changes'] else 'settings unchanged'}")
        elif m["action"] in ("done", "give_up"):
            out.append("  training ends; its final model is the best-val checkpoint of its lineage")
        for line in m["evidence"]:
            out.append(f"  | {line}")
    out += ["", "next:"] + [f"  {line.replace('<cycle>', cycle)}" for line in _next_steps(view)]
    return "\n".join(out)
