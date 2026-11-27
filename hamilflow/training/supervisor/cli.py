"""``hamilflow-train-supervisor``: look at and steer a supervised training cycle.

Runs on the login node, so it stays light: it reads state.json, the proposal, reports and the tail of the
current logs, writes small files, and makes at most one jobflow-remote call (``status --jobs`` reads the
job states, ``approve`` submits a flow). Exit codes: 0 ok, 1 error, 2 refused.

    hamilflow-train-supervisor status  <cycle_dir> [--jobs] [--json]
    hamilflow-train-supervisor review  <cycle_dir> [--member J] [--full] [--json]
    hamilflow-train-supervisor approve <cycle_dir> [--set J.key=value ...] [--edit] [--dry-run] [--resubmit]
    hamilflow-train-supervisor stop    <cycle_dir> [--member J] [--clear]
    hamilflow-train-supervisor finish  <cycle_dir> [--force]
    hamilflow-train-supervisor summary <cycle_dir> [--json] [--infer-tomls overrides.json [--infer-name infer]]

Never ``scancel`` / ``jf job cancel`` a train job: stop it with ``stop`` (STOP files), so it ends normally.
"""
import argparse
import copy
import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from ..ensemble import deep_merge, fold_dir, write_ensemble_infer_tomls, write_ensemble_train_tomls
from ..write_train_infer_toml import TrainConfig, validate_keys
from . import jobs, review
from . import status as status_view
from .policy import SupervisorConfig, plain_resume

OK, ERROR, REFUSED = 0, 1, 2
_TRAINS = ("resume", "retune", "restart_lower_lr")


class Refused(Exception):
    """A request the state does not allow (exit code 2)."""


# ---------------------------------- helpers ----------------------------------


def _pending_proposal(cycle_dir: Path, state) -> Path:
    if state["status"] != "awaiting_approval":
        raise Refused(f"no proposal pending: the cycle is {state['status']} (round {state['round']})")
    path = jobs.round_dir(cycle_dir, state["round"]) / "proposal.toml"
    if not path.is_file():
        raise Refused(f"state.json awaits approval but {path} is missing")
    return path


def _job_controller(project: str | None):
    from jobflow_remote import JobController

    return JobController.from_project_name(project)


# ---------------------------------- subcommands ----------------------------------


def cmd_status(args, out, job_controller=_job_controller) -> int:
    cycle_dir = Path(args.cycle_dir)
    state = jobs.read_state(cycle_dir)
    view = status_view.build_status(cycle_dir, state)
    if args.json:
        if args.jobs:
            view["jobs"] = _job_rows(state, job_controller)
        print(json.dumps(view, indent=1), file=out)
        return OK
    print(status_view.format_status(view, cycle=args.cycle_dir), file=out)
    if args.jobs:
        return _print_jobs(state, out, job_controller)
    return OK


_JOB_NAME = re.compile(r"^(?:train_member_(?P<fold>\d+)|supervise_round)_r(?P<round>\d+)$")


def _job_rows(state, job_controller) -> list[dict] | str:
    """This cycle's jobs (read only from the jobflow-remote DB), or the reason they cannot be read."""
    flow_ids = [flow["uuid"] for flow in state.get("flows", [])]
    if not flow_ids:
        return "no flow id in state.json"
    try:
        infos = job_controller(state.get("project")).get_jobs_info(flow_ids=flow_ids)
    except Exception as error:  # noqa: BLE001 - jobflow-remote missing, no DB: say so, nothing else breaks
        return f"jobflow-remote not available ({type(error).__name__}: {error})"
    return [{"name": str(i.name), "db_id": str(i.db_id), "state": str(getattr(i.state, "value", i.state))}
            for i in infos]


def _print_jobs(state, out, job_controller) -> int:
    """The jobs per round on one line each, then the way out of a stuck one."""
    rows = _job_rows(state, job_controller)
    if isinstance(rows, str):
        print(f"\njobs: {rows}", file=out)
        return ERROR
    by_round, other, problems = {}, {}, []
    for row in rows:
        match = _JOB_NAME.match(row["name"])
        if match:
            what = f"train {match['fold']}" if match["fold"] else "supervise"
            by_round.setdefault(int(match["round"]), []).append(f"{what} {row['state']}")
        else:
            other[row["state"]] = other.get(row["state"], 0) + 1
        if row["state"] == "FAILED" and row["name"].startswith(jobs.SUPERVISE_NAME):
            problems.append(f"{row['name']} [{row['db_id']}] FAILED: fix the cause (see its output), then "
                            f"`jf job rerun {row['db_id']}` (it decides the round again, once)")
        elif row["state"] == "FAILED":
            problems.append(f"{row['name']} [{row['db_id']}] FAILED: the job itself died (node failure, or a "
                            "SLURM kill: never scancel); the round's supervise job restarts the member")
        elif row["state"] == "REMOTE_ERROR":
            problems.append(f"{row['name']} [{row['db_id']}] REMOTE_ERROR (files not transferred, worker "
                            f"unreachable?): `jf job retry {row['db_id']}`")
    print("\njobs:", file=out)
    for n in sorted(by_round):
        print(f"  r{n}  {', '.join(sorted(by_round[n]))}", file=out)
    if other:
        print(f"  store_inputs etc.: {', '.join(f'{k} {v}' for k, v in sorted(other.items()))}", file=out)
    for problem in problems:
        print(f"  ! {problem}", file=out)
    return OK


def cmd_review(args, out) -> int:
    cycle_dir = Path(args.cycle_dir)
    state = jobs.read_state(cycle_dir)
    path = _pending_proposal(cycle_dir, state)
    if args.full:
        print(f"# {path}", file=out)
        print(path.read_text(), file=out)
        report = path.parent / "report.txt"
        if report.is_file():
            print(f"# {report}", file=out)
            print(report.read_text(), file=out)
        return OK
    view = review.build_review(cycle_dir, state, jobs.read_proposal(path))
    if args.member is not None:
        if str(args.member) not in state["members"]:
            raise ValueError(f"member {args.member} is not a member of this cycle")
        view["members"] = [m for m in view["members"] if m["fold"] == args.member]
    if args.json:
        print(json.dumps(view, indent=1), file=out)
    else:
        print(review.format_review(view, cycle=args.cycle_dir), file=out)
    return OK


# approve --set: J.key=value, with short names for the usual scheduler/optimizer keys
_SET_KEYS = {"patience": ("scheduler", "patience", int), "cooldown": ("scheduler", "cooldown", int),
             "rtol": ("scheduler", "rtol", float), "factor": ("scheduler", "factor", float),
             "lr": ("optimizer", "init_learning_rate", float),
             "min_scale": ("scheduler", "min_learning_rate_scale", float)}
_SET_RE = re.compile(r"(\d+)\.([A-Za-z_][\w.]*)=(.+)")


def _parse_value(raw: str):
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return raw


def _set_nested(d: dict, keys: Sequence[str], value) -> None:
    for key in keys[:-1]:
        d = d.setdefault(key, {})
        if not isinstance(d, dict):
            raise TypeError(f"cannot set {'.'.join(keys)}: {key} is not a table")
    d[keys[-1]] = value


def _apply_sets(proposal: dict, sets: Sequence[str], state: Mapping, out) -> None:
    """Apply ``--set J.key=value`` items in order. Keys: ``action`` (resume, done, give_up), the short
    names in ``_SET_KEYS``, or a full train.toml path (``process.train.scheduler.patience``)."""
    cfg = SupervisorConfig.from_dict(state["cfg"])
    for item in sets:
        match = _SET_RE.fullmatch(item.strip())
        if match is None:
            raise ValueError(f"--set {item!r}: expected J.key=value, e.g. 1.patience=15 or 2.action=resume")
        j, key, raw = int(match.group(1)), match.group(2), match.group(3)
        if str(j) not in state["members"]:
            raise ValueError(f"--set {item!r}: member {j} is not a member of this cycle")
        member = proposal["members"].setdefault(j, {"action": "done", "cause": "user", "reason": "",
                                                    "status": "", "fold_overrides": {}, "live_actions": {}})
        if key == "action":
            if raw in ("done", "give_up"):
                member.update(action=raw, cause="user", reason=f"{raw} (set by the user)",
                              continue_from=None, fold_overrides={})
            elif raw == "resume":
                runs = [r for r in state["members"][str(j)]["runs"] if r.get("run_dir")]
                last = runs[-1] if runs else {}
                cont, overrides = plain_resume(j, state["members"][str(j)]["logs"], last.get("run_dir"),
                                               last.get("exit"), cfg, state["base_overrides"])
                member.update(action="resume", cause="user", reason="resume (set by the user)",
                              continue_from=cont, fold_overrides=overrides, live_actions={})
            else:
                raise ValueError(f"--set {item!r}: action can be resume, done or give_up here; for a retune or "
                                 "a restart edit the proposal (--edit)")
        elif key in _SET_KEYS or key.startswith(("process.", "data.", "model.", "system.")):
            if member["action"] not in _TRAINS:
                raise ValueError(f"--set {item!r}: member {j} does not train ({member['action']}); set "
                                 f"{j}.action=resume first")
            if key in _SET_KEYS:
                part, name, cast = _SET_KEYS[key]
                try:
                    value = cast(raw)
                except ValueError:
                    raise ValueError(f"--set {item!r}: {key} must be a {cast.__name__}") from None
                _set_nested(member.setdefault("fold_overrides", {}), ["process", "train", part, name], value)
            else:
                _set_nested(member.setdefault("fold_overrides", {}), key.split("."), _parse_value(raw))
        else:
            raise ValueError(f"--set {item!r}: unknown key {key!r}; use action, {', '.join(_SET_KEYS)} or a "
                             "full train.toml path such as process.train.scheduler.patience")
        member["needs_approval"] = False
        print(f"set {item}", file=out)


def _validate(proposal, state) -> None:
    """The proposal's overrides must be valid train.toml keys (as merged into the round's TOMLs)."""
    base = deep_merge(state["base_overrides"], proposal.get("shared_overrides", {}))
    for j, member in proposal["members"].items():
        if str(j) not in state["members"]:
            raise ValueError(f"member {j} is not a member of this cycle")
        if member["action"] in _TRAINS:
            validate_keys(TrainConfig, deep_merge(base, member["fold_overrides"]))


def _submit(cycle_dir: Path, state: dict, before: dict, flow, submit: Callable | None, out) -> int:
    """Write the state, then submit. If submit_flow raises, the state is put back (approve can run again);
    on an interrupt (Ctrl-C, a kill) the flow may be in the DB already: the state stays running, for
    ``approve --resubmit`` (which checks the DB) to sort out."""
    jobs.write_state(cycle_dir, state)
    if submit is None:
        from jobflow_remote import submit_flow as submit
    try:
        submit(flow, worker=state["run_config"]["supervise"]["worker"], project=state.get("project"))
    except Exception as error:  # noqa: BLE001 - submit_flow refused: nothing was submitted
        jobs.write_state(cycle_dir, before)
        print(f"submission failed, state.json restored (the TOMLs stay): {type(error).__name__}: {error}", file=out)
        return ERROR
    except BaseException:  # Ctrl-C or a kill: the flow may have reached the DB, keep the state as running
        print(f"interrupted during submission: run `hamilflow-train-supervisor approve {cycle_dir} --resubmit`, "
              "which checks the DB first", file=out)
        raise
    print(f"submitted round {state['round']}", file=out)
    return OK


def cmd_approve(args, out, submit: Callable | None = None, editor_env: dict | None = None,
                job_controller=_job_controller) -> int:
    if args.resubmit:
        return cmd_resubmit(args, out, submit, job_controller)
    cycle_dir = Path(args.cycle_dir)
    state = jobs.read_state(cycle_dir)
    path = _pending_proposal(cycle_dir, state)
    if args.edit:
        editor = (editor_env or os.environ).get("EDITOR", "vi")
        result = subprocess.run([*shlex.split(editor), str(path)], check=False)
        if result.returncode != 0:
            print(f"the editor exited with {result.returncode}: nothing approved", file=out)
            return ERROR
    proposal = jobs.read_proposal(path)
    if proposal.get("after_round") != state["round"]:
        raise Refused(f"{path} is for round {proposal.get('after_round')}, state.json awaits round {state['round']}")
    if args.set:
        _apply_sets(proposal, args.set, state, out)
    _validate(proposal, state)
    next_round = state["round"] + 1
    pending = sorted(j for j, m in proposal["members"].items() if m["action"] in _TRAINS)
    removed = sorted(int(j) for j in state["members"] if int(j) not in proposal["members"])
    if not pending:
        raise Refused("no member trains in the proposal: use `finish` to end the cycle")
    continue_from = {j: proposal["members"][j]["continue_from"] for j in pending}
    fold_overrides = {j: proposal["members"][j]["fold_overrides"] for j in pending}
    live_actions = {j: proposal["members"][j]["live_actions"] for j in pending}
    print(f"round {next_round}: members {pending}", file=out)
    views = {m["fold"]: m for m in review.build_review(cycle_dir, state, proposal)["members"]}
    for j in pending:
        view = views[j]
        changes = "; ".join(view["changes"]) if view["changes"] else "settings unchanged"
        print(f"  member {j}: {proposal['members'][j]['action']}, starts {view['start']}; {changes}", file=out)
    if removed:
        print(f"  members {removed} are not in the proposal: treated as not training (done)", file=out)
    if args.dry_run:
        print("dry run: no TOML written, nothing submitted", file=out)
        return OK
    if args.set:  # keep the proposal file what was approved
        jobs.save_proposal(path, proposal)

    write_ensemble_train_tomls(
        deep_merge(state["base_overrides"], proposal.get("shared_overrides", {})), state["cycle_dir"], state["k"],
        round=next_round, folds=pending, continue_from=continue_from, fold_overrides=fold_overrides,
    )
    before = copy.deepcopy(state)
    record = state["history"]["rounds"][-1]
    for j, member in proposal["members"].items():
        decided = record["members"].setdefault(str(j), {})
        decided.update({key: member[key] for key in ("action", "continue_from", "fold_overrides", "live_actions")})
        decided.update({key: member[key] for key in ("cause", "reason") if member.get(key)})  # e.g. --set
        decided["needs_approval"], decided["approved"] = False, True
    for j in removed:
        decided = record["members"].setdefault(str(j), {})
        decided.update({"action": "done", "cause": "removed_by_user", "needs_approval": False, "approved": True})
    state["status"], state["round"] = "running", next_round
    flow = jobs.round_flow(state["cycle_dir"], next_round, state, pending, live_actions)
    state.setdefault("flows", []).append({"round": next_round, "uuid": flow.uuid})
    return _submit(cycle_dir, state, before, flow, submit, out)


def cmd_resubmit(args, out, submit: Callable | None, job_controller) -> int:
    """The state says round n runs, but its flow never reached jobflow-remote (e.g. approve was killed hard):
    submit round n again, after checking (read only) that no job of its recorded flow exists."""
    cycle_dir = Path(args.cycle_dir)
    state = jobs.read_state(cycle_dir)
    if state["status"] != "running":
        raise Refused(f"the cycle is {state['status']}: --resubmit is only for a round recorded as running")
    n = state["round"]
    flows = [flow["uuid"] for flow in state.get("flows", []) if flow["round"] == n]
    if not flows:
        raise Refused(f"no flow of round {n} is recorded in state.json")
    try:
        existing = job_controller(state.get("project")).get_jobs_info(flow_ids=flows)
    except Exception as error:  # without the check, never submit a round twice
        raise Refused(f"cannot check jobflow-remote ({type(error).__name__}: {error}). Check with "
                      f"`jf job list` that no job of round {n} exists, then run --resubmit where the DB is "
                      "reachable") from error
    if existing:
        raise Refused(f"round {n}'s flow has {len(existing)} job(s) in jobflow-remote: it was submitted; see "
                      "`status --jobs`")
    record = state["history"]["rounds"][-1]["members"] if state["history"]["rounds"] else None
    if record is None:
        pending = sorted(int(j) for j in state["members"])
        live_actions = {}
    else:
        pending = sorted(int(j) for j, m in record.items() if m["action"] in _TRAINS)
        live_actions = {int(j): record[j].get("live_actions", {}) for j in record}
    missing = [j for j in pending if not (fold_dir(cycle_dir, j) / jobs.train_toml_name(n)).is_file()]
    if missing:
        print(f"error: the TOMLs of round {n} are missing for members {missing}", file=out)
        return ERROR
    if args.dry_run:
        print(f"dry run: round {n} would be submitted again for members {pending}", file=out)
        return OK
    before = copy.deepcopy(state)
    flow = jobs.round_flow(state["cycle_dir"], n, state, pending, live_actions)
    state["flows"] = [f for f in state["flows"] if f["round"] != n] + [{"round": n, "uuid": flow.uuid}]
    print(f"round {n}: members {pending} (resubmission)", file=out)
    return _submit(cycle_dir, state, before, flow, submit, out)


def cmd_stop(args, out) -> int:
    cycle_dir = Path(args.cycle_dir)
    state = jobs.read_state(cycle_dir)
    if args.member is not None and str(args.member) not in state["members"]:
        print(f"member {args.member} is not a member of this cycle", file=out)
        return ERROR
    if state["status"] != "running" and not args.clear:
        raise Refused(f"the cycle is {state['status']}: no round is training, nothing to stop")
    here = jobs.round_dir(cycle_dir, state["round"])
    path = here / ("STOP_ROUND" if args.member is None else f"STOP_{args.member}")
    if args.clear:
        if path.exists():
            path.unlink()
            print(f"removed {path}", file=out)
        else:
            print(f"{path} was not set", file=out)
        return OK
    here.mkdir(parents=True, exist_ok=True)
    path.write_text("set by hamilflow-train-supervisor stop\n")
    print(f"wrote {path}: the watchdog stops at its next poll", file=out)
    return OK


def cmd_finish(args, out) -> int:
    cycle_dir = Path(args.cycle_dir)
    state = jobs.read_state(cycle_dir)
    if state["status"] == "awaiting_approval" and not args.force:
        raise Refused("a proposal is pending: approve it, or finish --force to drop it")
    if state["status"] == "running":
        # the round's supervise job owns state.json while the round runs: leave it a FINISH file
        path = jobs.finish_path(cycle_dir)
        path.write_text("set by hamilflow-train-supervisor finish\n")
        print(f"wrote {path}: round {state['round']}'s supervise job ends the cycle and writes the summary. "
              f"Round {state['round']} may still be training: `hamilflow-train-supervisor stop {cycle_dir}` "
              "stops it", file=out)
        return OK
    state["finish_requested"] = True
    state["status"] = "finished"
    jobs.write_state(cycle_dir, state)
    path = jobs.write_summary(cycle_dir, jobs.summary(state))
    print(f"finished: no further round. Summary in {path}", file=out)
    return OK


def cmd_summary(args, out) -> int:
    cycle_dir = Path(args.cycle_dir)
    state = jobs.read_state(cycle_dir)
    final = jobs.summary(state)
    if args.json:
        print(json.dumps(final, indent=1), file=out)
    else:
        print(f"{final['cycle_dir']}: round {final['round']}, {final['status']}", file=out)
        for fold, member in sorted(final["members"].items(), key=lambda kv: int(kv[0])):
            print(f"  member {fold}: {member['action']} ({member['cause']}) -> {member['final_run_dir']}", file=out)
            for settings in member["settings"]:
                shown = {k: v for k, v in settings.items() if k not in ("round", "action")}
                print(f"    round {settings['round']} {settings['action']}: {shown}", file=out)
    if args.infer_tomls is None:
        return OK
    overrides = json.loads(Path(args.infer_tomls).read_text())
    run_dirs = {int(fold): m["final_run_dir"] for fold, m in final["members"].items() if m["final_run_dir"]}
    missing = sorted(int(fold) for fold, m in final["members"].items() if not m["final_run_dir"])
    paths = write_ensemble_infer_tomls(overrides, state["cycle_dir"], state["k"], infer_name=args.infer_name,
                                       run_dirs=run_dirs)
    for path in paths:
        print(f"wrote {path}", file=out)
    if missing:
        print(f"no run with a checkpoint for members {missing}: no infer.toml for them", file=out)
        return ERROR
    return OK


# ---------------------------------- entry point ----------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hamilflow-train-supervisor", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("status", help="the round and what each member is doing now, alerts, STOP files")
    p.add_argument("cycle_dir")
    p.add_argument("--jobs", action="store_true", help="also the jobflow-remote job states (reads the DB)")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("review", help="the pending proposal: per member what is proposed, why, and what changes")
    p.add_argument("cycle_dir")
    p.add_argument("--member", type=int, help="this member only")
    p.add_argument("--full", action="store_true", help="the raw proposal.toml and the whole analysis report")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("approve", help="write the next round's TOMLs from the proposal and submit it")
    p.add_argument("cycle_dir")
    p.add_argument("--set", action="append", metavar="J.KEY=VALUE",
                   help="change member J before approving: action=resume|done|give_up, patience, cooldown, rtol, "
                        "factor, lr, min_scale, or a full train.toml path; repeatable, applied in order")
    p.add_argument("--edit", action="store_true", help="open the proposal in $EDITOR first")
    p.add_argument("--dry-run", action="store_true", help="check and show, write and submit nothing")
    p.add_argument("--resubmit", action="store_true",
                   help="submit the running round again if its flow never reached jobflow-remote (checks the DB)")
    p = sub.add_parser("stop", help="stop the round (STOP_ROUND) or one member (STOP_J) at the next poll")
    p.add_argument("cycle_dir")
    p.add_argument("--member", type=int)
    p.add_argument("--clear", action="store_true", help="remove the STOP file instead")
    p = sub.add_parser("finish", help="no further round; write supervisor/summary.json")
    p.add_argument("cycle_dir")
    p.add_argument("--force", action="store_true", help="also while a proposal is pending (it is dropped)")
    p = sub.add_parser("summary", help="final run dir, verdict and settings history per member")
    p.add_argument("cycle_dir")
    p.add_argument("--json", action="store_true")
    p.add_argument("--infer-tomls", metavar="OVERRIDES_JSON",
                   help="also write train_J/<infer-name>/infer.toml per member from its final run dir")
    p.add_argument("--infer-name", default="infer")
    return parser


def main(argv: Sequence[str] | None = None, *, out=None, submit: Callable | None = None,
         job_controller: Callable | None = None, editor_env: dict | None = None) -> int:
    """Run one subcommand; ``submit``, ``job_controller`` and ``editor_env`` are injectable for tests."""
    out = sys.stdout if out is None else out
    args = _parser().parse_args(argv)
    try:
        if not jobs.state_path(args.cycle_dir).is_file():
            print(f"no supervisor state in {args.cycle_dir} ({jobs.state_path(args.cycle_dir)})", file=out)
            return ERROR
        if args.command == "status":
            return cmd_status(args, out, job_controller or _job_controller)
        if args.command == "approve":
            return cmd_approve(args, out, submit, editor_env, job_controller or _job_controller)
        return {"review": cmd_review, "stop": cmd_stop, "finish": cmd_finish, "summary": cmd_summary}[
            args.command](args, out)
    except Refused as refusal:
        print(f"refused: {refusal}", file=out)
        return REFUSED
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"error: {type(error).__name__}: {error}", file=out)
        return ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
