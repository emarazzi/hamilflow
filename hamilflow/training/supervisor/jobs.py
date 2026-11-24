"""The jobflow side of the supervisor: one train job per member and round on the GPU
worker, one light supervise job per round, and the Maker that builds round 1.

Flow of a cycle, submitted once::

    round n:  train_member_{j}_r{n} (k jobs, GPU)  ->  supervise_round_r{n} (light worker)
              supervise: round_decision -> finished | awaiting approval | Response(replace=round n+1)

The supervise job has ``OnMissing.NONE`` (a FAILED train job gives it None, a crash for the policy)
and ``pass_manager_config = False``: jobflow 0.4.0 otherwise copies the supervise job's own manager
config (light worker) onto every job of the replacement (``Job.run``, jobflow/core/job.py, after the
``config_updates``), so the next round's train jobs would land on the login node. It sets the run
configs of the new jobs itself, from state.json. The ``store_inputs`` job jobflow adds for each
replace gets no manager config, so jobflow-remote gives it the supervise job's worker.

``state.json`` (``cycle_dir/supervisor/state.json``) is the single source of truth: written at
submission by ``make()`` and afterwards by the supervise jobs only, atomically. Run dirs come from the
``ExitInfo``s, never from globbing the outputs folders. This module needs jobflow; the rest of the
package does not.
"""
import contextlib
import copy
import json
import logging
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tomli_w
from jobflow import Flow, OnMissing, Response, job

from ..._logging import job_logging
from ..ensemble import deep_merge, fold_dir, train_toml_name, write_ensemble_train_tomls
from ..log_analysis import format_report, read_training_curve
from .policy import ExitInfo, RoundDecision, SupervisorConfig, append_history, new_history, round_decision
from .watchdog import last_checkpoint_epoch, run_supervised

logger = logging.getLogger(__name__)

TRAIN_NAME = "train_member"
SUPERVISE_NAME = "supervise_round"
STATE_VERSION = 1
_ACTIONS = ("done", "resume", "retune", "restart_lower_lr", "give_up")


# ---------------------------------- files ----------------------------------


def supervisor_dir(cycle_dir) -> Path:
    return Path(cycle_dir) / "supervisor"


def round_dir(cycle_dir, round_: int) -> Path:
    return supervisor_dir(cycle_dir) / f"round_{round_}"


def state_path(cycle_dir) -> Path:
    return supervisor_dir(cycle_dir) / "state.json"


def finish_path(cycle_dir) -> Path:
    """``supervisor/FINISH``: the user's ``finish`` while a round runs; the round's supervise job ends the cycle."""
    return supervisor_dir(cycle_dir) / "FINISH"


def summary_path(cycle_dir) -> Path:
    return supervisor_dir(cycle_dir) / "summary.json"


def write_summary(cycle_dir, final: Mapping) -> Path:
    path = summary_path(cycle_dir)
    path.write_text(json.dumps(final, indent=1))
    return path


def read_state(cycle_dir) -> dict:
    return json.loads(state_path(cycle_dir).read_text())


def write_state(cycle_dir, state: Mapping) -> Path:
    """Atomically: a temporary file in the same folder, flushed, then renamed over state.json."""
    path = state_path(cycle_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".state.", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return path


def _strip_none(value):
    """TOML has no null: drop None values (a missing continue_from means from scratch)."""
    if isinstance(value, Mapping):
        return {str(k): _strip_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_strip_none(v) for v in value]
    return value


def write_proposal(path, decision: RoundDecision) -> Path:
    """``round_{n}/proposal.toml``: per member, the action of the next round and its TOML overrides,
    editable. Read back with ``read_proposal`` (step 6). A member without ``continue_from`` starts from
    scratch; ``needs_approval`` marks the actions the policy asks about."""
    members = {}
    for j, action in sorted(decision.actions.items()):
        members[str(j)] = _strip_none({
            "action": action.action, "cause": action.cause, "reason": action.reason, "status": action.status,
            "needs_approval": action.needs_approval, "continue_from": action.continue_from,
            "fold_overrides": action.fold_overrides, "live_actions": action.live_actions,
        })
    data = {
        "round": decision.round + 1,
        "after_round": decision.round,
        "note": ("Edit action (done, resume, retune, restart_lower_lr, give_up), continue_from and fold_overrides "
                 "per member, then approve (hamilflow-train-supervisor approve, step 6). Members whose action "
                 "is not resume/retune/restart_lower_lr are not trained."),
        "shared_overrides": decision.shared_overrides,
        "members": members,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)
    return path


def read_proposal(path) -> dict:
    """The proposal as ``{"round": n + 1, "shared_overrides": ..., "members": {fold: {"action",
    "continue_from" (None: from scratch), "fold_overrides", "live_actions", ...}}}``, validated."""
    import tomllib

    with open(path, "rb") as f:
        data = tomllib.load(f)
    members = {}
    for key, member in data.get("members", {}).items():
        if member.get("action") not in _ACTIONS:
            raise ValueError(f"{path}: member {key}: action must be one of {_ACTIONS}, got {member.get('action')!r}")
        members[int(key)] = {**member, "continue_from": member.get("continue_from"),
                             "fold_overrides": member.get("fold_overrides", {}),
                             "live_actions": member.get("live_actions", {})}
    return {**data, "members": members}


# ---------------------------------- jobs ----------------------------------


@contextlib.contextmanager
def _light_process():
    """The supervise job runs on the login node: niced, BLAS single-threaded."""
    with contextlib.suppress(OSError):
        if (current := os.nice(0)) < 10:
            os.nice(10 - current)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - threadpoolctl comes with scikit-learn/numpy stacks
        limits = contextlib.nullcontext()
    else:
        limits = threadpool_limits(1)
    with limits:
        yield


@job
def train_member_job(cycle_dir: str, fold: int, round_: int, cfg: dict, member_logs_before: list[str],
                     shared_overrides: dict, live_actions: dict) -> dict:
    """Run and supervise fold ``fold``'s ``train_r{round}.toml`` (GPU worker). Returns ``ExitInfo.to_dict()``;
    never raises for trainer failures (``run_supervised``). The end time comes from SLURM (``SLURM_JOB_END_TIME``)."""
    with job_logging():
        fold_path = fold_dir(Path(cycle_dir).resolve(), fold)
        info = run_supervised(
            fold_path / train_toml_name(round_), fold_path, member_logs_before, SupervisorConfig.from_dict(cfg),
            round_dir(Path(cycle_dir).resolve(), round_), fold, shared_overrides, live_actions=live_actions,
        )
        return info.to_dict()


def _manager_config(run_config: Mapping | None) -> dict:
    """``Job.config.manager_config`` as jobflow-remote reads it (jobflow_remote/jobs/data.py,
    ``get_initial_job_doc_dict``: ``worker``, ``resources``, ``exec_config``, ``priority``)."""
    return {key: copy.deepcopy(value) for key, value in (run_config or {}).items()
            if key in ("worker", "resources", "exec_config", "priority") and value is not None}


def round_flow(cycle_dir, round_: int, state: Mapping, folds: Sequence[int],
               live_actions: Mapping[int, Mapping] | None = None) -> Flow:
    """The jobs of round ``round_``: one train job per fold of ``folds`` and the supervise job, with the
    run configs of state.json. Its TOMLs must be written already."""
    cycle_dir = str(Path(cycle_dir).resolve())
    live_actions = live_actions or {}
    train_jobs = {}
    for j in folds:
        train = train_member_job(cycle_dir=cycle_dir, fold=j, round_=round_, cfg=state["cfg"],
                                 member_logs_before=list(state["members"][str(j)]["logs"]),
                                 shared_overrides=state["base_overrides"], live_actions=dict(live_actions.get(j, {})))
        train.name = f"{TRAIN_NAME}_{j}_r{round_}"
        train.config.manager_config = _manager_config(state["run_config"].get("train"))
        train_jobs[j] = train
    supervise = supervise_round_job(cycle_dir=cycle_dir, round_=round_,
                                    exit_infos={str(j): t.output for j, t in train_jobs.items()})
    supervise.name = f"{SUPERVISE_NAME}_r{round_}"
    supervise.config.on_missing_references = OnMissing.NONE
    supervise.config.pass_manager_config = False
    supervise.config.manager_config = _manager_config(state["run_config"].get("supervise"))
    return Flow([*train_jobs.values(), supervise], output=supervise.output, name=f"{state['name']}_r{round_}")


def _forget_round(state: dict, round_: int) -> bool:
    """Drop what an earlier run of this round's supervise job recorded (rounds >= ``round_`` in the history,
    the members' runs and logs), so a rerun (``jf job rerun`` after a failure) decides it once. True if
    there was anything to drop."""
    rounds = state["history"]["rounds"]
    stale = any(r["round"] >= round_ for r in rounds) or any(
        run["round"] >= round_ for member in state["members"].values() for run in member["runs"])
    state["history"]["rounds"] = [r for r in rounds if r["round"] < round_]
    for member in state["members"].values():
        member["runs"] = [run for run in member["runs"] if run["round"] < round_]
        member["logs"] = _chain(member["runs"])
    return stale


def _chain(runs) -> list[str]:
    """A member's chained logs, oldest first, each once: the logs of its recorded runs."""
    logs = []
    for run in runs:
        if run.get("log") and run["log"] not in logs:
            logs.append(run["log"])
    return logs


def _record_runs(state: dict, round_: int, exit_infos: Mapping[int, ExitInfo | None]) -> None:
    """Each member's runs and chain from the ExitInfos (never by globbing): the log of its run dir, if
    written, crashed runs included."""
    for j, info in exit_infos.items():
        member = state["members"][str(j)]
        run_dir = None if info is None else info.run_dir
        log = None if run_dir is None else Path(run_dir) / "deepx.log"
        member["runs"].append({"round": round_, "run_dir": run_dir, "exit": None if info is None else info.to_dict(),
                               "log": str(log) if log is not None and log.is_file() else None})
        member["logs"] = _chain(member["runs"])


def final_run_dir(member: Mapping) -> str | None:
    """The run to infer with: the one holding the best val of the member's lineage (its chained logs, dead
    branches cut) if it has a complete ``best`` checkpoint of its params (what inference loads: the states
    can lag after a SIGINT); else the latest run with a complete params ``best`` (inference loads
    ``best``); else None. A few stats of the checkpoint dirs, no globbing."""
    try:
        curve = read_training_curve(member["logs"]) if member["logs"] else None
    except ValueError:
        curve = None
    if curve is not None and np.isfinite(curve.val).any():
        i = int(np.nanargmin(np.where(np.isfinite(curve.val), curve.val, np.nan)))
        job = next(job for job in curve.jobs if job.start <= i < job.stop)
        run = Path(job.path).parent
        if last_checkpoint_epoch(run, settle_s=0.0, kind="best", parts=("params",)) is not None:
            return str(run)
    for run in reversed(member["runs"]):
        if run["run_dir"] is None:
            continue
        if last_checkpoint_epoch(run["run_dir"], settle_s=0.0, kind="best", parts=("params",)) is not None:
            return run["run_dir"]
    return None


def summary(state: Mapping) -> dict:
    """The final output: per member, its final run dir (``final_run_dir``), verdict and the settings of each
    round it trained (rounds with a recorded run only). A member the user's ``finish`` stopped while it
    would have trained on gets the action ``finished_by_user``."""
    by_user = bool(state.get("finish_requested"))
    members = {}
    for key, member in state["members"].items():
        records = [(r["round"], r["members"].get(key)) for r in state["history"]["rounds"]]
        last = next((rec for _, rec in reversed(records) if rec is not None), None)
        trained = {run["round"] for run in member["runs"]}
        settings = []
        for n, rec in records:
            if rec is None or rec["action"] not in ("resume", "retune", "restart_lower_lr") or n + 1 not in trained:
                continue
            train = rec["fold_overrides"].get("process", {}).get("train", {})
            settings.append({"round": n + 1, "action": rec["action"],
                             **{k: v for part in ("optimizer", "scheduler") for k, v in train.get(part, {}).items()}})
        action = None if last is None else last["action"]
        if by_user and action in ("resume", "retune", "restart_lower_lr"):
            action = "finished_by_user"
        members[key] = {
            "final_run_dir": final_run_dir(member),
            "action": action,
            "cause": None if last is None else last["cause"],
            "reason": None if last is None else last["reason"],
            "overfitting": False if last is None else last.get("overfitting", False),
            "settings": settings,
        }
    return {"status": state["status"], "round": state["round"], "cycle_dir": state["cycle_dir"], "members": members}


@job
def supervise_round_job(cycle_dir: str, round_: int, exit_infos: dict):
    """After round ``round_``: decide, record, and continue (light worker, seconds: parses logs, writes
    TOML/JSON). ``exit_infos``: fold (str) -> ``ExitInfo.to_dict()`` of the members trained this round,
    None for a job that never returned."""
    with job_logging(), _light_process():
        state = read_state(cycle_dir)
        if _forget_round(state, round_):
            logger.warning(f"Round {round_} was recorded already (a rerun of its supervise job): its records are "
                           "dropped and the round is decided again")
        state["status"], state["round"] = "running", round_
        if finish_path(cycle_dir).exists():  # the user's `finish` while this round ran
            state["finish_requested"] = True
        cfg = SupervisorConfig.from_dict(state["cfg"])
        base = state["base_overrides"]
        infos = {int(j): None if info is None else ExitInfo.from_dict(info) for j, info in exit_infos.items()}
        _record_runs(state, round_, infos)
        member_logs = {int(j): member["logs"] for j, member in state["members"].items()}
        decision = round_decision(member_logs, infos, cfg, state["history"], shared_overrides=base, round_=round_)
        here = round_dir(cycle_dir, round_)
        here.mkdir(parents=True, exist_ok=True)
        (here / "report.txt").write_text(format_report(decision.analysis) + "\n" if decision.analysis
                                         else "no member has logged an epoch yet\n")
        (here / "decision.json").write_text(json.dumps(decision.to_dict(), indent=1))
        state["history"] = append_history(state["history"], decision)

        if decision.finished or state.get("finish_requested"):  # the user's `finish`: no further round
            state["status"] = "finished"
            write_state(cycle_dir, state)
            finish_path(cycle_dir).unlink(missing_ok=True)  # converted into finish_requested, now recorded
            final = summary(state)
            write_summary(cycle_dir, final)
            logger.info(f"Round {round_}: finished, summary in {summary_path(cycle_dir)}")
            return final
        if decision.needs_approval:
            proposal = write_proposal(here / "proposal.toml", decision)
            state["status"] = "awaiting_approval"
            write_state(cycle_dir, state)
            logger.info(f"Round {round_}: awaiting approval, see {proposal}")
            return {"status": "awaiting_approval", "round": round_, "proposal": str(proposal),
                    "report": str(here / "report.txt")}

        next_round = round_ + 1
        write_ensemble_train_tomls(
            deep_merge(base, decision.shared_overrides), cycle_dir, state["k"], round=next_round,
            folds=decision.pending_folds, continue_from=decision.continue_from, fold_overrides=decision.fold_overrides,
        )
        state["status"], state["round"] = "running", next_round
        write_state(cycle_dir, state)
        logger.info(f"Round {round_}: next round {next_round} for folds {decision.pending_folds}")
        flow = round_flow(cycle_dir, next_round, state, decision.pending_folds, decision.live_actions)
        return Response(replace=flow, output={"status": "running", "round": next_round})


# ---------------------------------- maker ----------------------------------


@dataclass
class TrainingSupervisorMaker:
    """Builds round 1 of a supervised ensemble training of ``cycle_dir``.

    ``train_overrides`` are the train.toml settings shared by all members (as for
    ``write_ensemble_train_tomls``); the supervisor's own ones (``cfg.shared_overrides()``) are merged
    into every round. ``*_train`` / ``*_supervise``: worker, resources (dict, JSON) and exec_config (a
    name or dict) of the GPU train jobs and of the light supervise jobs, kept in state.json for every
    later round (and for the approval CLI, with ``project``).
    """

    cfg: SupervisorConfig
    train_overrides: dict
    k: int
    resources_train: dict | None = None
    resources_supervise: dict | None = None
    train_worker: str | None = None
    supervise_worker: str | None = None
    exec_config_train: str | dict | None = None
    exec_config_supervise: str | dict | None = None
    project: str | None = None
    name: str = "train_supervisor"

    def make(self, cycle_dir, previous_cycle_dir=None, previous_round: int | None = None,
             continue_from=None) -> Flow:
        """Write round 1's TOMLs and state.json (at submission, light: a few small files, and it fails early
        on bad overrides), and return the round-1 Flow. Warm-start as ``write_ensemble_train_tomls``:
        ``previous_cycle_dir`` (+ ``previous_round`` for a cycle trained in rounds) or ``continue_from``.
        The member chains start here: earlier cycles' logs are not chained (contract 10)."""
        cycle_dir = Path(cycle_dir).resolve()
        missing = [name for name in ("train_worker", "supervise_worker") if getattr(self, name) is None]
        if missing:
            raise ValueError(
                f"set {' and '.join(missing)}: every job of every round gets its worker from state.json; without "
                "one, jobflow-remote gives a replaced round's jobs the supervise job's (light) worker, which would "
                "run GPU training on the login node")
        if state_path(cycle_dir).exists():
            raise FileExistsError(f"{state_path(cycle_dir)} exists: this cycle is already supervised")
        resources = {"resources_train": self.resources_train, "resources_supervise": self.resources_supervise}
        for what, value in resources.items():
            if value is not None and not isinstance(value, Mapping):
                raise TypeError(f"{what} must be a dict (it is kept in state.json), got {type(value).__name__}")
        paths = write_ensemble_train_tomls(
            deep_merge(self.train_overrides, self.cfg.shared_overrides()), cycle_dir, self.k, round=1,
            previous_cycle_dir=previous_cycle_dir, previous_round=previous_round, continue_from=continue_from,
        )
        _check_best_with_new_optimizer(paths)
        state = {
            "version": STATE_VERSION,
            "name": self.name,
            "cycle_dir": str(cycle_dir),
            "k": self.k,
            "base_overrides": copy.deepcopy(self.train_overrides),
            "cfg": self.cfg.to_dict(),
            "run_config": {
                "train": {"worker": self.train_worker, "resources": self.resources_train,
                          "exec_config": self.exec_config_train},
                "supervise": {"worker": self.supervise_worker, "resources": self.resources_supervise,
                              "exec_config": self.exec_config_supervise},
            },
            "project": self.project,
            "status": "running",
            "round": 1,
            "history": {**new_history(), "start": _round_1_start(paths)},
            "members": {str(j): {"logs": [], "runs": []} for j in range(1, self.k + 1)},
        }
        flow = round_flow(cycle_dir, 1, state, list(range(1, self.k + 1)))
        state["flows"] = [{"round": 1, "uuid": flow.uuid}]  # for the CLI's job status (jobflow-remote flow ids)
        write_state(cycle_dir, state)
        return flow


def warm_start_from(previous_cycle_dir) -> dict[int, dict]:
    """``continue_from`` for ``make()`` of the next AL cycle: each member starts from its final run in the
    previous, supervised cycle (``final_run_dir``: members end in different rounds, so no single round
    holds them all), from its ``best`` params, on the new data, with a new optimizer (``best`` is never
    loaded with the optimizer kept: params and states ``best`` can differ after a SIGINT). Reads the previous
    cycle's ``supervisor/summary.json``, or its state.json when there is none."""
    path = summary_path(previous_cycle_dir)
    final = json.loads(path.read_text()) if path.is_file() else summary(read_state(previous_cycle_dir))
    missing = sorted(int(j) for j, member in final["members"].items() if not member["final_run_dir"])
    if missing:
        raise ValueError(f"members {missing} of {previous_cycle_dir} have no run with a complete best checkpoint: "
                         "give their continue_from by hand (or None to start them from scratch)")
    return {int(j): {"previous_output_dir": member["final_run_dir"], "load_model_type": "best",
                     "new_training_data": True, "new_optimizer": True}
            for j, member in final["members"].items()}


def _check_best_with_new_optimizer(paths) -> None:
    """The supervisor never loads ``best`` with the optimizer kept (params and states ``best`` can differ after a SIGINT): refuse such a round 1."""
    import tomllib

    for path in paths:
        with open(path, "rb") as f:
            continued = tomllib.load(f)["process"]["train"]["continued"]
        if continued["enable"] and continued["load_model_type"] == "best" and not continued["new_optimizer"]:
            raise ValueError(f"{path} loads the best model with the optimizer kept: set new_optimizer = true "
                             "(warm_start_from does), or load latest")


def _round_1_start(paths) -> dict:
    """``history["start"]``: how each member's round 1 was started (a warm start's continued section), so a
    crash before round 1's first checkpoint starts it the same way again."""
    import tomllib

    start = {}
    for j, path in enumerate(paths, start=1):
        with open(path, "rb") as f:
            continued = tomllib.load(f)["process"]["train"]["continued"]
        entry = None
        if continued["enable"]:
            entry = {"previous_output_dir": continued["previous_output_dir"],
                     "new_training_data": continued["new_training_data"],
                     "load_model_type": continued["load_model_type"]}
        if entry is not None:
            entry["new_optimizer"] = continued["new_optimizer"]
        start[str(j)] = {"continue_from": entry, "fold_overrides": {}}
    return start
