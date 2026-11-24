"""Wire k-fold splits -> per-fold train.toml -> per-fold infer.toml for a DeepH ensemble.

One AL cycle lives in ``cycle_dir``::

    cycle_dir/
        dataset_split_{j}.json          (from dataset_split.write_kfold_splits)
        train_{j}/train.toml            (write_ensemble_train_tomls)
        train_{j}/<outputs_dir>/<run>/model
        train_{j}/<infer_name>/infer.toml   (write_ensemble_infer_tomls)

With ``round = n`` (supervised training, one round of jobs after the other), the train.toml of
the round is ``train_{j}/train_r{n}.toml`` and its runs go to ``<outputs_dir>_r{n}``
(``./outputs_r{n}`` for the usual ``./outputs``), so rounds never overwrite each other.

A relative ``data.outputs_dir`` in train.toml is resolved against ``train_{j}/``, the folder
DeepH is launched from. ``<run>`` is the timestamped folder DeepH writes for each training run;
they are named ``yyyy-mm-dd_hh-mm-ss``, and when there are several the most recent one is used.
"""
import copy
import json
import logging
import operator
import re
import tomllib
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

from .write_train_infer_toml import (
    TRAIN_RULES,
    TrainConfig,
    TrainDataConfig,
    TrainSystemConfig,
    apply_overrides,
    validate_constraints,
    validate_keys,
    write_infer_toml,
    write_train_toml,
)

logger = logging.getLogger(__name__)

_RUN_NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
_SPLIT_KEYS = ("dataset_split_json", "train_size", "validate_size", "test_size")
# what a per-fold continue_from entry may hold
_CONTINUE_KEYS = ("outputs_dir", "previous_output_dir", "load_model_type", "new_training_data", "new_optimizer")
_LOAD_MODEL_TYPES = ("best", "latest")


def fold_dir(cycle_dir, fold: int) -> Path:
    return Path(cycle_dir) / f"train_{fold}"


def train_toml_name(round: int | None = None) -> str:
    """``train.toml``, or ``train_r{round}.toml`` for one round of a supervised training."""
    return "train.toml" if round is None else f"train_r{round}.toml"


def deep_merge(base: dict, extra: dict) -> dict:
    """A copy of ``base`` with ``extra`` merged in: tables are merged key by key, any other value of
    ``extra`` replaces the one in ``base``."""
    merged = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def find_run_dir(outputs_dir) -> Path:
    """Most recent training run in ``outputs_dir``: the latest ``yyyy-mm-dd_hh-mm-ss`` subfolder
    holding a ``model/``."""
    outputs_dir = Path(outputs_dir)
    runs = sorted(p for p in outputs_dir.iterdir() if _RUN_NAME_RE.match(p.name) and (p / "model").is_dir())
    if not runs:
        raise FileNotFoundError(f"No yyyy-mm-dd_hh-mm-ss training run with a model/ folder in {outputs_dir}")
    if len(runs) > 1:
        warnings.warn(
            f"{len(runs)} training runs in {outputs_dir}, using the most recent: {runs[-1].name}",
            stacklevel=2,
        )
    return runs[-1]


def train_outputs_dir(cycle_dir, fold: int, round: int | None = None) -> Path:
    """``data.outputs_dir`` of fold ``fold``'s train.toml (of round ``round`` if given), resolved
    against its ``train_{fold}/``."""
    with open(fold_dir(cycle_dir, fold) / train_toml_name(round), "rb") as f:
        outputs_dir = Path(tomllib.load(f)["data"]["outputs_dir"])
    return outputs_dir if outputs_dir.is_absolute() else fold_dir(cycle_dir, fold) / outputs_dir


def _as_int(value) -> int | None:
    """``value`` as an int if it is an integer (numpy ones too, not bools), else None."""
    if isinstance(value, bool):
        return None
    try:
        return operator.index(value)
    except TypeError:
        return None


def _check_folds(name: str, folds_given, folds: list[int]) -> list[int]:
    """The folds of ``folds_given`` as ints, all among ``folds``."""
    converted = [_as_int(j) for j in folds_given]
    unknown = [j for j, c in zip(folds_given, converted, strict=True) if c is None or c not in folds]
    if unknown:
        raise ValueError(f"{name}: folds {unknown} are not among {folds}")
    return converted


def _positive_int(name: str, value) -> int:
    converted = _as_int(value)
    if converted is None or converted < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return converted


def _table(table: Mapping, key: str, where: str) -> Mapping:
    """``table[key]``, {} if absent; a clear error if it is not a table."""
    value = table.get(key, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"{where}.{key} must be a table (dict), got {type(value).__name__}")
    return value


def _continue_spec(spec, fold: int) -> dict | None:
    """Normalize one per-fold ``continue_from`` entry to ``{"outputs_dir" | "previous_output_dir", and
    optionally "load_model_type"}``; None stays None (that fold starts from scratch)."""
    if spec is None:
        return None
    if isinstance(spec, str | Path):
        return {"outputs_dir": spec}
    if not isinstance(spec, Mapping):
        raise TypeError(f"continue_from[{fold}] must be an outputs folder, a dict or None, got {type(spec).__name__}")
    unknown = sorted(set(spec) - set(_CONTINUE_KEYS))
    if unknown:
        raise ValueError(f"continue_from[{fold}]: unknown keys {unknown}, expected some of {_CONTINUE_KEYS}")
    if ("outputs_dir" in spec) == ("previous_output_dir" in spec):
        raise ValueError(f"continue_from[{fold}]: give exactly one of outputs_dir and previous_output_dir")
    if spec.get("load_model_type", "best") not in _LOAD_MODEL_TYPES:
        raise ValueError(f"continue_from[{fold}]: load_model_type must be one of {_LOAD_MODEL_TYPES}, "
                         f"got {spec['load_model_type']!r}")
    for key in ("new_training_data", "new_optimizer"):
        if not isinstance(spec.get(key, False), bool):
            raise TypeError(f"continue_from[{fold}]: {key} must be a bool, got {spec[key]!r}")
    return dict(spec)


def _check_owned(fold_override: dict, fold: int, continued: dict, round: int | None) -> None:
    """Per-fold overrides must not set what this function fills in for that fold."""
    if not isinstance(fold_override, Mapping):
        raise TypeError(f"fold_overrides[{fold}] must be a dict of overrides, got {type(fold_override).__name__}")
    where = f"fold_overrides[{fold}]"
    train = _table(_table(fold_override, "process", where), "train", f"{where}.process")
    dataloader = _table(train, "dataloader", f"{where}.process.train")
    owned = [f"process.train.dataloader.{key}" for key in _SPLIT_KEYS if key in dataloader]
    if round is not None and "outputs_dir" in _table(fold_override, "data", where):
        owned.append("data.outputs_dir (set by round)")
    if owned:
        raise ValueError(f"{where} sets {', '.join(owned)}, which are set per fold; remove them")
    fold_continued = _table(train, "continued", f"{where}.process.train")
    clashes = [key for key, value in fold_continued.items() if key in continued and value != continued[key]]
    if clashes:
        raise ValueError(f"fold_overrides[{fold}] sets process.train.continued {', '.join(clashes)}, which "
                         f"continue_from / previous_cycle_dir set to {', '.join(repr(continued[k]) for k in clashes)}")


def _round_outputs_dir(outputs_dir: str, round: int) -> str:
    """``<outputs_dir>_r{round}``, keeping a relative path relative (``./outputs`` -> ``./outputs_r2``).
    Unset, or the fold folder itself (``"."``), it is ``./outputs_r{round}``."""
    if outputs_dir == TrainDataConfig.outputs_dir or Path(outputs_dir) == Path("."):
        outputs_dir = "./outputs"
    return f"{outputs_dir.rstrip('/')}_r{round}"


def write_ensemble_train_tomls(
    overrides: dict,
    cycle_dir,
    k: int,
    previous_cycle_dir=None,
    continue_from: str | Mapping[int, str | Path | Mapping | None] | None = None,
    fold_overrides: Mapping[int, dict] | None = None,
    round: int | None = None,
    folds: Sequence[int] | None = None,
    previous_round: int | None = None,
) -> list[Path]:
    """Write ``train_{j}/train.toml`` (``train_r{round}.toml`` with ``round``) for every fold j = 1..k
    of ``cycle_dir``, or for the ``folds`` given.

    ``overrides`` holds the settings shared by all folds (as for ``write_train_toml``). Per fold
    this fills in the split file and its train/validate/test sizes, and sets the random seed to
    ``system.random_seed + j``. With ``previous_cycle_dir``, fold j warm-starts from the most
    recent run of the previous cycle's fold j (``process.train.continued`` with
    ``new_training_data=True``). With ``continue_from``, the name of an earlier outputs folder
    inside each ``train_{j}/`` (e.g. ``"outputs_3"``), fold j instead resumes from its most recent
    run there, on the same data (``new_training_data=False``). The other ``continued`` keys are
    taken from ``overrides``. ``previous_round = n`` (only with ``previous_cycle_dir``) warm-starts
    from a previous cycle trained in rounds: from ``train_r{n}.toml`` and its outputs folder, so
    every fold's final run must be in round n. When the members of the previous cycle stopped at
    different rounds, use the per-fold ``continue_from={j: {"outputs_dir": ...,
    "new_training_data": True}}`` instead (None for a fold that starts from scratch).

    Per fold, for a supervisor that tunes and resumes each member on its own:

    - ``continue_from`` can be a dict fold -> where to resume from, on the same data: an outputs
      folder (relative to ``train_{j}/`` or absolute; its most recent run is used), or a dict with
      ``outputs_dir`` or an explicit run dir ``previous_output_dir``, plus optionally
      ``load_model_type`` (``"latest"`` for a plain resume after a walltime or stop, ``"best"`` for
      a restart after a divergence or a retune), ``new_training_data`` (default False) and
      ``new_optimizer`` (else as in the overrides).
      None starts that fold from scratch. Every fold written needs an entry.
    - ``fold_overrides`` maps fold -> overrides deep-merged after the shared ones, e.g. that member's
      ``optimizer.init_learning_rate``, ``scheduler.min_learning_rate_scale``, ``patience``,
      ``cooldown`` or ``continued.new_optimizer``. It must not set the split keys, nor a
      ``continued`` key to another value than ``continue_from`` gives it.
    - ``round = n`` writes ``train_{j}/train_r{n}.toml`` with ``data.outputs_dir`` set to the
      shared one plus ``_r{n}`` (``./outputs_r{n}``), so rounds never overwrite each other.
    - ``folds`` restricts the folds written (default 1..k), e.g. to the members still training.
    """
    if previous_cycle_dir is not None and continue_from is not None:
        raise ValueError("Pass previous_cycle_dir or continue_from, not both")
    if previous_round is not None:
        if previous_cycle_dir is None:
            raise ValueError("previous_round is the round of previous_cycle_dir to warm-start from: pass both")
        previous_round = _positive_int("previous_round", previous_round)
    if round is not None:
        round = _positive_int("round", round)
    folds = _check_folds("folds", list(range(1, k + 1)) if folds is None else list(folds), list(range(1, k + 1)))
    if len(set(folds)) != len(folds):
        raise ValueError(f"folds has duplicates: {folds}")
    fold_overrides = fold_overrides or {}
    fold_overrides = dict(zip(_check_folds("fold_overrides", fold_overrides, folds), fold_overrides.values(),
                              strict=True))
    per_fold_continue = isinstance(continue_from, Mapping)
    if per_fold_continue:
        continue_from = dict(zip(_check_folds("continue_from", continue_from, folds), continue_from.values(),
                                 strict=True))
        missing = [j for j in folds if j not in continue_from]
        if missing:
            raise ValueError(f"continue_from has no entry for folds {missing} (use None to start one from scratch)")
    elif continue_from is not None and not isinstance(continue_from, str | Path):
        raise TypeError(f"continue_from must be an outputs folder name or a dict per fold, got "
                        f"{type(continue_from).__name__}")
    dataloader = overrides.get("process", {}).get("train", {}).get("dataloader", {})
    owned = [key for key in _SPLIT_KEYS if key in dataloader]
    if owned:
        raise ValueError(
            f"process.train.dataloader {', '.join(owned)} are set per fold from the split files; "
            "remove them from overrides"
        )
    base_seed = overrides.get("system", {}).get("random_seed", TrainSystemConfig.random_seed)
    # check every fold's input before writing anything, so a bad entry leaves no partial round behind
    specs = {j: _continue_spec(continue_from[j], j) if per_fold_continue else continue_from for j in folds}

    plans = []
    for j in folds:
        split_path = (Path(cycle_dir) / f"dataset_split_{j}.json").resolve()
        split = json.loads(split_path.read_text())
        fold = copy.deepcopy(overrides)
        fold.setdefault("system", {})["random_seed"] = base_seed + j
        if round is not None:
            data = fold.setdefault("data", {})
            data["outputs_dir"] = _round_outputs_dir(data.get("outputs_dir", TrainDataConfig.outputs_dir), round)
        train = fold.setdefault("process", {}).setdefault("train", {})
        train.setdefault("dataloader", {}).update({
            "dataset_split_json": str(split_path),
            "train_size": len(split["train"]),
            "validate_size": len(split["validate"]),
            "test_size": len(split.get("test", [])),
        })
        continued = _continued(cycle_dir, j, previous_cycle_dir, specs[j], previous_round)
        _check_owned(fold_overrides.get(j, {}), j, continued, round)
        fold = deep_merge(fold, fold_overrides.get(j, {}))
        if continued:
            fold["process"]["train"].setdefault("continued", {}).update(continued)
        validate_keys(TrainConfig, fold)
        validate_constraints(apply_overrides(TrainConfig(), fold), TRAIN_RULES)
        plans.append((j, fold))
    paths = [write_train_toml(fold, fold_dir(cycle_dir, j), filename=train_toml_name(round)) for j, fold in plans]
    if previous_cycle_dir is not None:
        of_round = "" if previous_round is None else f" (round {previous_round})"
        start = f"warm-started from the runs of {previous_cycle_dir}{of_round} (new training data)"
    elif per_fold_continue:
        start = f"resuming {sum(s is not None for s in specs.values())} of them (same data), per-fold sources"
    elif continue_from is not None:
        start = f"resuming from the runs in train_*/{continue_from} (same data)"
    else:
        start = "from scratch"
    which = f"{len(folds)} folds" if len(folds) == k else f"folds {folds}"
    round_text = "" if round is None else f", round {round}"
    logger.info(f"Wrote {train_toml_name(round)} for {which} of {cycle_dir}{round_text}, {start}")
    return paths


def _continued(cycle_dir, fold: int, previous_cycle_dir, spec, previous_round: int | None = None) -> dict:
    """The ``process.train.continued`` keys set for one fold (empty: from scratch, or as in the overrides)."""
    if previous_cycle_dir is not None:
        run, new_data = find_run_dir(train_outputs_dir(previous_cycle_dir, fold, round=previous_round)), True
    elif spec is None:
        return {}
    elif isinstance(spec, str | Path):
        run, new_data = find_run_dir(fold_dir(cycle_dir, fold) / spec), False
    elif "previous_output_dir" in spec:
        run, new_data = Path(spec["previous_output_dir"]), False
        if not run.is_absolute():
            run = fold_dir(cycle_dir, fold) / run
        if not (run / "model").is_dir():
            raise FileNotFoundError(f"continue_from[{fold}]: no model/ folder in previous_output_dir {run}")
    else:
        outputs = Path(spec["outputs_dir"])
        run, new_data = find_run_dir(outputs if outputs.is_absolute() else fold_dir(cycle_dir, fold) / outputs), False
    if isinstance(spec, Mapping):
        new_data = spec.get("new_training_data", new_data)
    continued = {"enable": True, "new_training_data": new_data, "previous_output_dir": str(run.resolve())}
    if isinstance(spec, Mapping):
        continued.update({key: spec[key] for key in ("load_model_type", "new_optimizer") if key in spec})
    return continued


def write_ensemble_infer_tomls(
    overrides: dict,
    cycle_dir,
    k: int,
    infer_name: str = "infer",
    run_dirs: Mapping[int, str | Path] | None = None,
) -> list[Path]:
    """Write ``train_{j}/{infer_name}/infer.toml`` for every fold j = 1..k of ``cycle_dir``.

    ``overrides`` holds the settings shared by all folds (as for ``write_infer_toml``); per fold
    this sets ``model.model_dir`` to the model of the most recent run in the outputs folder of
    that fold's train.toml. Use a different ``infer_name`` per inference dataset. ``run_dirs``
    (fold -> run dir, e.g. the supervisor's ``final_run_dir``s) gives each fold's run explicitly
    instead, and only those folds are written.
    """
    folds = range(1, k + 1) if run_dirs is None else sorted(run_dirs)
    paths = []
    for j in folds:
        run = find_run_dir(train_outputs_dir(cycle_dir, j)) if run_dirs is None else Path(run_dirs[j])
        model_dir = run / "model"
        if run_dirs is not None and not model_dir.is_dir():
            raise FileNotFoundError(f"fold {j}: no model/ folder in {run}")
        fold = copy.deepcopy(overrides)
        fold.setdefault("model", {})["model_dir"] = str(model_dir.resolve())
        logger.debug(f"fold {j}: model {model_dir}")
        paths.append(write_infer_toml(fold, fold_dir(cycle_dir, j) / infer_name))
    return paths
