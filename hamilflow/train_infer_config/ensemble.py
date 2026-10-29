"""Wire k-fold splits -> per-fold train.toml -> per-fold infer.toml for a DeepH ensemble.

One AL cycle lives in ``cycle_dir``::

    cycle_dir/
        dataset_split_{j}.json          (from dataset_split.write_kfold_splits)
        train_{j}/train.toml            (write_ensemble_train_tomls)
        train_{j}/<outputs_dir>/<run>/model
        train_{j}/<infer_name>/infer.toml   (write_ensemble_infer_tomls)

A relative ``data.outputs_dir`` in train.toml is resolved against ``train_{j}/``, the folder
DeepH is launched from. ``<run>`` is the timestamped folder DeepH writes for each training run;
they are named ``yyyy-mm-dd_hh-mm-ss``, and when there are several the most recent one is used.
"""
import copy
import json
import logging
import re
import tomllib
import warnings
from pathlib import Path

from .write_train_infer_toml import (
    TrainSystemConfig,
    write_infer_toml,
    write_train_toml,
)

logger = logging.getLogger(__name__)

_RUN_NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
_SPLIT_KEYS = ("dataset_split_json", "train_size", "validate_size", "test_size")


def fold_dir(cycle_dir, fold: int) -> Path:
    return Path(cycle_dir) / f"train_{fold}"


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


def train_outputs_dir(cycle_dir, fold: int) -> Path:
    """``data.outputs_dir`` of fold ``fold``'s train.toml, resolved against its ``train_{fold}/``."""
    with open(fold_dir(cycle_dir, fold) / "train.toml", "rb") as f:
        outputs_dir = Path(tomllib.load(f)["data"]["outputs_dir"])
    return outputs_dir if outputs_dir.is_absolute() else fold_dir(cycle_dir, fold) / outputs_dir


def write_ensemble_train_tomls(
    overrides: dict,
    cycle_dir,
    k: int,
    previous_cycle_dir=None,
    continue_from: str | None = None,
) -> list[Path]:
    """Write ``train_{j}/train.toml`` for every fold j = 1..k of ``cycle_dir``.

    ``overrides`` holds the settings shared by all folds (as for ``write_train_toml``). Per fold
    this fills in the split file and its train/validate/test sizes, and sets the random seed to
    ``system.random_seed + j``. With ``previous_cycle_dir``, fold j warm-starts from the most
    recent run of the previous cycle's fold j (``process.train.continued`` with
    ``new_training_data=True``). With ``continue_from``, the name of an earlier outputs folder
    inside each ``train_{j}/`` (e.g. ``"outputs_3"``), fold j instead resumes from its most recent
    run there, on the same data (``new_training_data=False``). The other ``continued`` keys are
    taken from ``overrides``.
    """
    if previous_cycle_dir is not None and continue_from is not None:
        raise ValueError("Pass previous_cycle_dir or continue_from, not both")
    dataloader = overrides.get("process", {}).get("train", {}).get("dataloader", {})
    owned = [key for key in _SPLIT_KEYS if key in dataloader]
    if owned:
        raise ValueError(
            f"process.train.dataloader {', '.join(owned)} are set per fold from the split files; "
            "remove them from overrides"
        )
    base_seed = overrides.get("system", {}).get("random_seed", TrainSystemConfig.random_seed)

    paths = []
    for j in range(1, k + 1):
        split_path = (Path(cycle_dir) / f"dataset_split_{j}.json").resolve()
        split = json.loads(split_path.read_text())
        fold = copy.deepcopy(overrides)
        fold.setdefault("system", {})["random_seed"] = base_seed + j
        train = fold.setdefault("process", {}).setdefault("train", {})
        train.setdefault("dataloader", {}).update({
            "dataset_split_json": str(split_path),
            "train_size": len(split["train"]),
            "validate_size": len(split["validate"]),
            "test_size": len(split.get("test", [])),
        })
        if previous_cycle_dir is not None:
            previous_outputs, new_data = train_outputs_dir(previous_cycle_dir, j), True
        elif continue_from is not None:
            previous_outputs, new_data = fold_dir(cycle_dir, j) / continue_from, False
        else:
            previous_outputs = None
        if previous_outputs is not None:
            train.setdefault("continued", {}).update({
                "enable": True,
                "new_training_data": new_data,
                "previous_output_dir": str(find_run_dir(previous_outputs).resolve()),
            })
        paths.append(write_train_toml(fold, fold_dir(cycle_dir, j)))
    if previous_cycle_dir is not None:
        start = f"warm-started from the runs of {previous_cycle_dir} (new training data)"
    elif continue_from is not None:
        start = f"resuming from the runs in train_*/{continue_from} (same data)"
    else:
        start = "from scratch"
    logger.info(f"Wrote train.toml for {k} folds of {cycle_dir}, {start}")
    return paths


def write_ensemble_infer_tomls(
    overrides: dict,
    cycle_dir,
    k: int,
    infer_name: str = "infer",
) -> list[Path]:
    """Write ``train_{j}/{infer_name}/infer.toml`` for every fold j = 1..k of ``cycle_dir``.

    ``overrides`` holds the settings shared by all folds (as for ``write_infer_toml``); per fold
    this sets ``model.model_dir`` to the model of the most recent run in the outputs folder of
    that fold's train.toml. Use a different ``infer_name`` per inference dataset.
    """
    paths = []
    for j in range(1, k + 1):
        model_dir = find_run_dir(train_outputs_dir(cycle_dir, j)) / "model"
        fold = copy.deepcopy(overrides)
        fold.setdefault("model", {})["model_dir"] = str(model_dir.resolve())
        logger.debug(f"fold {j}: model {model_dir}")
        paths.append(write_infer_toml(fold, fold_dir(cycle_dir, j) / infer_name))
    return paths
