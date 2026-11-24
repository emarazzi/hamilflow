"""A fake ``deeph-train`` for the watchdog tests: ``python fake_deeph_train.py <train.toml> <spec.json>``.

It mimics what the watchdog sees of DeepH-pack 1.0.6 (checked on a cluster, 2026-11-24): it reads the
train.toml and fails at start-up if ``[process.train.continued]`` lacks one of its six keys; it creates the
timestamped run dir in ``data.outputs_dir`` and writes the header lines of ``deepx.log`` (with the
continuation line of a continued run, in the real wording) and the epoch lines. With the optimizer kept, a
continued run carries on the epoch numbers of the checkpoint it loaded (``load_model_epoch``, or the newest
``latest`` when -1); with a new optimizer it starts at 1. Checkpoints are written as orbax does, atomically: a
``epoch_N.orbax-checkpoint-tmp-K`` dir, a ``_CHECKPOINT_METADATA`` file, then a rename to ``epoch_N``;
``latest`` every ``data.model_save.latest_interval`` epochs (on the global numbers) and at the run's first
epoch, ``best`` on each val improvement, and the epoch's log line only after its checkpoints. On SIGINT it
exits with code 1 after ``sigint_exit_s`` (real: ~4 s), optionally leaving a checkpoint tmp dir behind. The
spec (JSON) sets the losses, the scale, the pace and the signal handling. Standard library only: it runs with
the cleaned environment of the trainer.
"""
import datetime
import json
import math
import os
import re
import signal
import sys
import time
import tomllib
from pathlib import Path

CONTINUED_KEYS = ("enable", "new_training_data", "new_optimizer", "previous_output_dir", "load_model_type",
                  "load_model_epoch")
EPOCH_DIR = re.compile(r"^epoch_(\d+)$")


def stamp() -> str:
    return datetime.datetime.now().strftime("[ %m.%d-%H:%M:%S ]")


def checkpoint(run_dir: Path, kind: str, epoch: int, keep: int) -> None:
    """Write ``epoch_N`` in params and states as orbax does (tmp dir, metadata, rename), keeping the newest
    ``keep`` (DeepH: latest_num; best: 1)."""
    for part in ("params", "states"):
        pytree = run_dir / "model" / part / f"{kind}.pytree"
        tmp = pytree / f"epoch_{epoch}.orbax-checkpoint-tmp-0"
        tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "data").write_text(f"{kind} {epoch}\n")
        (tmp / "_CHECKPOINT_METADATA").write_text("{}\n")
        tmp.rename(pytree / f"epoch_{epoch}")
        done = sorted((p for p in pytree.iterdir() if EPOCH_DIR.match(p.name)), key=lambda p: int(p.name[6:]))
        for stale in done[:-keep]:
            for entry in stale.iterdir():
                entry.unlink()
            stale.rmdir()


def loaded_epoch(continued: dict) -> int:
    """The epoch of the checkpoint a continued run loads: ``load_model_epoch``, or the newest one."""
    if continued["load_model_epoch"] >= 0:
        return continued["load_model_epoch"]
    pytree = Path(continued["previous_output_dir"]) / "model" / "params" / f"{continued['load_model_type']}.pytree"
    epochs = [int(m.group(1)) for p in pytree.iterdir() if (m := EPOCH_DIR.match(p.name))] if pytree.is_dir() else []
    return max(epochs, default=0)


def main(toml_path: str, spec_path: str) -> int:
    spec = json.loads(Path(spec_path).read_text())
    if spec.get("pidfile"):
        Path(spec["pidfile"].replace("{}", str(os.getpid()))).write_text(str(os.getpid()))  # "{}": one per run
    print(f"PYTHONPATH={os.environ.get('PYTHONPATH')}", flush=True)
    print(f"VIRTUAL_ENV={os.environ.get('VIRTUAL_ENV')}", flush=True)
    print(f"FAKE_MARK={os.environ.get('FAKE_MARK')}", flush=True)
    if spec.get("sigint", "exit") == "ignore":
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    else:
        signal.signal(signal.SIGINT, signal.default_int_handler)  # KeyboardInterrupt, as deeph-train
    if spec.get("sigterm", "default") == "ignore":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

    parent = os.getppid()
    with open(toml_path, "rb") as f:
        config = tomllib.load(f)
    missing = [key for key in CONTINUED_KEYS if key not in config["process"]["train"].get("continued", {})]
    if missing:  # real DeepH reads every key, even with enable = false
        print("Traceback (most recent call last):\n  File \"deepx/train.py\", in _validate_continue_dir\n"
              f"AttributeError: 'ContinuedConfig' object has no attribute 'previous_output_dir' (missing {missing})",
              flush=True)
        return 1
    # per fold and round: "by_run": {"train_1/outputs_r2": {...}} overrides the spec for that run
    run_key = f"{Path.cwd().name}/{Path(config['data']['outputs_dir']).name}"
    spec = {**spec, **spec.get("by_run", {}).get(run_key, {})}
    time.sleep(spec.get("startup_s", 0.0))
    if spec.get("no_run_dir"):
        print("error before the run dir", flush=True)
        return 1
    outputs = Path(config["data"]["outputs_dir"])
    while (run_dir := outputs / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")).exists():
        time.sleep(0.05)
    (run_dir / "model").mkdir(parents=True)
    train = config["process"]["train"]
    scheduler, continued = train["scheduler"], train["continued"]
    interval = config["data"]["model_save"]["latest_interval"]
    n = spec.get("n_epochs", 20)
    val = spec.get("val") or [10 ** (-2 - 0.01 * i) for i in range(n)]
    loss = spec.get("train") or val
    scale = spec.get("scale") or [1.0] * n
    lines = [
        f"{stamp()} [dataloader] Train size: `40`. Val size: `10`. Test size: `0`.",
        f"{stamp()} [dataloader] Data sharding way: `2`. Batch size: `1`. Number of nodes each batch: `[944]`.",
        (f"{stamp()} [optimizer] Using the optimizer `AdamW` with: betas `[0.9, 0.999]`, eps `1e-08`, weight decay "
         f"strength `0.001`, and initial learning rate `{train['optimizer']['init_learning_rate']}`."),
        (f"{stamp()} [optimizer] Using the scheduler `ReduceOnPlateau` with: factor `{scheduler['factor']}`, "
         f"patience `{scheduler['patience']}`, rtol `{scheduler['rtol']}`, cooldown `{scheduler['cooldown']}`, "
         f"and accumulation size `40`."),
    ]
    first_epoch = 1
    if continued["enable"]:
        optimizer = "applying new optimizer" if continued["new_optimizer"] else "NOT applying new optimizer"
        if not continued["new_optimizer"]:
            first_epoch = loaded_epoch(continued) + 1
        lines.append(f"{stamp()} [model] We are loading the `{continued['load_model_type']}` model inside "
                     f"`{continued['previous_output_dir']}`. We are using the origin training dataset, and "
                     f"{optimizer}.")
    log = run_dir / "deepx.log"
    with open(log, "w") as f:
        f.write("\n".join(lines) + "\n")
    best = math.inf
    try:
        for i in range(n):
            epoch = first_epoch + i
            time.sleep(spec.get("epoch_s", 0.01))
            v, t, s = val[min(i, len(val) - 1)], loss[min(i, len(loss) - 1)], scale[min(i, len(scale) - 1)]
            if v < best:
                best = v
                checkpoint(run_dir, "best", epoch, keep=1)
            if i == 0 or epoch % interval == 0:  # deeph-train also saves at a run's first epoch
                checkpoint(run_dir, "latest", epoch, keep=10)
            with open(log, "a") as f:  # after the epoch's checkpoints, as deeph-train
                logged = spec.get("logged_epoch_s", spec.get("epoch_s", 0.01))
                f.write(f"{stamp()} [train] Epoch {epoch} | Time {logged:.4f} s | "
                        f"Train-Loss {t:.6e} | Val-Loss {v:.6e} | Scale {s}\n")
            if os.getppid() != parent and not spec.get("orphan_ok"):  # the watchdog is gone: do not linger
                return 3
            if spec.get("crash_at") == i + 1:
                print("Traceback (most recent call last): fake crash", flush=True)
                return 1
        time.sleep(spec.get("exit_s", 0.2))  # tear-down after the last checkpoint
    except KeyboardInterrupt:
        if spec.get("leave_tmp_on_sigint"):  # interrupted in the middle of an orbax save
            tmp = run_dir / "model" / "params" / "latest.pytree" / "epoch_9999.orbax-checkpoint-tmp-3"
            tmp.mkdir(parents=True, exist_ok=True)
        print("KeyboardInterrupt", flush=True)
        time.sleep(spec.get("sigint_exit_s", 0.2))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
