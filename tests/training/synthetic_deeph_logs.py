"""Synthetic DeepH-pack 1.0.6 training logs, in the exact line format of real logs (paths and machine
names left out), shared by the training tests (on sys.path through conftest.py)."""

import numpy as np

# the text after the run dir in a real log (DeepH-pack 1.0.6, cluster check 2026-11-24) of a run continued with
# new_optimizer = true, and with the optimizer kept (new_optimizer = false)
NEW_OPTIMIZER = "We are using the origin training dataset, and applying new optimizer."
KEPT_OPTIMIZER = "We are using the origin training dataset, and NOT applying new optimizer."


def write_log(
    path,
    train,
    val=None,
    scale=None,
    patience=120,
    cooldown=50,
    rtol=0.05,
    factor=0.5,
    accumulation=40,
    n_train=40,
    batch_size=1,
    lr0="0.001",
    continued_from=None,
    load_model_type="best",
    continuation=NEW_OPTIMIZER,
    first_epoch=1,
):
    """A log with the header and epoch lines of a real run (paths and machine names left out). With
    ``continued_from`` (a run dir), the continuation line of a ``[process.train.continued]`` run, with
    ``continuation`` the text after the run dir; ``lr0`` is written as is, as DeepH prints it.
    ``first_epoch``: the number of the first epoch line; DeepH continues the numbering of the checkpoint
    it loaded when the optimizer is kept (e.g. 9 after latest epoch 8), and starts at 1 otherwise."""
    train = np.asarray(train, dtype=float)
    val = train if val is None else np.asarray(val, dtype=float)
    scale = np.ones(len(train)) if scale is None else np.asarray(scale, dtype=float)
    lines = [
        f"[ 10.22-10:30:47 ] [dataloader] Train size: `{n_train}`. Val size: `10`. Test size: `0`.",
        (
            f"[ 10.22-10:30:47 ] [dataloader] Data sharding way: `2`. Batch size: `{batch_size}`. "
            "Number of nodes each batch: `[944]`, `[944]`, `[]`."
        ),
        (
            "[ 10.22-10:31:49 ] [optimizer] Using the optimizer `AdamW` with: betas `[0.9, 0.999]`, eps `1e-08`, "
            f"weight decay strength `0.001`, and initial learning rate `{lr0}`."
        ),
        (
            f"[ 10.22-10:31:49 ] [optimizer] Using the scheduler `ReduceOnPlateau` with: factor `{factor}`, "
            f"patience `{patience}`, rtol `{rtol}`, cooldown `{cooldown}`, and accumulation size `{accumulation}`."
        ),
        ("[ 10.22-10:31:52 ] [model] We will save the model into `/x/model`. The best model will be saved. "
         "The latest model (keep `10` each `100` epoch) will be saved."),
    ]
    if continued_from is not None:
        lines.append(
            f"[ 10.28-10:36:31 ] [model] We are loading the `{load_model_type}` model inside `{continued_from}`. "
            f"{continuation}"
        )
    lines += [
        f"[ 10.22-10:37:39 ] [train] Epoch {i + first_epoch} | Time 200.00 s | Train-Loss {t:.6e} | "
        f"Val-Loss {v:.6e} | Scale {s}"
        for i, (t, v, s) in enumerate(zip(train, val, scale, strict=True))
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def decay_then_noise(n_epochs=400, flat_from=200, noise=0.02, seed=0):
    """log10(loss) falls by 0.01 per epoch until `flat_from`, then only fluctuates."""
    rng = np.random.default_rng(seed)
    log_loss = -2.0 - 0.01 * np.minimum(np.arange(n_epochs), flat_from) + noise * rng.standard_normal(n_epochs)
    return 10**log_loss
