"""Write one DeepH train.toml per fold of a k-fold ensemble.

Reads the ``dataset_split_{1..k}.json`` files that make_kfold_splits.py wrote in CYCLE_DIR and
writes ``CYCLE_DIR/train_{j}/train.toml``. The settings below are shared by all folds; per fold
the split file, the train/validate/test sizes and the random seed (``random_seed + j``) are
filled in, so don't set ``process.train.dataloader.dataset_split_json`` or the sizes here.

Active learning: with PREVIOUS_CYCLE_DIR set, fold j continues training from the most recent
run of the previous cycle's fold j (``process.train.continued`` with ``new_training_data``).
Resuming within a cycle: with CONTINUE_FROM set to an earlier outputs folder (e.g. "outputs_3"),
fold j resumes from its most recent run in ``train_{j}/outputs_3``; give ``outputs_dir`` a new
name (e.g. "./outputs_4") so the runs stay apart.

Launch DeepH from each ``train_{j}/`` folder: a relative ``outputs_dir`` is resolved there.
Next step, once training is done: write_infer_tomls.py.

Run with:  python write_train_tomls.py
"""

from pathlib import Path

from hamilflow.training.ensemble import write_ensemble_train_tomls

# --- User settings -----------------------------------------------------------
CYCLE_DIR = Path("/path/to/your/models/cycle2")
PREVIOUS_CYCLE_DIR = None  # e.g. Path("/path/to/your/models/cycle1"); None = train from scratch
CONTINUE_FROM = None  # e.g. "outputs_3", inside each train_{j}/; can't be combined with PREVIOUS_CYCLE_DIR
K = 5

OVERRIDES = {
    "system": {
        "device": "gpu*2:0-1",
        "random_seed": 137,  # fold j uses random_seed + j
        "float_type": "tf32",
        "jax_memory_preallocate": False,
    },
    "data": {
        "inputs_dir": "/path/to/your/training/inputs",
        "outputs_dir": "./outputs",
        "graph": {
            "dataset_name": "tbmos2",
            "graph_type": "HS",  # "H" | "HS" | "Rho" | "Sap" | "S"
            "storage_type": "disk",
            "common_orbital_types": "s2p2d2f1",
        },
    },
    "model": {
        "net_type": "eagle",  # "sparrow" | "normal" | "eagle" | "accurate"
        "target_type": "H",  # "H" | "Rho"
        "loss_type": "mse",
        "advanced": {
            "net_irreps": "16x0e+16x1e+20x2e+16x3e+8x4e+4x5e+2x6e",  # required, no valid default
            "gaussian_basis_rmax": 12.0,
            "num_blocks": 3,
            "consider_parity": False,
            "standardize_gauge": True,
        },
    },
    "process": {
        "train": {
            "max_epoch": 10000,
            "dataloader": {"batch_size": 1},
            "optimizer": {"type": "adamw", "init_learning_rate": 1e-3},
            "scheduler": {
                "type": "reduce_on_plateau",
                "min_learning_rate_scale": 1e-7,
                "patience": 20,
                "cooldown": 10,
            },
            # Used only with PREVIOUS_CYCLE_DIR or CONTINUE_FROM; enable,
            # new_training_data and previous_output_dir are filled in per fold.
            "continued": {"new_optimizer": True, "load_model_type": "best"},
        },
    },
}
# -----------------------------------------------------------------------------

paths = write_ensemble_train_tomls(
    OVERRIDES, CYCLE_DIR, K, previous_cycle_dir=PREVIOUS_CYCLE_DIR, continue_from=CONTINUE_FROM
)
for path in paths:
    print(f"Wrote {path}")
