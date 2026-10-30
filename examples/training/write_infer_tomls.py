"""Write one DeepH infer.toml per trained fold of a k-fold ensemble.

For each fold j, reads ``CYCLE_DIR/train_{j}/train.toml`` to find its outputs folder, takes the
model of the most recent training run there (with a warning if there are several runs), and
writes ``CYCLE_DIR/train_{j}/{INFER_NAME}/infer.toml``. Use a different INFER_NAME for each
dataset you run inference on, e.g. "infer_calibrate", "infer_pool".

The predictions of the k models feed examples/uncertainty/.

Run with:  python write_infer_tomls.py
"""

from pathlib import Path

from hamilflow.training.ensemble import write_ensemble_infer_tomls

# --- User settings -----------------------------------------------------------
CYCLE_DIR = Path("/path/to/your/models/cycle2")
K = 5
INFER_NAME = "infer"

OVERRIDES = {
    "system": {"device": "gpu*8:0"},
    "data": {
        "inputs_dir": "/path/to/your/inference/inputs",
        "outputs_dir": "./outputs",
        "graph": {
            "dataset_name": "tbmos2",
            "graph_type": "S",  # "Sap" | "S"
            "storage_type": "memory",
        },
    },
    "model": {"load_model_type": "best"},  # model_dir is filled in per fold
    "process": {
        "infer": {
            "output_type": "h5",
            "output_into": "to_output",
            "target_symmetrize": True,
            "dataloader": {"batch_size": 1},
        },
    },
}
# -----------------------------------------------------------------------------

for path in write_ensemble_infer_tomls(OVERRIDES, CYCLE_DIR, K, infer_name=INFER_NAME):
    print(f"Wrote {path}")
