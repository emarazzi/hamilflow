"""Check the LR reductions and the convergence of a DeepH ensemble, and suggest ``patience`` and
``cooldown`` for ``type = "reduce_on_plateau"``, from the training logs.

Input: LOGS, one entry per ensemble member. A member trained over several jobs
(``[process.train.continued]``) is the list of its logs, in order.

Run with:  python analyze_training_logs.py
"""

from pathlib import Path

from hamilflow.training.log_analysis import analyze_training_logs, format_report

# --- User settings -----------------------------------------------------------
CYCLE_DIR = Path("/path/to/your/models/cycle1")
K = 5
LOGS = [CYCLE_DIR / f"train_{j}" / "deepx.log" for j in range(1, K + 1)]
# Over several jobs:
#   LOGS = [[CYCLE_DIR / f"train_{j}" / run / "deepx.log" for run in ("outputs_1", "outputs_2")]
#           for j in range(1, K + 1)]

MONITOR = "mean"  # loss seen by the scheduler: "train", "val" or "mean"
MIN_SCALE = 1e-4  # min_learning_rate_scale in train.toml
WALLTIME_H = 24.0  # walltime per job, to express the training length in jobs; None to skip
# -----------------------------------------------------------------------------

analysis = analyze_training_logs(LOGS, monitor=MONITOR, min_scale=MIN_SCALE, walltime_h=WALLTIME_H)
print(format_report(analysis))
