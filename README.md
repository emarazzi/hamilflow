# hamilflow

Community-maintained utilities around [DeepH](https://github.com/kYangLi/DeepH-dock) workflows: from picking training structures, through FHI-aims data generation and basis projection, to uncertainty estimates on model predictions. Not an official DeepH package.

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.12. `deepx-dock` is installed from its GitHub repository.

The dense eigensolvers (band structures, band uncertainty, orbital removal) run on a GPU when
CuPy is installed and a CUDA device is visible:

```bash
pip install -e .[gpu]
```

**This installs `cupy-cuda12x`, built for CUDA 12.** If the CUDA of your machine or cluster
module is another major version, replace it with the matching wheel, e.g. for CUDA 11:

```bash
pip uninstall cupy-cuda12x && pip install cupy-cuda11x
```

Check with `python -c "import cupy; cupy.show_config()"`; hamilflow also warns at the first
diagonalization if CuPy is installed but cannot use the GPU, and then runs on the CPU. Set
`HAMILFLOW_GPU=0` to always use the CPU, or `HAMILFLOW_GPU=1` to fail instead of falling back
(see `hamilflow/sparse_hamiltonian/dense_backend.py`).

## Components

### Candidate selection — `hamilflow.candidate_selection`

`CandidateSelector` picks representative structures from a trajectory using Average Minimum Distance (AMD) fingerprints. It selects by farthest-point sampling, k-means centroids, or a regex on frame labels, and can draw a 2D MDS map of the dataset. See [examples/candidate_selection/](examples/candidate_selection).

### DFT workflows — `hamilflow.dft_workflows`

jobflow/atomate2 flows that produce DeepH-ready data from FHI-aims, one job per structure:

- `GenerateAimsDFTData`: runs AIMS calculations (or collects existing ones), with optional AIMS → DeepH conversion.
- `GenerateProjectedDeephInputs`, `GenerateAimsToProjectedDeephData`: chain DFT, conversion and projection end to end. `GenerateOvlOnlyAimsToProjectedDeephData` does the same for overlap only.
- Helpers: `get_ksampling` for k-point settings.

### Projection — `hamilflow.projection`

`run_projection(ProjectionConfig)` removes orbitals from a DeepH Hamiltonian and overlap with a k-space Schur-complement reduction, following a `RemovalPlan` (JSON, dict or object). The reduction runs on the sparse `hamilflow.sparse_hamiltonian` objects, so it never builds the dense real-space arrays and works on large structures. Overlap-only and dummy-Hamiltonian modes are supported. See [examples/projection/](examples/projection).

### Uncertainty quantification — `hamilflow.uncertainty`

Tools that estimate how uncertain an ensemble of DeepH predictions is, both on the Hamiltonian matrices and on the band structures, measured against DFT references. This module is still changing quickly, so expect its API to move. See [examples/uncertainty/](examples/uncertainty).

### Supervised ensemble training — `hamilflow.training.supervisor`

Trains a k-fold DeepH ensemble through jobflow-remote with one submission, in rounds: each round runs
every member's `deeph-train` under a watchdog on a GPU worker, then a light supervise job decides from all
members' logs what each member does next. See
[examples/training/supervised_training.py](examples/training/supervised_training.py).

**Typical session**

```bash
python examples/training/supervised_training.py          # writes round 1 and submits the flow
hamilflow-train-supervisor status  CYCLE_DIR [--jobs]    # what each member is doing now, alerts, STOP files
hamilflow-train-supervisor review  CYCLE_DIR             # what is proposed per member, why, what changes
hamilflow-train-supervisor approve CYCLE_DIR [--set 1.patience=15 ...] [--edit]   # writes the next round, submits
hamilflow-train-supervisor stop    CYCLE_DIR [--member J]
hamilflow-train-supervisor summary CYCLE_DIR --infer-tomls infer_overrides.json
```

**Following a cycle.** `status` prints the round (N of `max_rounds`) and one line per member: what it is
doing now (queued, starting, training, stopping, stopped (walltime/…), done, given up), its run, the last
epoch's val, LR scale and LR, seconds per epoch, the time left before the watchdog's walltime stop, and the
decision that put it in this round (or, while a proposal waits, what is proposed). Alerts, STOP and FINISH
files follow, then the commands for what can be done next. `--jobs` adds the jobflow-remote states, one
line per round, and the fix for a FAILED or REMOTE_ERROR job; `--json` the same as JSON. It only reads the
ends of the current logs and the watchdog journals.

**The end of a cycle.** `summary` prints one line per member: its verdict in words (converged, stopped:
overfitting, max rounds reached, finished by the user, …), the rounds it trained, its epochs and best val,
and its final model (the run holding the lineage's best val, relative to the cycle); then each member's
settings per round, the first in full and only what changed after it (`r1 start: lr 0.001, patience 300, …
· r2 retune: patience 15, cooldown 7 · r3 resume`). `--json` is what `summary.json` holds.

**Reviewing a proposal.** `review` prints one line per member (proposed action, `*` if it needs your
approval, epochs, best val, LR scale, reductions, val/train and a val-loss sparkline), then for each member
where its next run starts (e.g. `from best of r1, NEW optimizer`) and only the settings that change
(`patience 120 → 25; cooldown 50 → 12; lr 0.002 (carried over)`), the evidence behind the decisions that
need approval (the overfit trend per LR stage, or a retune's onset and suggested patience), and
ready-to-copy commands. `review --member J` shows one member, `--json` the same as JSON, `--full` the raw
`proposal.toml` and the whole analysis report. `approve --set J.key=value` (repeatable, applied in order)
changes the proposal before approving: `action=resume|done|give_up` (resume fills in a plain resume from the
member's last run), `patience`, `cooldown`, `rtol`, `factor`, `lr`, `min_scale`, or a full train.toml path
(`process.train.scheduler.patience`). It is validated like `--edit` and saved to `proposal.toml`.

**Stopping.** Never `scancel` or `jf job cancel` a train job: jobflow then marks it FAILED and the chain
breaks. Use `hamilflow-train-supervisor stop` (it writes `STOP_ROUND` or `STOP_J`, obeyed within `poll_s`,
5 min). The run is then stopped cleanly and resumed in the next round. A `STOP_ROUND` waits for each
member's first `latest` checkpoint, so that no member is sent back to its round's start; `STOP_J` stops
member J at once. `finish` ends the cycle after the current round.

**Workers.** `train_worker` (GPU, runs `deeph-train`) and `supervise_worker` (light) are both required.
The light worker runs on the login node: it must see the cycle dir, and the `jf runner` must stay up. Its
jobs only parse logs and write small TOML/JSON files, in seconds. DeepH lives in its own venv:
`train_command` is a template like `["bash", "-c", "source /path/deeph-venv/bin/activate && exec deeph-train
{toml}"]`, started with the hamilflow venv's `PYTHONPATH`, `PYTHONHOME` and `VIRTUAL_ENV` removed.

**Walltime.** No SLURM signal reaches the job under jobflow-remote. The watchdog's own timer, reading the
end time from SLURM, stops `deeph-train` `walltime_margin` (900 s) before the time limit. Give the train
jobs a time limit well above 900 s plus the start-up (graph loading and JIT take minutes). Keep
`latest_interval` small (default 10 epochs): a stop loses at most that many epochs.

**Policy.** Each action is `auto` (act), `ask` (end the round with a `proposal.toml`; `review`, then
`approve`) or `off`:

| action | default | `off` means |
|---|---|---|
| `resume` (walltime/stop: from `latest`, optimizer kept) | auto | the member is done |
| `stop_converged` | auto | keep training |
| `retune` (pooled patience/cooldown: from `best`, new optimizer, LR carried over) | ask | plain resume |
| `divergence` (val > 10 × best for 5 epochs: from `best` at half the LR) | ask | give up (resuming is pointless) |
| `stop_overfit` | ask | keep training |
| `finished_not_converged` (ended at min scale or `max_epoch`) | ask | done |

A non-finite loss always stops the run, whatever the policy; restarting it follows `divergence`.

**Convergence and floors.** Scales are relative to the cycle's first initial LR. A member is converged when
its last `conv_stages` (2) completed reduced-LR stages each improved the best val by less than `conv_tol`
(2 %), **or** when its scale is ≤ `done_scale`. The training floor `min_scale` is read from your
`min_learning_rate_scale` and kept through every retune and restart.

> **Note:** the default `done_scale = 1e-3` stops members well before DeepH's own floor
> (`min_learning_rate_scale`, e.g. 1e-7). Lower it to train further down, or raise it to stop earlier.

**Resumes.** A plain resume loads `latest` with the optimizer kept, so the epoch numbers continue. A retune
or a divergence restart loads `best` with a new optimizer and the LR carried over, so the epochs restart at
1. Epochs the resumed model does not descend from (after the loaded checkpoint, or a crashed run started
again) are dead branches: the analysis leaves them out and the reports say so ("left out").

**Files.**

```
CYCLE_DIR/
  dataset_split_J.json
  train_J/train_rN.toml, train_J/outputs_rN/<timestamp>/{deepx.log, model/}
  supervisor/state.json                       the single source of truth (written atomically)
  supervisor/round_N/{report.txt, decision.json, proposal.toml, watchdog_J.log, alerts/, STOP_*}
  supervisor/summary.json, supervisor/FINISH
```

Do not edit `state.json` by hand while a round runs: the supervise job owns it; use the CLI.

**Recovery.** `status --jobs` shows the jobflow-remote states. A FAILED supervise job: fix the cause, then
`jf job rerun <id>`; it decides the round again, once. A REMOTE_ERROR job (files not transferred):
`jf job retry <id>`. A failed `approve` restores the state and can be run again. If `approve` was interrupted
during or before the submission, `approve --resubmit` submits the round again, after checking in the DB that
no job of that round exists.

**Inference.** `summary --infer-tomls overrides.json` writes `train_J/infer/infer.toml` for each member,
from the run holding the lineage's best validation loss, or failing that, the latest run with a complete
`best` checkpoint.

**Next AL cycle.** A new cycle starts fresh chains. Members of a supervised cycle end in different rounds,
so warm-start from each member's final run: `make(cycle_dir, continue_from=warm_start_from(previous_cycle_dir))`
(best params, new optimizer, new training data). `make(cycle_dir, previous_cycle_dir=..., previous_round=N)`
only works when every member ended in round N.

### Other utilities

- `hamilflow.band_structures`: k-path setup, band calculation and plotting, gap and shift analysis.
- `hamilflow.training`: writes DeepH training and inference TOML configs from dataclasses; analyzes training logs (LR reductions, convergence, `patience` and `cooldown` suggestions). Supervised training: see above.

## Testing

```bash
pip install -e .[test]
pytest
```

The suite takes about 15 seconds and little memory: it builds small synthetic structures on the fly (see
`tests/conftest.py`), so it needs no DFT data, no FHI-aims and no GPU. `tests/` mirrors the package.

```bash
pytest tests/projection                  # one part of the package
pytest -m "not slow"                     # skip the slowest tests
pytest --cov=hamilflow --cov-report=term # with coverage
```

Tests that need a GPU are marked `gpu` and skipped when none is available. GitHub Actions runs `ruff check .` and
`pytest` on every push; its runners have no GPU, so the GPU code paths have to be checked by hand on a machine with one.
`dft_workflows` is only lightly tested so far.
