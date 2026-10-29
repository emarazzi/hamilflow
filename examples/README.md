# Examples

User-facing scripts for the hamilflow components. Each script starts with a
docstring describing what it does, and most have a "User settings" section
(paths, SLURM resources, numerical options) to edit before running. The
folders follow the usual order of a DeepH active-learning loop:

    candidate_selection -> dft_workflows -> (projection) -> training -> uncertainty -> candidate_selection
                                                                    \-> band_structure

## Logging

hamilflow reports progress (structures started/finished, files written,
scheduling and memory estimates) with the standard `logging` module, under the
`hamilflow` logger namespace (one logger per module, e.g.
`hamilflow.uncertainty.band_uncertainty`), and reports misuse of its arguments
with `warnings.warn`. Without any configuration, warnings and errors (and
`warnings.warn` messages) go to stderr, the SLURM `.err` file, while progress
messages (INFO) are not shown. The scripts here that call code which logs
configure logging near their top, printing everything from INFO up to stdout
(the SLURM `.out` file) and saving a log file:

```python
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("my_run.log")],
)
logging.captureWarnings(True)  # send warnings.warn messages to the log file too
```

Levels can be set per module, e.g. to see the per-structure details logged at
DEBUG (CPU/GPU backend choice, symlink fallbacks, the model used by each fold):

```python
logging.getLogger("hamilflow").setLevel(logging.DEBUG)
```

or to keep only warnings and errors from the band uncertainty:

```python
logging.getLogger("hamilflow.uncertainty").setLevel(logging.WARNING)
```

The ill-conditioned overlap treatment logs every orbital it removes
(`[iter N] ...`, INFO) only with `verbose=True`; the band uncertainty runs it
with `verbose=False`.

The jobs submitted by `dft_workflows/` and `projection/` run in jobflow(-remote)
workers, not in the submitting script, so a configuration in these scripts does
not reach them. Instead each hamilflow job, if logging is not configured in
the worker (no handler on the root or `hamilflow` logger), prints hamilflow's
INFO messages and captured warnings to stdout, the job's output file, in the
format above (e.g. the `[projection] Nb=... workers=...` line), for the
duration of the job: afterwards the worker's logging is as it was. The handler
is attached to the `hamilflow` logger, not to the root, so jobflow's own
messages are not printed twice.

GPU use (CuPy, see `hamilflow.sparse_hamiltonian.dense_backend`) is logged at
INFO once per calculation: by the process that sets up the GPU, or, for the
worker processes of `BandUncertaintyCalculator.compute_parallel` and of
`band_structure/band_structure_batch_example.py --workers N`, once by the
parent (the workers then log it at DEBUG). The parent learns it from
`nvidia-smi`; if that fails, `compute_parallel` warns once when the first
worker reports using the GPU (its scheduling then has no GPU memory budget),
and the batch example's workers each log it at INFO. In your own process
pools, each worker that sets up the GPU logs it at INFO; to report it once
yourself, log it in the parent and start the workers inside
`hamilflow.sparse_hamiltonian.dense_backend.gpu_use_reported()`, which makes
them log it at DEBUG (with the forkserver start method, the Linux default from
Python 3.14, only if the server starts inside it; otherwise they still log at
INFO).

## `candidate_selection/`

Pick which structures to compute with DFT.

- `select_candidates.py`: selects N representative structures from an
  extended-XYZ dataset using Average Minimum Distance (AMD) fingerprints, by
  farthest-point sampling, k-means, or a regex on the frame labels. Writes the
  selected structures (`selected.extxyz`, ready for `dft_workflows/`) and an
  interactive 2D MDS map of the dataset.
- `uncertainty_map.py`: colors that 2D map by the ensemble band uncertainty
  from `uncertainty/ensemble_band_uncertainty.py`, to see which regions of
  structure space the models are unsure about.

## `dft_workflows/`

jobflow-remote flows that go from structures to projected DeepH training or
inference inputs with FHI-aims.

- `aims_to_projected_deeph.py`: full chain, per structure: AIMS SCF ->
  AIMS-to-DeepH conversion -> Schur projection -> truncation projection.
- `aims_overlap_only_to_projected_deeph.py`: same chain for the overlap matrix
  only (short, loosely converged AIMS runs), e.g. to prepare inputs for DeepH
  inference on new structures.
- `removal_plan_schur.json`, `removal_plan_truncate.json`: example removal
  plans for MoS2 (core shells folded out with the Schur complement, then
  high-l polarisation shells truncated). Their `_comment` keys explain the
  plan format and how to adapt it to other elements and basis sets.

## `projection/`

The same two-step projection, run on DeepH-format data that is already on
disk (e.g. the `deeph_full/` folder written by `dft_workflows/`).

- `two_step_projection.py`: Schur projection followed by truncation, for H and S.
- `two_step_projection_overlap_only.py`: the same for S only.
- `removal_plan_schur.json`, `removal_plan_truncate.json`: copies of the
  example plans in `dft_workflows/`.

The scripts in `dft_workflows/` and `projection/` read the removal plans from
their `WORKDIR`, so copy the plans there and edit them for your system.

## `train_infer_config/`

DeepH inputs for a k-model ensemble, one AL cycle per folder (`cycle_dir`).
Run in this order:

- `make_kfold_splits.py`: writes `dataset_split_{1..k}.json` for training a
  k-model ensemble. Variants of the same structure family (by default, names
  that differ only after their last `-`; set `GROUP_KEY` for other naming
  schemes) are spread over different validate sets. In later AL cycles, the
  previous splits are kept unchanged and only the new structures are added.
- `write_train_tomls.py`: one `train_{j}/train.toml` per fold, with the split
  file, train/validate sizes and seed filled in per fold. With a previous
  cycle, fold j continues training from that cycle's fold-j model; with
  `CONTINUE_FROM` (e.g. `outputs_3`), it resumes its own earlier run instead.
- `write_infer_tomls.py`: once training is done, one
  `train_{j}/<infer_name>/infer.toml` per fold, pointing to the model of the
  most recent training run of that fold (with a warning if there are several).

## `uncertainty/`

Uncertainty of an ensemble of DeepH models (e.g. from k-fold training) that
ran inference on the same structures.

- `ensemble_band_uncertainty.py`: spread of the band energies across the
  ensemble, per structure and k-point. Needs no DFT reference, so it can drive
  the selection of new structures. The "projected" method diagonalizes only the
  ensemble-averaged Hamiltonian and works in a band window around the gap.
- `ensemble_error_vs_dft.py`: error of the ensemble-averaged Hamiltonian's
  bands against DFT, on structures with a reference, to check how well the
  ensemble spread follows the true error.

## `band_structure/`

Band structures along a k-path from DeepH-format Hamiltonians (DFT or
predicted).

- `band_structure_example.py`: one structure, computes and plots the bands.
- `band_structure_batch_example.py`: every `structure_*` subfolder of a parent
  directory, in parallel processes.
- `band_structure_single_task.py` + `band_structure_slurm.sh`: one structure
  per SLURM array task, for large batches on a cluster.
