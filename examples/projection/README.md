# Projection Batch Examples

This folder contains scripts to run hamilflow projection over many independent subfolders.

## Expected Input Layout

Given a parent input directory, each case should be a subdirectory (default pattern: structure_*):

```text
parent_input/
  structure_0/
  structure_1/
  ...
  structure_N/
```

Each structure directory must contain the files required by `hamilflow.run_projection`.

## Scripts

- `run_projection_batch.py`: local multiprocessing batch run in one Python process.
- `run_projection_single_task.py`: run exactly one case (picked by `--index` or `SLURM_ARRAY_TASK_ID`).
- `run_projection_submit_slurm.sh`: submit a Slurm array job with one case per array task.
- `run_projection_array.sbatch`: worker script used by Slurm array tasks.

## 1) Local Batch Run (One Node, Multiple Workers)

Run all matching subdirectories from one command:

```bash
python examples/projection/run_projection_batch.py \
  --path /path/to/parent_input \
  --output-directory-parent /path/to/parent_output \
  --removal-plan /path/to/removal_plan.json \
  --reduction-mode schur \
  --max-workers 16
```

Notes:
- `--max-workers` should usually match available CPU cores.
- Output for `structure_k` is written to `/path/to/parent_output/structure_k`.
- `--reduction-mode` can be `schur` or `truncate`.

## 2) Single Task Run (Manual Debug)

Run one case by index (1-based):

```bash
python examples/projection/run_projection_single_task.py \
  --path /path/to/parent_input \
  --output-directory-parent /path/to/parent_output \
  --removal-plan /path/to/removal_plan.json \
  --reduction-mode truncate \
  --index 3
```

This is useful to debug one failing case before launching full batch jobs.

## 3) Slurm Array Submission (Recommended for Many Independent Cases)

Submit one array job where each task handles one structure directory:

```bash
    bash examples/projection/run_projection_submit_slurm.sh \
    /path/to/parent_input \
    /path/to/parent_output \
    /path/to/removal_plan.json \
    "structure_*" \
    1 \
    --reduction-mode schur \
  --account my_account \
  --partition cpu \
  --time 04:00:00 \
  --mem 16G
```

Arguments for submit script:
1. Parent input directory.
2. Parent output directory.
3. Removal plan JSON path.
4. Optional folder glob pattern (default: `structure_*`).
5. Optional CPUs per Slurm task (default: `1`).

Additional submit options:
- `--reduction-mode schur|truncate` (default: `schur`)
- `--job-name NAME`
- `--account ACCOUNT`
- `--partition PARTITION`
- `--qos QOS`
- `--time HH:MM:SS`
- `--mem SIZE` (example: `8G`)
- `--constraint VALUE`
- `--sbatch-arg ARG` (repeatable; pass any extra raw sbatch argument)

What happens:
- The submit script counts matching subdirectories.
- It submits `--array=1-N` where `N` is the count.
- Each array task runs `run_projection_single_task.py` and processes exactly one subdirectory.

## Quick Recommendation

- Use `run_projection_batch.py` for local runs or one-node multiprocessing.
- Use `run_projection_submit_slurm.sh` for cluster execution with independent per-case tasks.
