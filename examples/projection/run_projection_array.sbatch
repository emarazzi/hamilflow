#!/usr/bin/env bash
set -euo pipefail

# This script is executed by sbatch as a Slurm array job.
# Positional args:
#   1) parent input directory
#   2) parent output directory
#   3) removal plan file
#   4) structure pattern (optional, default: structure_*)
#   5) reduction mode (optional, default: schur)

PARENT_DIR=${1:?Missing PARENT_DIR}
OUTPUT_PARENT_DIR=${2:?Missing OUTPUT_PARENT_DIR}
REMOVAL_PLAN=${3:?Missing REMOVAL_PLAN}
PATTERN=${4:-structure_*}
REDUCTION_MODE=${5:-schur}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

python "$SCRIPT_DIR/run_projection_single_task.py" \
  --path "$PARENT_DIR" \
  --output-directory-parent "$OUTPUT_PARENT_DIR" \
  --removal-plan "$REMOVAL_PLAN" \
  --pattern "$PATTERN" \
  --reduction-mode "$REDUCTION_MODE"
