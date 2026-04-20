#!/usr/bin/env bash
set -euo pipefail

# Example Slurm header:
# #SBATCH --job-name=band-structure
# #SBATCH --array=1-100
# #SBATCH --cpus-per-task=1
# #SBATCH --mem=4G

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

PARENT_DIR=${1:?Usage: $0 PARENT_DIR [STRUCTURE_FILENAME] [OUTPUT_FILENAME]}
STRUCTURE_FILENAME=${2:-POSCAR}
OUTPUT_FILENAME=${3:-band_data.h5}

cd "$REPO_ROOT"

mapfile -t CASE_DIRS < <(find "$PARENT_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

if [[ ${#CASE_DIRS[@]} -eq 0 ]]; then
  echo "No case directories found under: $PARENT_DIR" >&2
  exit 1
fi

if [[ -z ${SLURM_ARRAY_TASK_ID:-} ]]; then
  echo "SLURM_ARRAY_TASK_ID is not set. Submit this script as a Slurm array job." >&2
  exit 1
fi

TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
if (( TASK_INDEX < 0 || TASK_INDEX >= ${#CASE_DIRS[@]} )); then
  echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID is out of range for ${#CASE_DIRS[@]} case directories." >&2
  exit 1
fi

CASE_DIR=${CASE_DIRS[$TASK_INDEX]}

python "$SCRIPT_DIR/band_structure_single_task.py" "$CASE_DIR" \
  --structure-filename "$STRUCTURE_FILENAME" \
  --output-filename "$OUTPUT_FILENAME"
