#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat >&2 <<'EOF'
Usage:
  run_projection_submit_slurm.sh PARENT_DIR OUTPUT_PARENT_DIR REMOVAL_PLAN [PATTERN] [CPUS_PER_TASK] [OPTIONS]

Options:
  --job-name NAME         Slurm job name (default: projection_batch)
  --account ACCOUNT       Slurm account
  --partition PARTITION   Slurm partition
  --qos QOS               Slurm QoS
  --time TIME             Slurm walltime (e.g., 02:00:00)
  --mem MEM               Slurm memory (e.g., 8G)
  --constraint VALUE      Slurm node constraint
  --reduction-mode MODE   Projection mode: schur|truncate (default: schur)
  --sbatch-arg ARG        Extra raw sbatch arg (repeatable)
EOF
  exit 1
fi

PARENT_DIR=$1
OUTPUT_PARENT_DIR=$2
REMOVAL_PLAN=$3
PATTERN=${4:-structure_*}
CPUS_PER_TASK=${5:-1}

shift 5 || true

JOB_NAME="projection_batch"
ACCOUNT=""
PARTITION=""
QOS=""
TIME_LIMIT=""
MEMORY=""
CONSTRAINT=""
REDUCTION_MODE="schur"
EXTRA_SBATCH_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-name)
      JOB_NAME=${2:?Missing value for --job-name}
      shift 2
      ;;
    --account)
      ACCOUNT=${2:?Missing value for --account}
      shift 2
      ;;
    --partition)
      PARTITION=${2:?Missing value for --partition}
      shift 2
      ;;
    --qos)
      QOS=${2:?Missing value for --qos}
      shift 2
      ;;
    --time)
      TIME_LIMIT=${2:?Missing value for --time}
      shift 2
      ;;
    --mem)
      MEMORY=${2:?Missing value for --mem}
      shift 2
      ;;
    --constraint)
      CONSTRAINT=${2:?Missing value for --constraint}
      shift 2
      ;;
    --reduction-mode)
      REDUCTION_MODE=${2:?Missing value for --reduction-mode}
      shift 2
      ;;
    --sbatch-arg)
      EXTRA_SBATCH_ARGS+=("${2:?Missing value for --sbatch-arg}")
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "$REDUCTION_MODE" != "schur" && "$REDUCTION_MODE" != "truncate" ]]; then
  echo "Invalid --reduction-mode: $REDUCTION_MODE (expected: schur|truncate)" >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ ! -d "$PARENT_DIR" ]]; then
  echo "Input parent directory not found: $PARENT_DIR" >&2
  exit 1
fi

if [[ ! -f "$REMOVAL_PLAN" ]]; then
  echo "Removal plan file not found: $REMOVAL_PLAN" >&2
  exit 1
fi

mapfile -t STRUCTURE_DIRS < <(find "$PARENT_DIR" -mindepth 1 -maxdepth 1 -type d -name "$PATTERN" | sort)
N=${#STRUCTURE_DIRS[@]}

if [[ $N -eq 0 ]]; then
  echo "No structure directories matching '$PATTERN' found under: $PARENT_DIR" >&2
  exit 1
fi

SBATCH_CMD=(
  sbatch
  --array="1-${N}"
  --cpus-per-task="$CPUS_PER_TASK"
  --job-name="$JOB_NAME"
)

if [[ -n "$ACCOUNT" ]]; then
  SBATCH_CMD+=(--account="$ACCOUNT")
fi
if [[ -n "$PARTITION" ]]; then
  SBATCH_CMD+=(--partition="$PARTITION")
fi
if [[ -n "$QOS" ]]; then
  SBATCH_CMD+=(--qos="$QOS")
fi
if [[ -n "$TIME_LIMIT" ]]; then
  SBATCH_CMD+=(--time="$TIME_LIMIT")
fi
if [[ -n "$MEMORY" ]]; then
  SBATCH_CMD+=(--mem="$MEMORY")
fi
if [[ -n "$CONSTRAINT" ]]; then
  SBATCH_CMD+=(--constraint="$CONSTRAINT")
fi

for arg in "${EXTRA_SBATCH_ARGS[@]}"; do
  SBATCH_CMD+=("$arg")
done

SBATCH_CMD+=(
  "$SCRIPT_DIR/run_projection_array.sbatch"
  "$PARENT_DIR"
  "$OUTPUT_PARENT_DIR"
  "$REMOVAL_PLAN"
  "$PATTERN"
  "$REDUCTION_MODE"
)

echo "Submitting ${N} tasks with reduction mode: ${REDUCTION_MODE}"
"${SBATCH_CMD[@]}"
