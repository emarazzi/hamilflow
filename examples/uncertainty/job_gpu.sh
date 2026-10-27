#!/bin/bash
# SLURM job for ensemble_band_uncertainty.py (or ensemble_error_vs_dft.py) on one GPU.
#
# The orbital removal and the dense diagonalizations run on the GPU, shared by the structures
# computed at once (one per allocated CPU); the CPUs load the Hamiltonians and do the projection.
# A few CPUs per GPU keep it busy: on an A100, up to 4 for ~800-orbital structures, 2 for larger
# ones. Needs CuPy in the environment: `pip install -e .[gpu]` installs cupy-cuda12x; on CUDA 11
# use cupy-cuda11x instead (see the README).
#
# Adapt the partition, account, time, memory, GPU type and environment to your cluster.

#SBATCH --job-name=band_unc_gpu
#SBATCH --output=slurm.%j.out
#SBATCH --error=slurm.%j.err
#SBATCH --partition=gpu
#SBATCH --account=my_account
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4        # = number of structures computed at once on the GPU
#SBATCH --mem=60G                # host memory: structures start only while their estimate fits
#SBATCH --gres=gpu:1             # e.g. gpu:A100:1 to ask for a GPU type

source /path/to/venv/bin/activate

# Stop with an error instead of silently running on the CPU if CuPy cannot use the GPU
# (wrong cupy-cudaXXx wheel, no device visible, ...). HAMILFLOW_GPU=0 forces the CPU.
export HAMILFLOW_GPU=1

# record what the job got
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import cupy; print('cupy', cupy.__version__, 'CUDA runtime', cupy.cuda.runtime.runtimeGetVersion())"

python ensemble_band_uncertainty.py
