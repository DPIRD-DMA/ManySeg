#!/bin/bash -l
# Slurm array over a task file: one manyseg-bench run per array element on 1 MI250X GCD.
# Usage: sbatch --array=1-N%K [--time=HH:MM:SS] manyseg_array.sbatch <tasks.txt> <code_sha>
#SBATCH --account=$PROJECT-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --job-name=manyseg
#SBATCH --output=$MANYSEG_ROOT/logs/%x_%A_%a.out
set -eu
TASKS=$1; CODE=$2
ROOT=$MANYSEG_ROOT
module load pytorch/2.7.1-rocm6.3.3
export MANYSEG_BENCH_DATA=$ROOT/data OMP_NUM_THREADS=8 MIOPEN_USER_DB_PATH=/tmp/miopen-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
# The runner sets cudnn.benchmark=True; on ROCm that triggers an exhaustive MIOpen kernel search
# per conv shape (minutes per run, repeated in every array task). FAST = heuristic selection.
export MIOPEN_FIND_MODE=FAST
# Compute nodes have no outbound internet: pretrained weights are pre-fetched on a login node
# into $ROOT/hf (see scripts/setonix/README.md); offline mode makes a missing file fail fast.
export HF_HOME=$ROOT/hf HF_HUB_OFFLINE=1
mkdir -p "$MIOPEN_USER_DB_PATH"
cd "$ROOT/code/$CODE"
srun -N 1 -n 1 -c 8 --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest \
  singularity exec "$SINGULARITY_CONTAINER" \
  bash "$ROOT/code/$CODE/scripts/setonix/run_task.sh" "$TASKS" "$SLURM_ARRAY_TASK_ID"
# Note: the module's `pytorch-shell` wrapper starts an interactive shell and silently ignores
# a script/-c argument (exit 0, no output), so we call singularity exec directly.
