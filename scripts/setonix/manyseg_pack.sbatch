#!/bin/bash -l
# Slurm array of "packs": array element a runs tasks (a-1)*N+1 .. a*N of the task file in
# parallel, one MI250X GCD each, sharing one staged dataset copy in the job's private /tmp.
# For datasets too large to copy once per task (CloudSEN12+: 157-191 GB). Each job has its own
# /tmp (job_container/tmpfs on /nvmetmp), so array *jobs* cannot share a staged copy; the steps
# of one job can, and the shared files also share the page cache.
# Steps are named t<task index>, so `sacct -j <job> -o JobID,JobName` maps a step to its task
# (step ids follow launch timing, not task order). Usage: sbatch --array=1-M [--gres=gpu:N,tmp:400G --ntasks=N] [--time=HH:MM:SS] manyseg_pack.sbatch <tasks.txt> <code_sha> <N>
# Node-local NVMe is a GRES on Setonix (tmp:3500G per node) and must be requested explicitly;
# CPUs must not be (each GPU brings 8 cores; the cli_filter rejects --cpus-per-task).
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8,tmp:400G
#SBATCH --ntasks=8
#SBATCH --time=06:00:00
#SBATCH --job-name=manyseg-pack
set -eu
TASKS=$1; CODE=$2; N=$3
ROOT="${MANYSEG_ROOT:?set MANYSEG_ROOT to the working directory on scratch}"
module load pytorch/2.7.1-rocm6.3.3
export MANYSEG_BENCH_DATA=$ROOT/data OMP_NUM_THREADS=8 MIOPEN_FIND_MODE=FAST
export HF_HOME=$ROOT/hf HF_HUB_OFFLINE=1 MANYSEG_CODE_SHA=$CODE
export MANYSEG_SHARED_STAGE=/tmp/manyseg-stage
cd "$ROOT/code/$CODE"
TOTAL=$(wc -l < "$TASKS")
START=$(( (SLURM_ARRAY_TASK_ID - 1) * N + 1 ))
pids=()
for (( i = 0; i < N; i++ )); do
  IDX=$(( START + i ))
  [ "$IDX" -le "$TOTAL" ] || break
  DB=/tmp/miopen-$IDX; mkdir -p "$DB"
  MIOPEN_USER_DB_PATH=$DB MIOPEN_CUSTOM_CACHE_DIR=$DB \
    srun --exact --job-name="t$IDX" -N 1 -n 1 -c 8 --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest \
    singularity exec "$SINGULARITY_CONTAINER" \
    bash "$ROOT/code/$CODE/scripts/setonix/run_task.sh" "$TASKS" "$IDX" \
    > "$ROOT/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_t${IDX}.out" 2>&1 &
  pids+=("$!")
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$(( fail + 1 )); done
echo "pack $SLURM_ARRAY_TASK_ID: ${#pids[@]} tasks, $fail failed"
[ "$fail" -eq 0 ]
