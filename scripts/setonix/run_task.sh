#!/bin/bash
# Runs line $2 of task file $1 (manyseg-bench train arguments, shell-quoted) inside the
# PyTorch container, using the ManySeg venv on scratch.
set -eu
TASKS=$1; IDX=$2
ROOT=$MANYSEG_ROOT
source "$ROOT/venv/bin/activate"
# Import manyseg / manyseg_bench from *this* job's code snapshot, not whatever snapshot the
# venv's editable install happens to point at.
CODE_DIR=$(pwd)
export PYTHONPATH="$CODE_DIR/packages/manyseg/src:$CODE_DIR/packages/manyseg-bench/src${PYTHONPATH:+:$PYTHONPATH}"
LINE=$(sed -n "${IDX}p" "$TASKS")
[ -n "$LINE" ] || { echo "no task $IDX in $TASKS"; exit 2; }
echo "task $IDX: manyseg-bench $LINE"
# Memory-mapped reads on Lustre page-fault at ~0.4 MB/s (E011, 2026-10-14). Stage the dataset's
# cache onto node-local NVMe with one sequential copy and point the bench at the local copy.
DS=$(python - "$LINE" <<'PY'
import shlex, sys
args = shlex.split(sys.argv[1])
name = args[args.index("--dataset") + 1] if "--dataset" in args else "oxford_pet"
from manyseg_bench.datasets import get_dataset
ds = get_dataset(name)
print(getattr(ds, "CACHE_NAME", ds.INFO.name), *(ds.INFO.inputs or {}))
PY
)
read -r DS INPUTS <<< "$DS"
t0=$(date +%s)
if [ -n "${MANYSEG_SHARED_STAGE:-}" ]; then
  # Pack jobs (manyseg_pack.sbatch): the job's steps share one staged copy in the job-private
  # /tmp (Slurm deletes it at job end). Only the files this dataset variant reads are copied
  # (e.g. CloudSEN12 S2-only skips the 34 GB of S1GBM/DEM); the first step copies, the others
  # wait on the lock and reuse it.
  LOCAL=$MANYSEG_SHARED_STAGE
  mkdir -p "$LOCAL/cache/$DS"
  exec 9>"$LOCAL/stage.lock"
  flock 9
  SRC="$MANYSEG_BENCH_DATA/cache/$DS"
  WANT=("$SRC"/*.json "$SRC"/*_masks.npy)
  if [ -n "$INPUTS" ]; then
    for k in $INPUTS; do WANT+=("$SRC"/*_images."$k".npy); done
  else
    WANT+=("$SRC"/*_images*.npy)
  fi
  for f in "${WANT[@]}"; do
    dst="$LOCAL/cache/$DS/$(basename "$f")"
    [ -f "$dst" ] || { cp "$f" "$dst.part" && mv "$dst.part" "$dst"; }
  done
  flock -u 9
else
  LOCAL="/tmp/manyseg-${SLURM_JOB_ID:-x}-${IDX}"
  mkdir -p "$LOCAL/cache"
  trap 'rm -rf -- "/tmp/manyseg-${SLURM_JOB_ID:-x}-${IDX}"' EXIT
  cp -r "$MANYSEG_BENCH_DATA/cache/$DS" "$LOCAL/cache/"
fi
[ -d "$MANYSEG_BENCH_DATA/raw/$DS" ] && { ln -sfn "$MANYSEG_BENCH_DATA/raw" "$LOCAL/raw" || true; }
echo "staged cache/$DS to $LOCAL in $(( $(date +%s) - t0 )) s ($(du -sh "$LOCAL/cache/$DS" | cut -f1))"
export MANYSEG_BENCH_DATA="$LOCAL"
python -c "import torch; print('torch', torch.__version__, 'gpu', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python -c "import manyseg, manyseg_bench; print('code', manyseg.__file__)"
eval "python -m manyseg_bench.cli $LINE"
