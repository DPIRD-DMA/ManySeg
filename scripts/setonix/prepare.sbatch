#!/bin/bash -l
# Build a dataset cache on Setonix from raw/ already on scratch: prepare onto the node's NVMe
# /tmp (sequential writes; CPU nodes' /tmp is RAM-backed and too small for e.g. CloudSEN12+'s
# 191 GB), then one sequential copy to $ROOT/data/cache. Uses a GPU node for its NVMe only.
# Usage: sbatch prepare.sbatch <dataset> <code_sha>
#SBATCH --account=$PROJECT-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --job-name=manyseg-prepare
#SBATCH --output=$MANYSEG_ROOT/logs/%x_%j.out
set -eu
DS=$1; CODE=$2
ROOT=$MANYSEG_ROOT
module load pytorch/2.7.1-rocm6.3.3
cd "$ROOT/code/$CODE"
srun -N 1 -n 1 -c 8 singularity exec "$SINGULARITY_CONTAINER" bash -c "
  set -eu
  source $ROOT/venv/bin/activate
  export PYTHONPATH=$ROOT/code/$CODE/packages/manyseg/src:$ROOT/code/$CODE/packages/manyseg-bench/src
  mkdir -p /tmp/d/cache && ln -s $ROOT/data/raw /tmp/d/raw
  CACHE=\$(python -c \"from manyseg_bench.datasets import get_dataset as g; d = g('$DS'); print(getattr(d, 'CACHE_NAME', d.INFO.name))\")
  t0=\$(date +%s)
  MANYSEG_BENCH_DATA=/tmp/d python -m manyseg_bench.cli prepare $DS
  echo \"prepared in \$(( \$(date +%s) - t0 )) s: \$(du -sh /tmp/d/cache/\$CACHE | cut -f1)\"
  rm -rf $ROOT/data/cache/\$CACHE.part
  cp -r /tmp/d/cache/\$CACHE $ROOT/data/cache/\$CACHE.part
  mv $ROOT/data/cache/\$CACHE.part $ROOT/data/cache/\$CACHE
  echo \"copied in \$(( \$(date +%s) - t0 )) s total\"
"
