#!/bin/bash
# CPU oracle experiments (E039/E040/...) on a Setonix work node.
#   sbatch --time=01:00:00 cpu_oracle.sbatch <code dir under $ROOT/code> <experiment dir> "<runner args>"
# e.g. sbatch cpu_oracle.sbatch vec-abc123 experiments/E039_unified_roads "run39.py tune --procs 120"
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=200G
#SBATCH --job-name=vec-oracle
set -euo pipefail
ROOT="${MANYSEG_ROOT:?set MANYSEG_ROOT to the working directory on scratch}"
CODE="$ROOT/code/$1"; EXP="$2"; ARGS="$3"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export E028_DATA="$ROOT/data/vectors/osm_e028" E031_DATA="$ROOT/data/vectors/3dbag"
export E028_BEST="$ROOT/data/vectors/e028_best.json"
export PYTHONPATH="$CODE/packages/manyseg/src:$CODE/packages/manyseg-bench/src:${PYTHONPATH:-}"
cd "$CODE/$EXP"
echo "start $(date) on $(hostname): $ARGS"
eval "$ROOT/venv-cpu/bin/python $ARGS"
echo "done $(date)"
