#!/usr/bin/env bash
set -u
export MANYSEG_BENCH_DATA="$MANYSEG_BENCH_DATA"
RECIPE=${RECIPE:-fastai_like}
OUT=${OUT_ROOT:-.}/experiments/E010_instance_heads/runs
SET='{"epochs": 40, "batch_size": 16}'
run() { echo "=== $*"; uv run manyseg-bench train --provider manyseg --dataset ai4boundaries \
          --recipe "$RECIPE" --set "$SET" --out "$OUT" --evaluate-test "$@" || echo "FAILED $*"; }
for seed in 0 1 2; do
  run --seeds $seed --tag extent_cc --task-kwargs '{"mode": "extent_cc"}'
  run --seeds $seed --tag bd_shared --task-kwargs '{"mode": "boundary_distance"}'
  run --seeds $seed --tag bd_separate --task-kwargs '{"mode": "boundary_distance", "decoders": "separate"}'
done
echo "ALL DONE"
