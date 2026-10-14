#!/usr/bin/env bash
set -u
export MANYSEG_BENCH_DATA="$MANYSEG_BENCH_DATA"
RECIPE=${RECIPE:-fastai_like}
OUT=${OUT_ROOT:-.}/experiments/E011_tree_points/runs
SET='{"epochs": 30, "batch_size": 8}'
run() { echo "=== $*"; uv run manyseg-bench train --provider manyseg --dataset oam_tcd \
          --recipe "$RECIPE" --set "$SET" --out "$OUT" --evaluate-test "$@" || echo "FAILED $*"; }
for seed in 0 1 2; do
  run --seeds $seed --tag hm_mse --task-kwargs '{"heatmap": "regression"}'
  run --seeds $seed --tag hm_focal --task-kwargs '{"heatmap": "segmentation"}'
  run --seeds $seed --tag density --task-kwargs '{"density": true}'
done
echo "ALL DONE"
