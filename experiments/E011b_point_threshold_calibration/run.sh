#!/usr/bin/env bash
set -u
export MANYSEG_BENCH_DATA="${MANYSEG_BENCH_DATA:?set to the dataset root}"
RECIPE=${RECIPE:-fastai_like}
OUT=${OUT_ROOT:-.}/experiments/E011b_point_threshold_calibration/runs
SET='{"epochs": 30, "batch_size": 8}'
# EMIT_TASKS=<file>: append each run's manyseg-bench arguments instead of running (Setonix).
mb() {
  if [ -n "${EMIT_TASKS:-}" ]; then printf '%q ' "$@" >> "$EMIT_TASKS"; echo >> "$EMIT_TASKS"
  else uv run manyseg-bench "$@"; fi
}
for seed in 0 1 2; do
  mb train --provider manyseg --dataset oam_tcd --recipe "$RECIPE" --set "$SET" --out "$OUT" \
    --evaluate-test --seeds $seed --tag hm_mse_calibrated \
    --task-kwargs '{"heatmap": "regression", "calibrate": true}' || echo "FAILED $seed"
done
echo "ALL DONE"
