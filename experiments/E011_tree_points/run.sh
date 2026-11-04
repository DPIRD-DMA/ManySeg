#!/usr/bin/env bash
set -u
export MANYSEG_BENCH_DATA="${MANYSEG_BENCH_DATA:?set to the dataset root}"
RECIPE=${RECIPE:-fastai_like}
OUT=${OUT_ROOT:-.}/experiments/E011_tree_points/runs
SET='{"epochs": 30, "batch_size": 8}'
# EMIT_TASKS=<file>: append each run's manyseg-bench arguments (shell-quoted, one run per line)
# instead of running it; used to submit the same arms as Slurm arrays (scripts/setonix/).
mb() {
  if [ -n "${EMIT_TASKS:-}" ]; then printf '%q ' "$@" >> "$EMIT_TASKS"; echo >> "$EMIT_TASKS"
  else uv run manyseg-bench "$@"; fi
}
run() { echo "=== $*"; mb train --provider manyseg --dataset oam_tcd \
          --recipe "$RECIPE" --set "$SET" --out "$OUT" --evaluate-test "$@" || echo "FAILED $*"; }
for seed in 0 1 2; do
  run --seeds $seed --tag hm_mse --task-kwargs '{"heatmap": "regression"}'
  run --seeds $seed --tag hm_focal --task-kwargs '{"heatmap": "segmentation"}'
  run --seeds $seed --tag density --task-kwargs '{"density": true, "density_activation": null}'
done
echo "ALL DONE"
