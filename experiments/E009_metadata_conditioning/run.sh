#!/usr/bin/env bash
set -u
export MANYSEG_BENCH_DATA="${MANYSEG_BENCH_DATA:?set to the dataset root}"
RECIPE=${RECIPE:-fastai_like}
OUT=${OUT_ROOT:-.}/experiments/E009_metadata_conditioning/runs
SET='{"epochs": 60, "batch_size": 16}'
FUSION='"fusion": {"type": "intermediate", "op": "concat"}'
# EMIT_TASKS=<file>: append each run's manyseg-bench arguments (shell-quoted, one run per line)
# instead of running it; used to submit the same arms as Slurm arrays (scripts/setonix/).
mb() {
  if [ -n "${EMIT_TASKS:-}" ]; then printf '%q ' "$@" >> "$EMIT_TASKS"; echo >> "$EMIT_TASKS"
  else uv run manyseg-bench "$@"; fi
}
run() { echo "=== $*"; mb train --provider manyseg --input-mode dict \
          --recipe "$RECIPE" --set "$SET" --out "$OUT" --evaluate-test "$@" || echo "FAILED $*"; }
for fold in 0 1 2 3 4; do
  for seed in 0 1 2; do
    S="--split events:5:$fold --seeds $seed"
    run $S --dataset sen1floods11 --tag none-f$fold --options "{$FUSION}"
    run $S --dataset sen1floods11_meta --tag film-f$fold --options "{$FUSION}"
    run $S --dataset sen1floods11_meta --tag channels-f$fold \
        --options "{$FUSION, \"input_options\": {\"meta\": {\"conditioning\": \"channels\"}}}"
    run $S --dataset sen1floods11_meta_nogeo --tag film_nogeo-f$fold --options "{$FUSION}"
  done
done
for seed in 0 1 2; do
  run --seeds $seed --dataset sen1floods11 --tag none_official --options "{$FUSION}"
  run --seeds $seed --dataset sen1floods11_meta --tag film_official --options "{$FUSION}"
done
echo "ALL DONE"
