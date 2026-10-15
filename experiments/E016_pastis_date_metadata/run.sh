#!/usr/bin/env bash
# E016 (registered), PASTIS single-date crop types, local RTX 4090.
# Q1: does the acquisition day of year help (FiLM / channels, two shuffled-date controls)?
# Q2: do Sentinel-2 resolution band groups with a shared encoder beat the stack (E015 replication)?
# 6 arms x 3 seeds = 18 runs; e016-none is the shared baseline of Q1 and Q2.
set -u
export MANYSEG_BENCH_DATA="${MANYSEG_BENCH_DATA:-$MANYSEG_BENCH_DATA}"
RECIPE=${RECIPE:-fastai_like}
OUT=${OUT_ROOT:-.}/experiments/E016_pastis_date_metadata/runs
SET='{"epochs": 30, "batch_size": 16}'
CHANNELS='{"input_options": {"meta": {"conditioning": "channels"}}}'
# PASTIS band order (wavelength order, B01/B09/B10 removed): B2 B3 B4 B5 B6 B7 B8 B8A B11 B12
G='"band_groups": {"s2_10m": {"bands": [0, 1, 2, 6], "factor": 1}, "s2_20m": {"bands": [3, 4, 5, 7, 8, 9], "factor": 2}}, "fusion": {"type": "intermediate", "op": "concat"}'
SHARED='"band_group_inputs": {"s2_20m": {"backbone": "shared:s2_10m"}}'
# EMIT_TASKS=<file>: append each run's manyseg-bench arguments (shell-quoted, one run per line)
# instead of running it.
mb() {
  if [ -n "${EMIT_TASKS:-}" ]; then printf '%q ' "$@" >> "$EMIT_TASKS"; echo >> "$EMIT_TASKS"
  else uv run manyseg-bench "$@"; fi
}
FAILS=0
run() { echo "=== $*"; mb train --provider manyseg --arch unet --encoder resnet34 \
          --weights tv_in1k --recipe "$RECIPE" --set "$SET" --out "$OUT" --evaluate-test "$@" \
          || { echo "FAILED $*"; FAILS=$((FAILS + 1)); }
        # Pre-registered stopping rule: abort on a systemic failure (more than 2 failed runs).
        if [ "$FAILS" -gt 2 ]; then echo "ABORT: $FAILS failed runs"; exit 1; fi; }
for seed in 0 1 2; do
  S="--seeds $seed"
  run $S --dataset pastis --tag e016-none                         # S2 only (stack)
  run $S --dataset pastis_meta --input-mode dict --tag e016-film  # doy, FiLM (default)
  run $S --dataset pastis_meta --input-mode dict --options "$CHANNELS" --tag e016-channels
  run $S --dataset pastis_meta_shuffled --input-mode dict --tag e016-film-shuffled
  run $S --dataset pastis_meta_tileshuffled --input-mode dict --tag e016-film-tileshuffled
  run $S --dataset pastis --options "{$G, $SHARED}" --tag e016-groups-shared  # Q2
done
echo "ALL DONE"
