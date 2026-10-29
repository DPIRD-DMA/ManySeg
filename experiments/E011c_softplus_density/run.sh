#!/usr/bin/env bash
# E011c: plain vs softplus density head on OAM-TCD (E011 density-arm protocol).
# PREFLIGHT=1: one epoch of each arm (seed 0, models saved for the visual check) instead.
set -u
export MANYSEG_BENCH_DATA="$MANYSEG_BENCH_DATA"
RECIPE=${RECIPE:-fastai_like}
EXP=${OUT_ROOT:-.}/experiments/E011c_softplus_density
PLAIN='{"density": true}'
SOFT='{"density": true, "density_activation": "softplus"}'
mb() { uv run manyseg-bench "$@"; }
if [ -n "${PREFLIGHT:-}" ]; then
  for arm in density density_softplus; do
    kw=$PLAIN; [ "$arm" = density_softplus ] && kw=$SOFT
    mb train --provider manyseg --dataset oam_tcd --recipe "$RECIPE" \
      --set '{"epochs": 1, "batch_size": 8}' --out "$EXP/preflight" --seeds 0 \
      --tag "preflight_$arm" --task-kwargs "$kw" --save-model || { echo "PREFLIGHT FAILED"; exit 1; }
  done
  echo "PREFLIGHT DONE"; exit 0
fi
SET='{"epochs": 30, "batch_size": 8}'
for seed in 0 1 2; do
  for arm in density density_softplus; do
    kw=$PLAIN; [ "$arm" = density_softplus ] && kw=$SOFT
    echo "=== seed $seed $arm ($(date))"
    mb train --provider manyseg --dataset oam_tcd --recipe "$RECIPE" --set "$SET" \
      --out "$EXP/runs" --evaluate-test --seeds $seed --tag "$arm" --task-kwargs "$kw" \
      || echo "FAILED seed $seed $arm"
  done
done
echo "ALL DONE"
