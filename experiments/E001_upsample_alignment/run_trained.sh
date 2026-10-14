#!/usr/bin/env bash
# E001b: does grid-consistent ("aligned") upsampling change trained accuracy / boundary F1?
set -u
export MANYSEG_BENCH_DATA="$MANYSEG_BENCH_DATA"
RECIPE=${RECIPE:?set RECIPE}
OUT=${OUT_ROOT:-.}/experiments/E001_upsample_alignment/runs
for seed in 0 1 2; do
  for up in bilinear aligned; do  # nearest baseline = E003 unet runs (identical config)
    echo "=== upsample=$up seed=$seed recipe=$RECIPE"
    uv run manyseg-bench train --provider manyseg --arch unet --recipe $RECIPE --seeds $seed \
      --options "{\"decoder_kwargs\": {\"upsample\": \"$up\"}}" --set '{"epochs": 20}' \
      --out $OUT --evaluate-test --tag "e001b-$up" || echo "FAILED $up $seed"
  done
done
echo "ALL DONE"
