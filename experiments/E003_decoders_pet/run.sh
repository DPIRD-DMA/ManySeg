#!/usr/bin/env bash
set -u
export MANYSEG_BENCH_DATA="$MANYSEG_BENCH_DATA"
RECIPE=${RECIPE:?set RECIPE}
OUT=${OUT_ROOT:-.}/experiments/E003_decoders_pet/runs
for seed in 0 1 2; do
  for dec in unet fastai_unet fpn deeplabv3plus upernet segformer; do
    echo "=== decoder=$dec seed=$seed recipe=$RECIPE"
    uv run manyseg-bench train --provider manyseg --arch $dec --recipe $RECIPE --seeds $seed \
      --set '{"epochs": 20}' --out $OUT --evaluate-test --tag e003 || echo "FAILED $dec $seed"
  done
done
echo "ALL DONE"
