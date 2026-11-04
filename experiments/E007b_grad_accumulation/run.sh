#!/usr/bin/env bash
# E007b (local RTX 4090): gradient accumulation vs encoder-BN freezing at batch 2.
set -u
export MANYSEG_BENCH_DATA="${MANYSEG_BENCH_DATA:?set to the dataset root}"
RECIPE=${RECIPE:-fastai_like}
OUT=${OUT_ROOT:-.}/experiments/E007b_grad_accumulation/runs
for seed in 0 1 2; do
  for ds in sen1floods11 oxford_pet; do
    ep=$([ "$ds" = oxford_pet ] && echo 20 || echo 60)
    # arm: batch accum freeze_encoder_bn tag
    for arm in "16 1 false b16" "2 1 false b2" "2 1 true b2-frozen" "2 8 false b2x8" \
               "2 8 true b2x8-frozen"; do
      read -r bs k fz tag <<< "$arm"
      uv run manyseg-bench train --provider manyseg --dataset "$ds" --encoder resnet34 \
        --weights tv_in1k --arch unet --recipe "$RECIPE" --seeds "$seed" --out "$OUT" \
        --set "{\"epochs\": $ep, \"batch_size\": $bs, \"grad_accum_steps\": $k, \"freeze_encoder_bn\": $fz}" \
        --tag "e007b-$ds-$tag" --evaluate-test || echo "FAILED $ds $tag $seed"
    done
  done
done
echo "ALL DONE"
