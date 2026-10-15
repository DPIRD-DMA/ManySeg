#!/usr/bin/env bash
# E008: STAGE=train (6 checkpointed runs) then STAGE=evaluate (each run under 6 variants).
set -u
export MANYSEG_BENCH_DATA="${MANYSEG_BENCH_DATA:-$MANYSEG_BENCH_DATA}"
RECIPE=${RECIPE:-fastai_like}
STAGE=${STAGE:-train}
E=${OUT_ROOT:-.}/experiments/E008_precision_accuracy
if [ "$STAGE" = train ]; then
  for seed in 0 1 2; do
    uv run manyseg-bench train --provider manyseg --dataset oxford_pet --encoder resnet34 \
      --weights tv_in1k --arch unet --recipe "$RECIPE" --set '{"epochs": 20}' --seeds "$seed" \
      --out "$E/runs" --tag e008-pet --evaluate-test --save-model || echo "FAILED pet $seed"
    uv run manyseg-bench train --provider manyseg --dataset sen1floods11 --encoder resnet34 \
      --weights tv_in1k --arch unet --recipe "$RECIPE" --set '{"epochs": 60, "batch_size": 16}' \
      --seeds "$seed" --out "$E/runs" --tag e008-s1f11 --evaluate-test --save-model \
      || echo "FAILED s1f11 $seed"
  done
else
  for run in "$E"/runs/*_manyseg_unet_resnet34_s*.json; do
    CUDA_VISIBLE_DEVICES="${GPU_FOR_EVAL-0}" uv run manyseg-bench evaluate "$run" \
      --variants fp32,bf16,fp16 --device cuda --out "$E/runs/eval" || echo "FAILED gpu $run"
    CUDA_VISIBLE_DEVICES="" uv run manyseg-bench evaluate "$run" --device cpu --threads 4 \
      --variants int8_pt2e,int8_pt2e_compiled,int8_weight_only --out "$E/runs/eval" \
      || echo "FAILED cpu $run"
  done
fi
echo "ALL DONE"
