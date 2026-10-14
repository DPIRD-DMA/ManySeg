#!/usr/bin/env bash
# E014: eager vs torch.compile x fp32/bf16/fp16 on an idle RTX 4090 (inference b=1,8; train b=8).
set -u
OUT=${OUT_ROOT:-.}/experiments/E014_idle_gpu_profile/runs
for enc in resnet34 convnext_tiny; do
  for dec in unet fastai_unet fpn segformer upernet; do
    for prec in fp32 bf16 fp16; do
      for comp in "" "--compile"; do
        uv run manyseg-bench profile --provider manyseg --encoder "$enc" --weights none \
          --arch "$dec" --size 512,512 --in-channels 3 --num-classes 2 --batch-sizes 1,8 \
          --train-batch-size 8 --precision "$prec" $comp --device cuda --out "$OUT" \
          --tag "e014-$dec-$prec${comp:+-compile}" || echo "FAILED $enc $dec $prec $comp"
      done
    done
  done
done
echo "ALL DONE"
