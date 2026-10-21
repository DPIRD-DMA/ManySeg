#!/usr/bin/env bash
# E025 follow-up: the other decoders that used the nearest gather. STATE=before|after.
set -u
D=experiments/E025_smp_training_speed
OUT=$D/runs/decoders_${STATE:?set STATE}.jsonl
mkdir -p $D/runs
for dec in unetplusplus fpn fastai_unet; do for c in "" "--compile"; do
  libs="manyseg"; [ "$dec" != fastai_unet ] && libs="manyseg smp"
  for lib in $libs; do
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)
    uv run python $D/bench.py --lib $lib --backbone edgenext_small --decoder $dec --size 509 \
      --batch 16 $c --tag "$gpu" 2>&1 | grep '^{' >> "$OUT" || echo "FAILED $lib $dec $c"
    echo "done $lib $dec $c"
  done; done; done
echo ALL DONE
