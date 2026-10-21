#!/usr/bin/env bash
# E025: ManySeg vs SMP training-step timing grid. STATE=before|after names the output file.
set -u
D=experiments/E025_smp_training_speed
OUT=$D/runs/${STATE:?set STATE=before or after}.jsonl
mkdir -p $D/runs
for bb in edgenext_small convnextv2_femto; do for size in 509 512; do for c in "" "--compile"; do
  for lib in manyseg smp; do
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)
    uv run python $D/bench.py --lib $lib --backbone $bb --size $size $c --tag "$gpu" 2>&1 \
      | grep '^{' >> "$OUT" || echo "FAILED $lib $bb $size $c"
    echo "done $lib $bb $size $c"
  done; done; done; done
echo ALL DONE
