#!/usr/bin/env bash
# E026 follow-up: reruns after two harness fixes (checkpointing with timm's in-place ReLUs;
# CloudSEN12 ignore labels in the fetch benchmark). Waits for part 2.
set -u
MAIN="${MANYSEG_REPO:?set to the main checkout}"
D="experiments/E026_perf_audit"
R="$MAIN/$D/runs"
until [ -f "$MAIN/experiments/e026_rest.log.exit" ]; do sleep 20; done
for c in enc enc,dec; do
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)
  uv run python "$D/model_bench.py" --model unet_r34 --ckpt $c --tag "$gpu" 2>&1 | grep '^{' \
    >> "$R/models.jsonl" || echo "FAILED ckpt $c"
  echo "done ckpt $c"
done
for rep in 0 1; do for v in seq prefetch; do
  uv run python "$D/fetch_bench.py" --variant $v --seed $rep 2>&1 | grep '^{' >> "$R/fetch.jsonl" \
    || echo "FAILED fetch $v"
  echo "done fetch $v rep $rep"
done; done
echo ALL DONE
