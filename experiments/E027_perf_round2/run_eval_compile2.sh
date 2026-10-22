#!/usr/bin/env bash
# E027 follow-up, clean rerun (the first pass overlapped a CPU-heavy test run).
set -u
D="experiments/E027_perf_round2"
until [ -f experiments/e027_evalc.log.exit ]; do sleep 20; done
for rep in 1 2; do for m in unet_pvt segformer_pvt; do for v in same same_pad static_pad; do
  uv run python "$D/eval_compile_bench.py" --model $m --variant $v 2>&1 | grep '^{' \
    >> "$D/runs/eval_compile_clean.jsonl" || echo "FAILED $m $v"
  echo "done $m $v rep $rep"
done; done; done
echo ALL DONE
