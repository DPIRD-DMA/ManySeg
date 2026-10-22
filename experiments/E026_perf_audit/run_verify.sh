#!/usr/bin/env bash
# E026 verification: the applied library changes (this branch, no patches) against the E026
# patched numbers; plus round-2 candidate ln_dtype (A/B). Run from the perf/e026 worktree.
set -u
D="experiments/E026_perf_audit"
OUT="$D/runs/verify.jsonl"
one() {
  local gpu
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)
  uv run python "$D/model_bench.py" "$@" --tag "$gpu" 2>&1 | grep '^{' >> "$OUT" || echo "FAILED $*"
  echo "done $*"
}
for m in fastai_r34 fastai_edgenext unet_r34_cross segformer_r34 segformer_pvt upernet_r34 \
         deeplab_r34 unet_r34_gn; do one --model $m; done
one --model xattn_r18 --batch 8
one --model unet_r34_ts --batch 8 --size 256
one --model unet_r34 --ckpt enc   # library set_grad_checkpointing via the harness hook
uv run python "$D/check.py" --patch ln_dtype 2>&1 | grep '^{' >> "$D/check.jsonl" || echo "FAILED check"
for rep in 1 2; do for m in unet_convnext unet_pvt unet_edgenext; do
  one --model $m; one --model $m --patch ln_dtype; done; done
one --model unet_vit; one --model unet_vit --patch ln_dtype
echo ALL DONE
