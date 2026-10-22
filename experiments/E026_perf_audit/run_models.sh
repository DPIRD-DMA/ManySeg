#!/usr/bin/env bash
# E026 model grid: current code vs each patch, alternating (A B A B) so drift hits both.
# Run from the frozen worktree; results go to the main tree's runs/.
set -u
MAIN="$WORKSPACE/ManySeg"
D="experiments/E026_perf_audit"
OUT="$MAIN/$D/runs/models.jsonl"
mkdir -p "$MAIN/$D/runs"

one() {  # one() <args...>
  local gpu
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)
  uv run python "$D/model_bench.py" "$@" --tag "$gpu" 2>&1 | grep '^{' >> "$OUT" \
    || echo "FAILED $*"
  echo "done $*"
}

ab() {  # ab <patch> <reps> <args...>: current and patched, interleaved
  local p=$1 reps=$2; shift 2
  for _ in $(seq "$reps"); do one "$@"; one "$@" --patch "$p"; done
}

# preflight: one short run through the same path
one --model unet_r34 --steps 3 --warmup 1 || exit 1

# --- patches, eager (2 reps) and compiled (1 rep) ---
for m in fastai_r34 fastai_edgenext unet_r34_cross; do ab tail_dtype 2 --model $m; done
for m in fastai_r34 unet_r34_cross; do ab tail_dtype 1 --model $m --compile default; done
ab fastai_layout 2 --model fastai_r34 --channels-last
ab cross_exact 1 --model unet_r34_cross --compile default
for m in segformer_r34 segformer_pvt; do
  ab segformer_fuse 2 --model $m; ab segformer_fuse 1 --model $m --compile default; done
for m in upernet_r34 deeplab_r34 fpn_r34; do ab interp_dtype 2 --model $m; done
ab gn_dtype 2 --model unet_r34_gn
ab split_conv 2 --model unet_r34
ab split_conv 1 --model unet_r34 --compile default
ab split_conv 2 --model unetpp_r34 --batch 8
ab split_conv 1 --model unetpp_r34 --batch 8 --compile default
ab pad_fast 2 --model unet_r34
ab mean_agg 2 --model unet_r34_ts --batch 8 --size 256
ab xattn_contig 2 --model xattn_r18 --batch 8

# --- gradient checkpointing (memory for time) ---
for c in "" enc dec enc,dec; do one --model unet_r34 ${c:+--ckpt $c}; done
for c in "" enc; do one --model unet_convnext ${c:+--ckpt $c}; done
for c in "" enc; do one --model unet_r34_ts --batch 8 --size 256 ${c:+--ckpt $c}; done

# --- inference: eager vs compile vs CUDA graphs ---
for m in unet_r34 unet_pvt segformer_pvt; do for b in 1 8; do
  for c in "" default reduce-overhead; do
    one --model $m --mode infer --batch $b ${c:+--compile $c}
  done; done; done
echo ALL DONE
