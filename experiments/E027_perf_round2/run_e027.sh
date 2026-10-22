#!/usr/bin/env bash
# E027: round-2 candidates. Run from the perf/e027 worktree. Current vs patched alternate.
set -u
D="experiments/E027_perf_round2"
R="$D/runs"; mkdir -p "$R"
gpu() { nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; }
j() { grep '^{' >> "$1" || echo "FAILED $2"; echo "done $2"; }

# equivalence on CUDA (bf16) for the dtype patches
for p in fusion_dtype neck_dtype unet_bilinear_dtype; do
  uv run python "$D/check2.py" --patch $p 2>&1 | j "$R/check.jsonl" "check $p"; done

# evaluation: host syncs (eager), 2 interleaved reps
for cfg in "--model segformer_pvt --size 256 --batch 32" "--model unet_pvt --size 256 --batch 32" \
           "--model unet_r34 --size 256 --batch 32" "--model unet_r34 --size 512 --batch 8"; do
  for rep in 1 2; do for p in "" eval_nosync; do
    # shellcheck disable=SC2086
    uv run python "$D/eval_bench.py" $cfg ${p:+--patch $p} --tag "$(gpu)" 2>&1 \
      | j "$R/eval.jsonl" "eval $cfg $p"
  done; done
done
# evaluation compiled: unpadded last batch vs padded vs CUDA graphs (all with eval_nosync)
for c in "--compile default" "--compile default --pad-last" "--compile reduce-overhead --pad-last"; do
  for m in unet_pvt segformer_pvt; do
    # shellcheck disable=SC2086
    uv run python "$D/eval_bench.py" --model $m --size 256 --batch 32 --patch eval_nosync $c \
      --tag "$(gpu)" 2>&1 | j "$R/eval.jsonl" "eval $m $c"
  done
done

# dtype patches: training step A/B, 2 reps
ab() { local p=$1; shift; for rep in 1 2; do
  uv run python "$D/model_bench2.py" "$@" --tag "$(gpu)" 2>&1 | j "$R/models.jsonl" "model $*"
  uv run python "$D/model_bench2.py" "$@" --patch2 $p --tag "$(gpu)" 2>&1 \
    | j "$R/models.jsonl" "model $* $p"; done; }
ab fusion_dtype --model late_concat_extent --batch 8
ab fusion_dtype --model inter_concat_extent --batch 8
ab neck_dtype --model unet_vit
ab unet_bilinear_dtype --model unet_r34_bilinear
ab unet_bilinear_dtype --model unet_r34_aligned

# object metrics (instance / point evaluation)
for rep in 1 2; do for p in "" objects_cpu; do
  uv run python "$D/objects_bench.py" ${p:+--patch $p} 2>&1 | j "$R/objects.jsonl" "objects $p"
done; done

# host RAM of a device-resident dataset
for m in "" --mmap; do uv run python "$D/ram_bench.py" $m 2>&1 | j "$R/ram.jsonl" "ram $m"; done
echo ALL DONE
