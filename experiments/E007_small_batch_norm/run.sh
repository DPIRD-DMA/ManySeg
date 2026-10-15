#!/usr/bin/env bash
# E007: decoder norm x encoder BN at batch 2, plus the batch-16 default (30 runs).
set -u
export MANYSEG_BENCH_DATA="${MANYSEG_BENCH_DATA:-$MANYSEG_BENCH_DATA}"
RECIPE=${RECIPE:-fastai_like}
OUT=${OUT_ROOT:-.}/experiments/E007_small_batch_norm/runs
mb() {
  if [ -n "${EMIT_TASKS:-}" ]; then printf '%q ' "$@" >> "$EMIT_TASKS"; echo >> "$EMIT_TASKS"
  else uv run manyseg-bench "$@"; fi
}
for seed in 0 1 2; do
  for ds in oxford_pet sen1floods11; do
    ep=$([ "$ds" = oxford_pet ] && echo 20 || echo 60)
    # Trimmed 2026-10-15 for the Pawsey budget (see README amendment): batch 2 x {batch, group}
    # x {train, frozen}; batch 16 default only.
    for arm in "2 batch false" "2 batch true" "2 group false" "2 group true" "16 batch false"; do
      read -r bs norm encbn <<< "$arm"
      {
          dn=$([ "$norm" = none ] && echo null || echo "\"$norm\"")
          mb train --provider manyseg --dataset "$ds" --encoder resnet34 --weights tv_in1k \
            --arch unet --options "{\"decoder_kwargs\": {\"norm\": $dn}}" --recipe "$RECIPE" \
            --set "{\"epochs\": $ep, \"batch_size\": $bs, \"freeze_encoder_bn\": $encbn}" \
            --seeds "$seed" --out "$OUT" --evaluate-test \
            --tag "e007-$ds-b$bs-$norm-encbn_$([ "$encbn" = true ] && echo frozen || echo train)" \
            || echo "FAILED $ds $bs $norm $encbn $seed"
      }
    done
  done
done
echo "ALL DONE"
