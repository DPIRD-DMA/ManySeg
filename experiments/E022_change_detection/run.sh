#!/usr/bin/env bash
# E022: binary change detection, early fusion vs Siamese fusions (README.md). Local RTX 4090.
set -u
export MANYSEG_BENCH_DATA="${MANYSEG_BENCH_DATA:-$MANYSEG_BENCH_DATA}"
OUT=${OUT_ROOT:-.}/experiments/E022_change_detection/runs
SEEDS=${SEEDS:-0,1,2}
SIAM='"input_options": {"t2": {"backbone": "shared:t1", "share_norm": true}}'
fails=0
run() {  # tag dataset input_mode options set
  uv run manyseg-bench train --provider manyseg --arch unet --encoder resnet34 \
    --weights tv_in1k --recipe fastai_like --dataset "$2" --input-mode "$3" --options "$4" \
    --set "$5" --seeds "$SEEDS" --out "$OUT" --evaluate-test --tag "e022-$1" \
    || { echo "FAILED $1"; fails=$((fails + 1)); }
}
for ds in ${DATASETS:-oscd levircd}; do
  case $ds in
    oscd)    SET="{\"epochs\": ${OSCD_EPOCHS:-100}, \"batch_size\": 16}" ;;
    levircd) SET="{\"epochs\": ${LEVIR_EPOCHS:-30}, \"batch_size\": 16}" ;;
  esac
  for arm in ${ARMS:-stack concat diff concat_diff}; do
    case $arm in
      stack) run "$ds-stack" "$ds" stack '{}' "$SET" ;;
      *) run "$ds-siam-$arm" "$ds" dict \
           "{\"fusion\": {\"type\": \"intermediate\", \"op\": \"$arm\"}, $SIAM}" "$SET" ;;
    esac
  done
done
[ "$fails" -eq 0 ] || { echo "$fails arm(s) FAILED"; exit 1; }
echo "ALL DONE"
