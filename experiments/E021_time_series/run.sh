#!/usr/bin/env bash
# E021: time-series inputs vs single date and dates-as-channels (README.md). Local RTX 4090.
set -u
export MANYSEG_BENCH_DATA="${MANYSEG_BENCH_DATA:-$MANYSEG_BENCH_DATA}"
OUT=${OUT_ROOT:-.}/experiments/E021_time_series/runs
SEEDS=${SEEDS:-0,1,2}
fails=0
run() {  # arm dataset input_mode options set
  uv run manyseg-bench train --provider manyseg --arch unet --encoder resnet34 \
    --weights tv_in1k --recipe fastai_like --dataset "$2" --input-mode "$3" --options "$4" \
    --set "$5" --seeds "$SEEDS" --out "$OUT" --evaluate-test --tag "e021-$1" \
    || { echo "FAILED $1"; fails=$((fails + 1)); }
}
MEAN='{"input_options": {"s2": {"temporal": "mean"}}}'
ATT='{"input_options": {"s2": {"temporal": "attention"}}}'
for ds in ${DATASETS:-pastis_ts ai4boundaries_s2}; do
  case $ds in
    pastis_ts) SET="{\"epochs\": ${EPOCHS:-30}, \"batch_size\": 16}" ;;
    ai4boundaries_s2) SET="{\"epochs\": ${EPOCHS:-30}, \"batch_size\": 8}" ;;
  esac
  for arm in ${ARMS:-single stack mean attention}; do
    case $arm in
      single)    run "$ds-single" "${ds}_single" stack '{}' "$SET" ;;
      stack)     run "$ds-stack" "$ds" stack '{}' "$SET" ;;
      mean)      run "$ds-mean" "$ds" dict "$MEAN" "$SET" ;;
      attention) run "$ds-attention" "$ds" dict "$ATT" "$SET" ;;
    esac
  done
done
[ "$fails" -eq 0 ] || { echo "$fails arm(s) FAILED"; exit 1; }
echo "ALL DONE"
