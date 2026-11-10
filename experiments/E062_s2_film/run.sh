#!/usr/bin/env bash
# Run E062 arms on one GPU, two at a time. Usage: run.sh <seed> [arm ...]
set -u
cd "$(dirname "$0")"
seed=$1; shift
arms=("$@")
[ ${#arms[@]} -eq 0 ] && arms=(aerial s2enc film_med film_ts film_ctx)
mkdir -p runs
run() { rm -f "runs/$1${TAG:-}_s$seed/result.json"; ../../.venv/bin/python train.py ${EXTRA:-} --arm "$1" --seed "$seed" \
          --out "runs/$1${TAG:-}_s$seed" > "runs/$1${TAG:-}_s$seed.log" 2>&1 \
        && echo "DONE $1${TAG:-} s$seed" || echo "FAILED $1${TAG:-} s$seed"; }
i=0
for a in "${arms[@]}"; do
  run "$a" &
  i=$((i + 1))
  if [ $((i % 2)) -eq 0 ]; then wait; fi
done
wait
echo "ALL DONE seed $seed"
