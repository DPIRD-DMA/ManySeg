#!/usr/bin/env bash
# Run E056 arms on one GPU, two at a time. Usage: run.sh <seed> [arm ...]
set -u
cd "$(dirname "$0")"
seed=$1; shift
arms=("$@")
[ ${#arms[@]} -eq 0 ] && arms=(early mid_concat rgb thermal mid_gated)
mkdir -p runs
run() { ../../.venv/bin/python train.py --arm "$1" --seed "$seed" --data ~/data_e056_mfnet/npz \
          --epochs 80 --out "runs/$1_s$seed" > "runs/$1_s$seed.log" 2>&1 \
        && echo "DONE $1 s$seed" || echo "FAILED $1 s$seed"; }
i=0
for a in "${arms[@]}"; do
  run "$a" &
  i=$((i + 1))
  if [ $((i % 2)) -eq 0 ]; then wait; fi
done
wait
echo "ALL DONE seed $seed"
