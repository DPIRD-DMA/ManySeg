#!/usr/bin/env bash
# Run E055 arms on one GPU, two at a time. Usage: run.sh <seed> [arm ...]
set -u
cd "$(dirname "$0")"
seed=$1; shift
arms=("$@")
[ ${#arms[@]} -eq 0 ] && arms=(none film film_shuffled channels probe nozoom)
mkdir -p runs
run() { ../../.venv/bin/python train.py ${EXTRA:-} --arm "$1" --seed "$seed" --data ~/data_e055_nyu/npz \
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
