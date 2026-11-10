#!/usr/bin/env bash
# E062: seed 0, val screen, then (if it passes) the shuffled control and seeds 1-2.
set -u
cd "$(dirname "$0")"
echo "START $(date)"
check() { for a in "$@"; do [ -f runs/${a}_s$seed/result.json ] || { echo "FAILED missing result $a s$seed"; exit 1; }; grep RESULT runs/${a}_s$seed.log; done; }
seed=0; ./run.sh 0; check aerial s2enc film_med film_ts film_ctx
../../.venv/bin/python screen.py; rc=$?
if [ $rc -eq 3 ]; then echo "STOP screen failed $(date)"; echo "PIPELINE DONE"; exit 0; fi
[ $rc -eq 0 ] || { echo "FAILED screen error"; exit 1; }
pick=$(../../.venv/bin/python -c "import json;print(json.load(open('runs/screen.json'))['pick'])")
seed=0; ./run.sh 0 ${pick}_shuf; check ${pick}_shuf
for seed in 1 2; do
  ./run.sh $seed aerial s2enc $pick ${pick}_shuf; check aerial s2enc $pick ${pick}_shuf
done
echo "PIPELINE DONE $(date)"
