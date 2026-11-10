#!/usr/bin/env bash
# E061 overnight: wait for prep, seed 0, val screen, then seeds 1-2 only if the screen passes.
set -u
cd "$(dirname "$0")"
echo "START $(date)"
while [ ! -f ~/data_e061/prep.exit ]; do sleep 30; done
[ "$(cat ~/data_e061/prep.exit)" = 0 ] || { echo "FAILED prep"; exit 1; }
echo "PREP OK $(date)"; tail -4 ~/data_e061/prep.log
./run.sh 0
for a in aerial ctx10 ctx20 ctx40; do
  [ -f runs/${a}_s0/result.json ] || { echo "FAILED missing result $a s0"; exit 1; }
  grep RESULT runs/${a}_s0.log
done
../../.venv/bin/python screen.py; rc=$?
if [ $rc -eq 3 ]; then echo "STOP screen failed $(date)"; echo "PIPELINE DONE"; exit 0; fi
[ $rc -eq 0 ] || { echo "FAILED screen error"; exit 1; }
pick=$(../../.venv/bin/python -c "import json;print(json.load(open('runs/screen.json'))['pick'])")
for s in 1 2; do
  ./run.sh $s aerial ctx10 $pick
  for a in aerial ctx10 $pick; do
    [ -f runs/${a}_s$s/result.json ] || { echo "FAILED missing result $a s$s"; exit 1; }
    grep RESULT runs/${a}_s$s.log
  done
done
echo "PIPELINE DONE $(date)"
