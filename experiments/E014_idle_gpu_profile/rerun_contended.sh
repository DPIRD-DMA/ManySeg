#!/usr/bin/env bash
# Re-runs every E014 config whose latest record saw other GPU compute processes at start or end
# (pre-registered: contended configs are re-run or excluded). Repeat until none remain.
set -u
OUT=${OUT_ROOT:-.}/experiments/E014_idle_gpu_profile/runs
python3 - "$OUT" <<'PY' | while read -r enc dec prec comp; do
import glob, json, sys
latest = {}
for f in sorted(glob.glob(sys.argv[1] + "/*.json")):
    d = json.load(open(f))
    key = (d["spec"]["encoder"], d["spec"]["arch"], d["precision"], d["compiled"])
    latest[key] = d
for (enc, dec, prec, comp), d in sorted(latest.items()):
    a = d["env"].get("gpu_contention_at_start") or {}
    b = d.get("gpu_contention_at_end") or {}
    if a.get("other_compute_processes", 0) or b.get("other_compute_processes", 0):
        print(enc, dec, prec, "--compile" if comp else "-")
PY
  [ "$comp" = "-" ] && comp=""
  echo "=== rerun $enc $dec $prec $comp"
  uv run manyseg-bench profile --provider manyseg --encoder "$enc" --weights none \
    --arch "$dec" --size 512,512 --in-channels 3 --num-classes 2 --batch-sizes 1,8 \
    --train-batch-size 8 --precision "$prec" $comp --device cuda --out "$OUT" \
    --tag "e014-$dec-$prec${comp:+-compile}" < /dev/null || echo "FAILED $enc $dec $prec $comp"
done
echo "RERUN DONE"
