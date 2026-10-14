#!/usr/bin/env bash
# watch.sh JOBID...: every 5 min, rsync Setonix results back into experiments/*/runs and print a
# status line per job (states summarised); exits when none of the jobs remain in the queue.
set -u
MAIN="$WORKSPACE/ManySeg"
R=$MANYSEG_ROOT
JOBS=$(IFS=,; echo "$*")
while true; do
  for e in E004_channel_adapters E009_metadata_conditioning E010_instance_heads E011_tree_points; do
    mkdir -p "$MAIN/experiments/$e/runs"
    rsync -a -e "ssh -o BatchMode=yes" "$HOST:$R/experiments/$e/runs/" \
      "$MAIN/experiments/$e/runs/" 2>/dev/null
  done
  STATES=$(ssh -o BatchMode=yes $HOST "sacct -X -n -j $JOBS -o JobName,State" 2>/dev/null \
           | grep -v '^ #' | awk 'NF==2 {c[$1" "$2]++} END {for (k in c) print k"="c[k]}' | sort | tr '\n' ' ')
  echo "$(date +%H:%M) $STATES"
  if echo "$STATES" | grep -qE "FAILED|TIMEOUT|OUT_OF_ME|CANCELLED|NODE_FAIL"; then
    echo "ALERT: failures present: $STATES"
  fi
  LEFT=$(ssh -o BatchMode=yes $HOST "squeue -h -j $JOBS -o %i" 2>/dev/null | grep -v '^ #' | grep -c .)
  [ "$LEFT" -eq 0 ] && { echo "SETONIX DONE"; break; }
  sleep 300
done
