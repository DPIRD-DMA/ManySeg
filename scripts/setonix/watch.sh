#!/usr/bin/env bash
# watch.sh JOBID...: every 5 min, rsync Setonix results back into experiments/*/runs and print a
# status line per job (states summarised); exits when none of the jobs remain in the queue.
set -u
MAIN="$WORKSPACE/ManySeg"
R=$MANYSEG_ROOT
JOBS=$(IFS=,; echo "$*")
while true; do
  # Every experiment's runs/ (E*/runs/**), so new experiments need no edit here.
  rsync -a -e "ssh -o BatchMode=yes" --include='E*/' --include='E*/runs/***' --exclude='*' \
    "$HOST:$R/experiments/" "$MAIN/experiments/" 2>/dev/null
  STATES=$(ssh -o BatchMode=yes $HOST "sacct -X -n -j $JOBS -o JobName,State" 2>/dev/null \
           | grep -v '^ #' | awk 'NF==2 {c[$1" "$2]++} END {for (k in c) print k"="c[k]}' | sort | tr '\n' ' ')
  echo "$(date +%H:%M) $STATES"
  if echo "$STATES" | grep -qE "FAILED|TIMEOUT|OUT_OF_ME|NODE_FAIL"; then
    echo "ALERT: failures present: $STATES"
  fi
  LEFT=$(ssh -o BatchMode=yes $HOST "squeue -h -j $JOBS -o %i" 2>/dev/null | grep -v '^ #' | grep -c .)
  [ "$LEFT" -eq 0 ] && { echo "SETONIX DONE"; break; }
  sleep 300
done
