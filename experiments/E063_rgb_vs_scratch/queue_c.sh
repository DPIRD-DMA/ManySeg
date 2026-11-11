#!/usr/bin/env bash
# E063 part C: starts when the part A/B queue has ended, from E063's frozen commit (README.md).
set -u
export OUT_ROOT="${OUT_ROOT:?set OUT_ROOT}"
FROZEN="${RUNS_ROOT:?set RUNS_ROOT}/${E063_FROZEN:?set E063_FROZEN}"
HERE="$(cd "$(dirname "$0")" && pwd)"
while systemctl --user is-active --quiet e055-queue; do sleep 60; done
echo "=== part C $(git -C "$FROZEN" rev-parse --short HEAD) $(date)"
cd "$FROZEN" && "$HERE/run_c.sh"
rc=$?
echo "=== part C finished rc=$rc $(date)"
exit "$rc"
