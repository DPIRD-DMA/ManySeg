#!/usr/bin/env bash
# Launch run_e030.sh as a transient user service capped at 12 GB (amendment 2): detached from
# any terminal session, and an out-of-memory kills only this job.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$here/runs"
systemd-run --user --unit "e030-$(date +%Y%m%d-%H%M%S)" -p MemoryMax=12G -p MemorySwapMax=0 \
  -p "WorkingDirectory=$here" -p "StandardOutput=append:$here/runs/run_e030.log" \
  -p "StandardError=append:$here/runs/run_e030.log" --setenv=PATH="$PATH" \
  --setenv=HOME="$HOME" "$here/run_e030.sh"
