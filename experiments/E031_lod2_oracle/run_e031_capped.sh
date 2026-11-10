#!/usr/bin/env bash
# Launch run_e031.sh as a transient user service capped at 12 GB : detached from
# any terminal session, and an out-of-memory kills only this job.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$here/runs"
systemd-run --user --unit "e031-$(date +%Y%m%d-%H%M%S)" -p MemoryMax=8G -p MemorySwapMax=0 \
  -p "WorkingDirectory=$here" -p "StandardOutput=append:$here/runs/run_e031.log" \
  -p "StandardError=append:$here/runs/run_e031.log" --setenv=PATH="$PATH" \
  --setenv=HOME="$HOME" "$here/run_e031.sh"
