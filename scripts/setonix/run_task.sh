#!/bin/bash
# Runs line $2 of task file $1 (manyseg-bench train arguments, shell-quoted) inside the
# PyTorch container, using the ManySeg venv on scratch.
set -eu
TASKS=$1; IDX=$2
ROOT=$MANYSEG_ROOT
source "$ROOT/venv/bin/activate"
LINE=$(sed -n "${IDX}p" "$TASKS")
[ -n "$LINE" ] || { echo "no task $IDX in $TASKS"; exit 2; }
echo "task $IDX: manyseg-bench $LINE"
python -c "import torch; print('torch', torch.__version__, 'gpu', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
eval "manyseg-bench $LINE"
