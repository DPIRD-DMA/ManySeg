# Running ManySeg experiments on Pawsey Setonix

These scripts are the recipe behind the Setonix results, kept for reference; adapt them to your own
allocation. Set `MANYSEG_ROOT` to a working directory on scratch and pass your project on submission:
`sbatch --account=<project> --output=$MANYSEG_ROOT/logs/%x_%j.out ...`.

Layout under `$MANYSEG_ROOT`: `code/<sha>/` (frozen snapshots via
`git archive`), `data/{cache,raw}` (rsync of the bench data root), `venv/` (built on top of the
`pytorch/2.7.1-rocm6.3.3` container), `hf/` (pre-fetched weights), `experiments/` (results),
`logs/`.

1. Snapshot code: `git archive HEAD | ssh <host> "mkdir -p $MANYSEG_ROOT/code/$SHA && tar -x -C $MANYSEG_ROOT/code/$SHA"`.
2. venv (once): `module load pytorch/2.7.1-rocm6.3.3; venv venv`, then inside the container
   `pip install ...` with **numpy pinned to the container's 1.26** (the
   container torch is built against NumPy 1; pyarrow < 16 for the same reason) and
   `pip install --no-deps -e code/<sha>/packages/{manyseg,manyseg-bench}`. The repo keeps no
   constraints file, so write the pins yourself.
3. Pre-fetch pretrained weights on a login node, so runs use a fixed copy (`HF_HUB_OFFLINE=1`):
   `HF_HOME=$MANYSEG_ROOT/hf python -c "import timm; timm.create_model('resnet34.tv_in1k', pretrained=True)"`.
4. Tasks: `EMIT_TASKS=tasks.txt OUT_ROOT=$MANYSEG_ROOT RECIPE=fastai_like bash experiments/<E>/run.sh`
   writes the experiment's exact runs, one shell-quoted argument line each. Only 11 `run.sh` files
   handle `EMIT_TASKS`: E004, E006, E007, E009, E010, E011, E011b, E012, E013, E016 and E017. The
   others ignore it and **start training**, so check first with
   `grep -l EMIT_TASKS experiments/*/run.sh`.
5. Submit: `sbatch --account=<project> --array=1-N%K manyseg_array.sbatch $MANYSEG_ROOT/tasks.txt <sha>` (1 GCD per task).
6. Copy results back: `rsync <host>:$MANYSEG_ROOT/experiments/<E>/runs/ experiments/<E>/runs/`.

Gotchas found on 2026-10-14: memory-mapped reads on Lustre page-fault at ~0.4 MB/s, so each task stages its
dataset cache onto node-local NVMe (`/tmp`, 3.5 TB per node) before training; the module's `pytorch-shell` wrapper is an interactive shell and
silently ignores scripts (exit 0, no output), so use `singularity exec`. `cudnn.benchmark` triggers an
exhaustive MIOpen search per conv shape, hence `MIOPEN_FIND_MODE=FAST`. Results are on MI250X
with torch 2.7.1/ROCm 6.3; compare arms only within one machine (each experiment runs entirely
on one).
