"""E027: evaluation throughput (``manyseg_bench.training.trainer.evaluate``) with or without
patches, on a synthetic device-resident split through the bench's real ``GPULoader``.

* ``--patch eval_nosync``: evaluator without host syncs (patches2.py).
* ``--compile default|reduce-overhead``: the model compiled; ``--pad-last`` pads the final
  short batch to the eval batch size (and drops the padding from the metrics), so compiled
  evaluation sees one shape. Batches are built once up front, for every variant.

Two full passes are timed separately (the first includes compilation / autotuning).

    uv run python experiments/E027_perf_round2/eval_bench.py --model segformer_pvt --size 256
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "E026_perf_audit"))
import models  # noqa: E402
import patches2  # noqa: E402
from manyseg_bench.datasets.gpu_loader import GPULoader  # noqa: E402
from manyseg_bench.training.trainer import evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--patch", default="")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--n", type=int, default=500, help="images (not a multiple of the batch)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--compile", default="")
    ap.add_argument("--pad-last", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    applied = patches2.apply(a.patch.split(","))
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 256, (a.n, a.size, a.size, 3), dtype=np.uint8)
    masks = rng.integers(0, a.classes, (a.n, a.size, a.size), dtype=np.uint8)
    loader = GPULoader(imgs, masks, a.batch, [0.45] * 3, [0.25] * 3, dev, train=False)
    model = models.build(a.model, 3, a.classes).to(dev).eval()
    run = model
    if a.compile:
        kw = {} if a.compile == "default" else {"mode": a.compile}
        run = torch.compile(model, **kw)
    # batches built once (both variants), so passes time the model and the evaluator only
    batches = []
    for x, y in loader:
        if a.pad_last and y.shape[0] < a.batch:  # the model returns only the real rows
            x = torch.cat([x, x[:1].expand(a.batch - y.shape[0], *x.shape[1:])])
        batches.append((x, y))
    run = _SliceToTarget(run)
    loader_obj = _PairLoader(batches, run)
    times, metrics = [], None
    for _ in range(2):
        torch.cuda.synchronize()
        t = time.perf_counter()
        metrics = evaluate(run, loader_obj, a.classes, dev, "bf16")
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    rec = {"model": a.model, "patches": applied, "size": a.size, "n": a.n, "batch": a.batch,
           "compile": a.compile, "pad_last": a.pad_last, "first_pass_s": times[0],
           "second_pass_s": times[1], "images_per_s": a.n / times[1],
           "miou": metrics["miou"], "loss": metrics["loss"],
           "boundary_f1": metrics["boundary_f1"],
           "gpu_before": a.tag, "torch": torch.__version__,
           "gpu_after": subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                        "--format=csv,noheader"], capture_output=True,
                                       text=True).stdout.strip()}
    print(json.dumps(rec), flush=True)


class _SliceToTarget(torch.nn.Module):
    """Calls the (compiled) model on the padded batch; the loader tells it how many rows
    are real, and it returns only those."""

    def __init__(self, inner):
        super().__init__()
        self.inner, self.n = inner, None

    def forward(self, x):
        out = self.inner(x)
        return out if self.n is None else out[:self.n]

    def eval(self):  # evaluate() calls model.eval()
        return self


class _PairLoader:
    def __init__(self, batches, model):
        self.batches, self.model = batches, model

    def __iter__(self):
        for x, y in self.batches:
            self.model.n = y.shape[0]
            yield x, y


if __name__ == "__main__":
    main()
