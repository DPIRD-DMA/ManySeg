"""E027: compiled evaluation inside training, as the bench trainer does it.

The model is compiled once (``torch.compile(model)``, as ``trainer.train``), trained for a few
steps at the training batch size, then evaluated on a split whose size is not a multiple of
the evaluation batch size. Variants:

* ``same``: the training callable, unpadded last batch (the trainer today).
* ``same_pad``: the training callable, last batch padded to the evaluation batch size.
* ``static_pad``: a separate ``torch.compile(model, dynamic=False)`` for evaluation, padded.

Each runs train steps + eval twice (an "epoch" each); the second epoch's eval is reported.

    uv run python experiments/E027_perf_round2/eval_compile_bench.py --variant static_pad
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "E026_perf_audit"))
import models  # noqa: E402
import patches2  # noqa: E402
from eval_bench import _PairLoader, _SliceToTarget  # noqa: E402
from manyseg_bench.datasets.gpu_loader import GPULoader  # noqa: E402
from manyseg_bench.training.trainer import evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unet_pvt")
    ap.add_argument("--variant", required=True, choices=["same", "same_pad", "static_pad"])
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--train-batch", type=int, default=16)
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()
    patches2.apply(["eval_nosync"])
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 256, (a.n_eval, a.size, a.size, 3), dtype=np.uint8)
    masks = rng.integers(0, 4, (a.n_eval, a.size, a.size), dtype=np.uint8)
    val = GPULoader(imgs, masks, a.eval_batch, [0.45] * 3, [0.25] * 3, dev, train=False)
    model = models.build(a.model, 3, 4).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    run = torch.compile(model)
    eval_run = torch.compile(model, dynamic=False) if a.variant == "static_pad" else run
    pad = a.variant != "same"
    batches = []
    for x, y in val:
        if pad and y.shape[0] < a.eval_batch:
            x = torch.cat([x, x[:1].expand(a.eval_batch - y.shape[0], *x.shape[1:])])
        batches.append((x, y))
    sliced = _SliceToTarget(eval_run)
    ev_loader = _PairLoader(batches, sliced)
    xt = torch.randn(a.train_batch, 3, a.size, a.size, device=dev)
    yt = torch.randint(0, 4, (a.train_batch, a.size, a.size), device=dev)
    out = {"model": a.model, "variant": a.variant, "train_batch": a.train_batch,
           "eval_batch": a.eval_batch, "n_eval": a.n_eval}
    for epoch in range(2):
        model.train()
        t = time.perf_counter()
        for _ in range(a.steps):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = run(xt)
            F.cross_entropy(o.float(), yt).backward()
            opt.step()
        torch.cuda.synchronize()
        out[f"train_s_{epoch}"] = time.perf_counter() - t
        model.eval()  # the wrapper's eval() does not reach the model
        t = time.perf_counter()
        m = evaluate(sliced, ev_loader, 4, dev, "bf16")
        torch.cuda.synchronize()
        out[f"eval_s_{epoch}"] = time.perf_counter() - t
        model.train()
    out["eval_images_per_s"] = a.n_eval / out["eval_s_1"]
    out["miou"] = m["miou"]
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
