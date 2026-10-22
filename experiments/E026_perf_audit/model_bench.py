"""E026: time one training step (or inference call) of a model, with or without patches.

One configuration per process (fresh allocator and compile caches), one JSON line out:

    uv run python experiments/E026_perf_audit/model_bench.py --model fastai_r34 \
        --patch tail_dtype [--compile default] [--mode infer] [--ckpt enc]

bf16 autocast, fp32 weights. Train step = forward + fp32 cross-entropy + backward +
zero_grad (as E025). Times from CUDA events, median after warm-up; peak memory is
``max_memory_allocated`` over the timed steps.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import models  # noqa: E402
import patches  # noqa: E402


def gpu_state() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                               "--format=csv,noheader"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(models.MODELS))
    ap.add_argument("--patch", default="", help="comma-separated patch names (patches.py)")
    ap.add_argument("--ckpt", default="", help="enc, dec or enc,dec: gradient checkpointing")
    ap.add_argument("--mode", default="train", choices=["train", "infer"])
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--in-ch", type=int, default=3)
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--compile", default="", help="'' (eager), default, reduce-overhead, "
                                                   "max-autotune")
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    applied = patches.apply(a.patch.split(","))
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    dev = "cuda"
    model = models.build(a.model, a.in_ch, a.classes).to(dev)
    model.train(a.mode == "train")
    ck = set(filter(None, a.ckpt.split(",")))
    n_ck = {}
    if "enc" in ck:
        n_ck["enc"] = patches.enable_encoder_checkpointing(model)
    if "dec" in ck:
        n_ck["dec"] = patches.enable_decoder_checkpointing(model)
    mf = torch.channels_last if a.channels_last else torch.contiguous_format
    model = model.to(memory_format=mf)
    params = sum(p.numel() for p in model.parameters())
    if a.compile:
        kw = {} if a.compile == "default" else {"mode": a.compile}
        if a.mode == "infer" and a.compile != "default":
            kw["dynamic"] = False  # CUDA graphs need static shapes
        step_model = torch.compile(model, **kw)
    else:
        step_model = model
    x = models.make_input(a.model, a.batch, a.size, a.in_ch, dev)
    if a.channels_last:
        x = (x.contiguous(memory_format=mf) if isinstance(x, torch.Tensor)
             else {k: v.contiguous(memory_format=mf) if v.ndim == 4 else v for k, v in x.items()})
    y = torch.randint(0, a.classes, (a.batch, a.size, a.size), device=dev)

    def train_step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = step_model(x)
        loss = F.cross_entropy(out.float(), y)
        loss.backward()
        model.zero_grad(set_to_none=True)

    @torch.inference_mode()
    def infer_step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return step_model(x)

    step = train_step if a.mode == "train" else infer_step
    warmup = a.warmup + (10 if a.compile else 0)
    first = None
    for i in range(warmup):
        if i == 0:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            step()
            e.record()
            e.synchronize()
            first = s.elapsed_time(e)
        else:
            step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()
    times = []
    for _ in range(a.steps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        step()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    ts = sorted(times)
    rec = {"model": a.model, "patches": applied, "ckpt": sorted(ck), "ckpt_modules": n_ck,
           "mode": a.mode, "size": a.size, "batch": a.batch, "in_ch": a.in_ch,
           "compile": a.compile, "channels_last": a.channels_last, "params_m": params / 1e6,
           "median_ms": statistics.median(times), "p10_ms": ts[len(ts) // 10],
           "p90_ms": ts[len(ts) * 9 // 10], "first_call_ms": first,
           "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
           "peak_over_static_gib": (torch.cuda.max_memory_allocated() - base_mem) / 2**30,
           "steps": a.steps, "gpu_before": a.tag, "gpu_after": gpu_state(),
           "torch": torch.__version__}
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
