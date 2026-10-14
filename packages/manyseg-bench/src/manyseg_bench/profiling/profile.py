"""Inference and training-step profiling of a built model under a fixed protocol."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn

from .timing import (
    count_flops,
    count_params,
    peak_memory_mib,
    reset_peak,
    serialized_size_bytes,
    time_fn,
)

_DTYPES = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def _autocast(device: torch.device, precision: str):
    dt = _DTYPES[precision]
    return nullcontext() if dt is None else torch.autocast(device.type, dtype=dt)


def profile_model(
    model: nn.Module,
    *,
    in_channels: int,
    img_size: tuple[int, int],
    device: str | torch.device = "cuda",
    batch_sizes: tuple[int, ...] = (1, 8, 32),
    precision: str = "bf16",
    compile: bool = False,
    channels_last: bool = False,
    train_batch_size: int | None = 16,
    warmup: int = 10,
    iters: int = 30,
) -> dict:
    device = torch.device(device)
    h, w = img_size
    model = model.to(device).eval()
    mf = torch.channels_last if channels_last else torch.contiguous_format
    model = model.to(memory_format=mf)
    res: dict = {
        "params": count_params(model),
        "serialized_bytes": serialized_size_bytes(model),
        "flops_per_image": count_flops(model, torch.randn(1, in_channels, h, w, device=device)),
        "precision": precision, "compiled": compile, "channels_last": channels_last,
        "img_size": [h, w], "device": str(device), "inference": [], "train": None,
    }
    run = torch.compile(model) if compile else model

    for bs in batch_sizes:
        x = torch.randn(bs, in_channels, h, w, device=device).to(memory_format=mf)

        def step(x=x):
            with torch.no_grad(), _autocast(device, precision):
                return run(x)

        try:
            reset_peak(device)
            t = time_fn(step, device, warmup=warmup, iters=iters)
        except torch.OutOfMemoryError:
            res["inference"].append({"batch_size": bs, "oom": True})
            continue
        ips = bs / (t["median_ms"] / 1e3)
        res["inference"].append({
            "batch_size": bs, **t, "images_per_s": ips,
            "megapixels_per_s": ips * h * w / 1e6,
            "peak_mem_mib": peak_memory_mib(device),
        })

    if train_batch_size:
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=1e-6)
        x = torch.randn(train_batch_size, in_channels, h, w, device=device).to(memory_format=mf)

        def tstep():
            opt.zero_grad(set_to_none=True)
            with _autocast(device, precision):
                loss = run(x).float().mean()
            loss.backward()
            opt.step()

        try:
            reset_peak(device)
            t = time_fn(tstep, device, warmup=max(3, warmup // 2), iters=max(10, iters // 2))
            res["train"] = {"batch_size": train_batch_size, **t,
                            "images_per_s": train_batch_size / (t["median_ms"] / 1e3),
                            "peak_mem_mib": peak_memory_mib(device)}
        except torch.OutOfMemoryError:
            res["train"] = {"batch_size": train_batch_size, "oom": True}
        model.eval()
    return res
