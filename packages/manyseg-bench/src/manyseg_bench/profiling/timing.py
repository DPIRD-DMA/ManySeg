"""Device-correct timing: warmup, explicit synchronisation, repeated measurement."""

from __future__ import annotations

import io
import statistics
import time
from collections.abc import Callable

import torch
from torch import nn


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def time_fn(fn: Callable[[], object], device: torch.device, warmup: int = 10,
            iters: int = 30, min_time_s: float = 0.0) -> dict:
    """Wall-clock per call with synchronisation around each call. Returns ms statistics;
    ``first_call_ms`` is the first warmup call alone (compilation, cache and allocator warmup)."""
    synchronize(device)
    t0 = time.perf_counter()
    fn()
    synchronize(device)
    first = (time.perf_counter() - t0) * 1e3
    for _ in range(warmup - 1):
        fn()
    synchronize(device)
    times = []
    start_all = time.perf_counter()
    while len(times) < iters or (time.perf_counter() - start_all) < min_time_s:
        t0 = time.perf_counter()
        fn()
        synchronize(device)
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return {
        "median_ms": statistics.median(times),
        "p10_ms": times[int(0.1 * (len(times) - 1))],
        "p90_ms": times[int(0.9 * (len(times) - 1))],
        "iters": len(times),
        "first_call_ms": first,
    }


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def serialized_size_bytes(model: nn.Module) -> int:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell()


def count_flops(model: nn.Module, x: torch.Tensor) -> int | None:
    """Forward FLOPs (2 x MACs) using torch's FlopCounterMode; None if unsupported."""
    from torch.utils.flop_counter import FlopCounterMode

    try:
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(x)
        return int(fc.get_total_flops())
    except Exception:  # noqa: BLE001 - some ops are not countable
        return None


def peak_memory_mib(device: torch.device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20
    return None


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
