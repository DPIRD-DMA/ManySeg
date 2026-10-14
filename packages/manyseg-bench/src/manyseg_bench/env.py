"""Hardware/software provenance recorded with every result."""

from __future__ import annotations

import os
import platform
import subprocess
from importlib import metadata

import torch


def _pkg(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def gpu_contention() -> dict | None:
    """Snapshot of *other* processes on the GPU. Timing under contention is not clean."""
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        procs = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    util, used, total = (int(v) for v in q[0].split(", "))
    me = os.getpid()
    others = [p for p in procs if p and int(p.split(", ")[0]) != me]
    return {
        "gpu_util_pct": util,
        "mem_used_mib": used,
        "mem_total_mib": total,
        "other_compute_processes": len(others),
        "contended": util > 5 or len(others) > 0,
    }


def environment(device: str | torch.device = "cpu") -> dict:
    device = torch.device(device)
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "packages": {p: _pkg(p) for p in
                     ("manyseg", "manyseg-bench", "timm", "segmentation-models-pytorch",
                      "fastai", "torchvision", "numpy")},
        "device": str(device),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        env["gpu"] = props.name
        env["gpu_mem_gib"] = round(props.total_memory / 2**30, 1)
        env["gpu_contention_at_start"] = gpu_contention()
    elif device.type == "mps":
        env["gpu"] = "Apple MPS"
    return env
