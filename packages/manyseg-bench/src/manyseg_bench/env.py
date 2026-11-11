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


def _smi(*args: str) -> list[list[str]]:
    """Rows of an ``nvidia-smi --format=csv,noheader,nounits`` query, split into fields."""
    out = subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=10, check=True).stdout
    return [[f.strip() for f in ln.split(",")] for ln in out.strip().splitlines() if ln.strip()]


def _cuda_uuid(index: int) -> str | None:
    """The CUDA device's UUID as torch reports it (None if unavailable)."""
    try:
        return str(torch.cuda.get_device_properties(index).uuid)
    except Exception:
        return None


def _mib(v: str) -> int:
    """A process's ``used_memory`` in MiB; 0 when ``nvidia-smi`` cannot report it ("[N/A]")."""
    try:
        return int(v)
    except ValueError:
        return 0


def _norm_uuid(u: str) -> str:
    return u.lower().removeprefix("gpu-")


def gpu_contention(device: str | torch.device = "cuda") -> dict | None:
    """Snapshot of the run's GPU and of *other* compute processes on it. Timing under
    contention is not clean.

    ``contended`` is True when another compute process is on this GPU
    (``other_compute_processes`` > 0; ``other_mem_mib`` is their memory). Utilisation
    (``gpu_util_pct``) and ``mem_used_mib`` are recorded as telemetry only: they include this
    process's own kernels and memory, so at the end of a run they say nothing about others.

    The GPU is ``device`` (index from the device, else the current CUDA device), found in
    ``nvidia-smi`` by its UUID (so ``CUDA_VISIBLE_DEVICES`` renumbering does not matter; by
    index when torch reports no UUID), and only processes on that GPU count. ``None`` for a
    non-CUDA device, without ``nvidia-smi`` (e.g. ROCm), or when its output cannot be parsed:
    telemetry never fails a run.
    """
    try:
        dev = torch.device(device)
        if dev.type != "cuda":
            return None
        index = dev.index if dev.index is not None else torch.cuda.current_device()
        uuid = _cuda_uuid(index)
        gpus = _smi("--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total")
        if uuid is not None:
            row = next(g for g in gpus if _norm_uuid(g[1]) == _norm_uuid(uuid))
        else:
            row = next(g for g in gpus if int(g[0]) == index)
        util, used, total = (int(v) for v in row[2:5])
        procs = _smi("--query-compute-apps=gpu_uuid,pid,used_memory")
        me = os.getpid()
        others = [p for p in procs
                  if _norm_uuid(p[0]) == _norm_uuid(row[1]) and int(p[1]) != me]
        other_mem = sum(_mib(p[2]) for p in others)
    except Exception:  # no nvidia-smi / no CUDA / "[N/A]" or unexpected output
        return None
    return {
        "gpu_index": int(row[0]),
        "gpu_util_pct": util,
        "mem_used_mib": used,
        "mem_total_mib": total,
        "other_compute_processes": len(others),
        "other_mem_mib": other_mem,
        "contended": len(others) > 0,
    }


def code_version() -> str | None:
    """Code snapshot that produced a result: $MANYSEG_CODE_SHA (set by batch scripts) or the
    git commit of the repository holding this package, with ``-dirty`` if any tracked file
    anywhere in that repository (library, benchmark, experiments) has uncommitted changes.
    Untracked files do not count (data, run outputs and scratch files would otherwise mark
    every run dirty), so commit new source files before a run. ``None`` outside a git
    checkout (e.g. an installed wheel) or without git."""
    sha = os.environ.get("MANYSEG_CODE_SHA")
    if sha:
        return sha
    here = os.path.dirname(os.path.abspath(__file__))

    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", here, *args], capture_output=True, text=True,
                              timeout=10).stdout.strip()

    try:
        root = git("rev-parse", "--show-toplevel")
        if not root:
            return None
        rev = git("rev-parse", "--short", "HEAD")
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain",
                                "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return (rev + ("-dirty" if dirty else "")) or None


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
        "code_version": code_version(),
        "hip": torch.version.hip,
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        env["gpu"] = props.name
        env["gpu_mem_gib"] = round(props.total_memory / 2**30, 1)
        env["gpu_contention_at_start"] = gpu_contention(device)
    elif device.type == "mps":
        env["gpu"] = "Apple MPS"
    return env
