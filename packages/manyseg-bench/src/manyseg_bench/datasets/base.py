"""Prepared datasets are uint8 arrays cached on disk: images (N, H, W, C), masks (N, H, W).

Targets encoded from geometry (vector-output tasks) are several label arrays per split,
``{split}_masks.{name}.npy`` (e.g. ``corner``, ``direction``), loaded as a dict."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


def check_native_size(size: int | None, native: int, dataset: str) -> None:
    """For datasets whose tiles have a fixed size: ``prepare(size=...)`` may only restate it
    (``manyseg-bench prepare --size`` must not be silently ignored)."""
    if size is not None and int(size) != native:
        raise ValueError(f"{dataset} has fixed {native} x {native} tiles; size={size} is not "
                         "supported (omit --size)")


def data_root(path: str | os.PathLike | None = None) -> Path:
    """Resolve the data root: explicit path > $MANYSEG_BENCH_DATA > ~/.cache/manyseg-bench."""
    p = path or os.environ.get("MANYSEG_BENCH_DATA") or Path.home() / ".cache" / "manyseg-bench"
    return Path(p).expanduser()


@dataclass(frozen=True)
class DatasetInfo:
    """Static dataset description. For multi-input datasets ``inputs`` maps input names to
    channel counts and ``mean``/``std`` are dicts keyed by input name (in stored units, filled
    from the prepared cache's ``meta.json`` when computed from the training split).

    ``vectors`` declares non-raster (metadata) inputs, ``{name: {"fields": {...}}}`` in
    ManySeg's vector-input format; their values come from the dataset module's
    ``load_vectors`` (``(N, columns)`` float32, NaN = missing)."""

    name: str
    num_classes: int
    in_channels: int
    class_names: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    ignore_index: int | None = None
    license: str = ""
    source: str = ""
    splits: tuple[str, ...] = ("train", "val", "test")
    test_splits: tuple[str, ...] = ("test",)
    inputs: dict | None = None
    vectors: dict | None = None
    timeseries: dict | None = None  # {input: T}: (N, T, H, W, C) arrays in the cache
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Split:
    images: np.ndarray | dict[str, np.ndarray]  # (N, H, W, C) or {input: (N, H, W, C)}
    masks: np.ndarray | dict[str, np.ndarray]  # (N, H, W) uint8, or {name: (N, [K,] H, W)}
    names: list[str]
    vectors: dict[str, np.ndarray] | None = None  # {name: (N, columns) float32}


def save_split(cache: Path, split: str, images, masks: np.ndarray, names: list[str]) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    if isinstance(images, dict):
        for k, v in images.items():
            np.save(cache / f"{split}_images.{k}.npy", v)
    else:
        np.save(cache / f"{split}_images.npy", images)
    # one label representation per split: files of another (or of dropped entries) would be
    # loaded instead of / with the new labels
    for f in [cache / f"{split}_masks.npy", *cache.glob(f"{split}_masks.*.npy")]:
        f.unlink(missing_ok=True)
    if isinstance(masks, dict):
        for k, v in masks.items():
            np.save(cache / f"{split}_masks.{k}.npy", v)
    else:
        np.save(cache / f"{split}_masks.npy", masks)
    (cache / f"{split}_names.json").write_text(json.dumps(names))


def load_split(cache: Path, split: str, mmap: bool = True) -> Split:
    mode = "r" if mmap else None
    masks, names = cache / f"{split}_masks.npy", cache / f"{split}_names.json"
    mask_files = sorted(cache.glob(f"{split}_masks.*.npy"))
    single = cache / f"{split}_images.npy"
    multi = sorted(cache.glob(f"{split}_images.*.npy"))
    if not (masks.exists() or mask_files) or not names.exists() or not (single.exists() or multi):
        raise FileNotFoundError(f"dataset split {split!r} not prepared in {cache}. "
                                "Run `manyseg-bench prepare <dataset>`.")
    if single.exists():
        images = np.load(single, mmap_mode=mode)
    else:
        images = {f.name[len(split) + 8:-4]: np.load(f, mmap_mode=mode) for f in multi}
    labels = (np.load(masks, mmap_mode=mode) if masks.exists() else
              {f.name[len(split) + 7:-4]: np.load(f, mmap_mode=mode) for f in mask_files})
    return Split(images, labels, json.loads(names.read_text()))


def channel_stats(images: np.ndarray, chunk: int = 16) -> tuple[list[float], list[float]]:
    """Per-channel NaN-aware mean and std (+1e-6, as in dataset preparation) of ``(N, H, W, C)``
    images, accumulated in float64 over ``chunk`` samples at a time (bounded memory). uint8
    images are reported in [0, 1] units, as ``DatasetInfo.mean`` / the loader expect."""
    c = images.shape[-1]
    s, ss, n = np.zeros(c), np.zeros(c), np.zeros(c)
    for i in range(0, images.shape[0], chunk):
        x = images[i:i + chunk].reshape(-1, c).astype(np.float64)
        ok = np.isfinite(x)
        x = np.where(ok, x, 0.0)
        s += x.sum(0)
        ss += (x * x).sum(0)
        n += ok.sum(0)
    mean = s / np.maximum(n, 1)
    std = np.sqrt(np.maximum(ss / np.maximum(n, 1) - mean ** 2, 0.0))
    unit = 255.0 if images.dtype == np.uint8 else 1.0
    return (mean / unit).tolist(), (std / unit + 1e-6).tolist()


def load_meta(cache: Path) -> dict:
    p = cache / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}
