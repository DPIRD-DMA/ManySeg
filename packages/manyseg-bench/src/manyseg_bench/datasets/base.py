"""Prepared datasets are uint8 arrays cached on disk: images (N, H, W, C), masks (N, H, W)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


def data_root(path: str | os.PathLike | None = None) -> Path:
    """Resolve the data root: explicit path > $MANYSEG_BENCH_DATA > ~/.cache/manyseg-bench."""
    p = path or os.environ.get("MANYSEG_BENCH_DATA") or Path.home() / ".cache" / "manyseg-bench"
    return Path(p).expanduser()


@dataclass(frozen=True)
class DatasetInfo:
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
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Split:
    images: np.ndarray  # (N, H, W, C) uint8
    masks: np.ndarray   # (N, H, W) uint8
    names: list[str]


def save_split(cache: Path, split: str, images: np.ndarray, masks: np.ndarray,
               names: list[str]) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / f"{split}_images.npy", images)
    np.save(cache / f"{split}_masks.npy", masks)
    (cache / f"{split}_names.json").write_text(json.dumps(names))


def load_split(cache: Path, split: str, mmap: bool = True) -> Split:
    mode = "r" if mmap else None
    files = [cache / f"{split}_{k}" for k in ("images.npy", "masks.npy", "names.json")]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"dataset split not prepared; missing {missing}. "
                                "Run `manyseg-bench prepare <dataset>`.")
    return Split(np.load(files[0], mmap_mode=mode), np.load(files[1], mmap_mode=mode),
                 json.loads(files[2].read_text()))
