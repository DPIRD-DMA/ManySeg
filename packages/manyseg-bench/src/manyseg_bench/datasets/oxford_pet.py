"""Oxford-IIIT Pet trimap segmentation (Parkhi et al. 2012). License: CC BY-SA 4.0.

Classes: pet (trimap 1), background (2), boundary (3). Official ``trainval``/``test`` lists.
Validation = deterministic 10% of ``trainval`` (sha1 of image name), so ``test`` is only
touched for final reporting. Images are resized to a fixed square (bilinear, antialiased;
masks nearest) once at preparation; every provider then sees identical pixels.
"""

from __future__ import annotations

import hashlib
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from .base import DatasetInfo, save_split

URLS = {
    "images.tar.gz": "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
    "annotations.tar.gz": "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
}
EXPECTED_COUNTS = {"trainval": 3680, "test": 3669}

INFO = DatasetInfo(
    name="oxford_pet",
    num_classes=3,
    in_channels=3,
    class_names=("pet", "background", "boundary"),
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    license="CC BY-SA 4.0",
    source="https://www.robots.ox.ac.uk/~vgg/data/pets/",
    notes=("val = sha1(name) % 10 == 0 subset of trainval", "fixed square resize"),
)


def is_val(name: str) -> bool:
    return int(hashlib.sha1(name.encode()).hexdigest(), 16) % 10 == 0


def download(raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    for fname, url in URLS.items():
        if not (raw / fname).exists():
            subprocess.run(["curl", "-sSfL", "-o", str(raw / fname), url], check=True)
    for fname in URLS:
        subprocess.run(["tar", "xzf", fname], cwd=raw, check=True)


def _read_list(raw: Path, split: str) -> list[str]:
    lines = (raw / "annotations" / f"{split}.txt").read_text().split("\n")
    return [ln.split()[0] for ln in lines if ln.strip()]


def _load_one(args):
    raw, name, size = args
    img = Image.open(raw / "images" / f"{name}.jpg").convert("RGB")
    tri = Image.open(raw / "annotations" / "trimaps" / f"{name}.png")
    img = np.asarray(img.resize((size, size), Image.BILINEAR, reducing_gap=None))
    m = np.asarray(tri.resize((size, size), Image.NEAREST))
    return img, m


def prepare(raw: Path, cache: Path, size: int = 256, workers: int = 4) -> dict:
    raw, cache = Path(raw), Path(cache)
    if not (raw / "annotations" / "trainval.txt").exists():
        download(raw)
    lists = {s: _read_list(raw, s) for s in ("trainval", "test")}
    for s, n in EXPECTED_COUNTS.items():
        if len(lists[s]) != n:
            raise RuntimeError(f"integrity: {s} has {len(lists[s])} entries, expected {n}")
    splits = {
        "train": [n for n in lists["trainval"] if not is_val(n)],
        "val": [n for n in lists["trainval"] if is_val(n)],
        "test": lists["test"],
    }
    summary = {}
    with ProcessPoolExecutor(workers) as ex:
        for split, names in splits.items():
            out = list(ex.map(_load_one, [(raw, n, size) for n in names], chunksize=32))
            imgs = np.stack([o[0] for o in out])
            masks = np.stack([o[1] for o in out])
            vals = set(np.unique(masks).tolist())
            if not vals <= {1, 2, 3}:
                raise RuntimeError(f"integrity: unexpected trimap values {vals} in {split}")
            save_split(cache, split, imgs, (masks - 1).astype(np.uint8), names)
            summary[split] = {"n": len(names), "class_pixel_frac": [
                float((masks == v).mean()) for v in (1, 2, 3)]}
    return summary
