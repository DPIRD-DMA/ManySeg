"""EuroSAT multispectral (Helber et al. 2019). License: MIT.

Image classification, 10 land-use / land-cover classes, 27,000 Sentinel-2 L1C patches of
64x64 px at 10 m with all 13 bands (the 20 m and 60 m bands are upsampled in the source), in
the source band order B01 B02 B03 B04 B05 B06 B07 B08 B09 B10 B11 B12 B8A.

Splits: TorchGeo's *spatial* split (``eurosat-spatial-{train,val,test}.txt``, 16,200 / 5,400
/ 5,400), which separates patches geographically; the random split leaks neighbouring patches
between train and test. Raw data: TorchGeo's mirror of ``EuroSATallBands.zip`` (identical to
the Zenodo release, much faster).

Preprocessing (identical for every model): digital numbers / 1e4 as float16; per-band
mean/std from the training split in ``meta.json``. Training: random-resized crops back to
64x64 with h/v flips (overhead imagery).
"""

from __future__ import annotations

import io
import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np

from .base import DatasetInfo, channel_stats, check_native_size, save_split

HF = "https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/"
CLASSES = ("AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture",
           "PermanentCrop", "Residential", "River", "SeaLake")
BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12",
         "B8A")
EXPECTED = {"train": 16200, "val": 5400, "test": 5400}

INFO = DatasetInfo(
    name="eurosat",
    num_classes=10,
    in_channels=13,
    class_names=CLASSES,
    mean=(), std=(),  # filled from meta.json
    license="MIT",
    source="https://github.com/phelber/EuroSAT (TorchGeo spatial split)",
    notes=("TorchGeo spatial split", "bands: " + " ".join(BANDS)),
)
TASK = "classification"
LOADER_KWARGS = {"vflip": True}
EVAL_BATCH_SIZE = 256
SIZE = 64  # patches as published (prepare does not resize)


def download(raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    files = ["EuroSATallBands.zip"] + [f"eurosat-spatial-{s}.txt" for s in EXPECTED]
    for f in files:
        if not (raw / f).exists():
            subprocess.run(["curl", "-sSfL", "-o", str(raw / f), HF + f], check=True)


def _split_names(raw: Path, split: str) -> list[str]:
    lines = (raw / f"eurosat-spatial-{split}.txt").read_text().split()
    return [ln.removesuffix(".jpg").removesuffix(".tif") for ln in lines if ln.strip()]


def label_of(name: str) -> int:
    return CLASSES.index(name.rsplit("_", 1)[0])


def prepare(raw: Path, cache: Path, size: int | None = None) -> dict:
    """Cache the 13-band patches at their published ``SIZE`` px; another ``size`` raises
    (before any download)."""
    import tifffile

    check_native_size(size, SIZE, "eurosat")
    size = SIZE

    raw, cache = Path(raw), Path(cache)
    if not (raw / "EuroSATallBands.zip").exists():
        download(raw)
    with zipfile.ZipFile(raw / "EuroSATallBands.zip") as z:
        members = {Path(m).stem: m for m in z.namelist() if m.endswith(".tif")}
        summary, stats = {}, {}
        for split, n_expected in EXPECTED.items():
            names = _split_names(raw, split)
            if len(names) != n_expected:
                raise RuntimeError(f"integrity: {split} lists {len(names)}, expected {n_expected}")
            missing = [n for n in names if n not in members]
            if missing:
                raise RuntimeError(f"integrity: {len(missing)} {split} patches missing from the "
                                   f"archive, e.g. {missing[:3]}")
            images = np.empty((len(names), size, size, 13), np.float16)
            for i, n in enumerate(names):
                a = tifffile.imread(io.BytesIO(z.read(members[n])))
                if a.shape != (size, size, 13):
                    raise RuntimeError(f"integrity: {n} has shape {a.shape}")
                images[i] = a.astype(np.float32) / 1e4
            labels = np.array([label_of(n) for n in names], np.int64)
            if split == "train":
                stats["mean"], stats["std"] = channel_stats(images)
            save_split(cache, split, images, labels, names)
            summary[split] = {"n": len(names),
                              "class_counts": np.bincount(labels, minlength=10).tolist()}
    (cache / "meta.json").write_text(json.dumps({**stats, "summary": summary}, indent=1))
    return summary
