"""Sen1Floods11 hand-labelled subset (Bonafilia et al. 2020). License: CC BY 4.0.

Multi-input flood mapping: Sentinel-1 GRD (VV, VH; dB), Sentinel-2 L1C (13 bands) and a
Copernicus DEM, 512x512 chips at 10 m, binary water labels (-1 = no data -> ignored).

Splits are the official ``flood_{train,valid,test,bolivia}_data.txt`` lists. Chips from the same
flood events occur in train, valid and test (spatially correlated), so the Bolivia event, which
is held out entirely, is reported as a separate out-of-distribution test split.

Preprocessing (identical for every model): S2 reflectance / 1e4; S1 dB as is; DEM minus its
per-chip median (relative relief); per-channel mean/std computed on the training split only and
stored in ``meta.json``. Training uses random-resized 256x256 crops with h/v flips; evaluation
uses full 512x512 chips.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import DatasetInfo, save_split

HF_URL = ("https://huggingface.co/datasets/blumenstiel/Sen1Floods11/resolve/main/"
          "sen1floods11_v1_1.tar.gz")
SPLIT_FILES = {"train": "flood_train_data.txt", "val": "flood_valid_data.txt",
               "test": "flood_test_data.txt", "bolivia": "flood_bolivia_data.txt"}
EXPECTED = {"train": 252, "val": 89, "test": 90, "bolivia": 15}
INPUTS = {"s1": 2, "s2": 13}
ALL_INPUTS = {"s1": 2, "s2": 13, "dem": 1}

INFO = DatasetInfo(
    name="sen1floods11",
    num_classes=2,
    in_channels=sum(INPUTS.values()),
    class_names=("land", "water"),
    mean={}, std={},  # filled from meta.json
    ignore_index=255,
    license="CC BY 4.0",
    source="https://github.com/cloudtostreet/Sen1Floods11",
    splits=("train", "val", "test", "bolivia"),
    test_splits=("test", "bolivia"),
    inputs=INPUTS,
    notes=("train/val/test share flood events; bolivia is an event-level holdout",),
)
INFO_DEM = DatasetInfo(**{**INFO.to_dict(), "name": "sen1floods11_dem",
                          "inputs": ALL_INPUTS, "in_channels": sum(ALL_INPUTS.values())})
LOADER_KWARGS = {"out_size": (256, 256), "scale": (0.25, 1.0), "vflip": True}
EVAL_BATCH_SIZE = 8


def _root(raw: Path) -> Path:
    return raw / "sen1floods11_v1.1"


def download(raw: Path) -> None:
    import subprocess

    raw.mkdir(parents=True, exist_ok=True)
    tar = raw / "sen1floods11_v1_1.tar.gz"
    if not tar.exists():
        subprocess.run(["curl", "-sSfL", "-o", str(tar), HF_URL], check=True)
    subprocess.run(["tar", "xzf", tar.name], cwd=raw, check=True)


def _read_chip(root: Path, name: str):
    import tifffile

    d = root / "data"
    s1 = tifffile.imread(d / "S1GRDHand" / f"{name}_S1Hand.tif").astype(np.float32)
    s2 = tifffile.imread(d / "S2L1CHand" / f"{name}_S2Hand.tif").astype(np.float32) / 1e4
    dem = tifffile.imread(d / "CopernicusDEM" / f"{name}_DEM.tif").astype(np.float32)
    dem = (dem - np.median(dem))[None]
    lab = tifffile.imread(d / "LabelHand" / f"{name}_LabelHand.tif")
    if set(np.unique(lab).tolist()) - {-1, 0, 1}:
        raise RuntimeError(f"integrity: unexpected label values in {name}")
    lab = np.where(lab < 0, 255, lab).astype(np.uint8)
    hwc = {k: np.moveaxis(v, 0, -1) for k, v in (("s1", s1), ("s2", s2), ("dem", dem))}
    return hwc, lab


def prepare(raw: Path, cache: Path, size: int = 512, workers: int = 4) -> dict:
    raw, cache = Path(raw), Path(cache)
    root = _root(raw)
    if not (root / "splits").exists():
        download(raw)
    summary, stats = {}, None
    for split, fname in SPLIT_FILES.items():
        names = [n.strip() for n in (root / "splits" / fname).read_text().split("\n")
                 if n.strip()]
        if len(names) != EXPECTED[split]:
            raise RuntimeError(f"integrity: {split} has {len(names)} chips, "
                               f"expected {EXPECTED[split]}")
        arrays = {k: np.empty((len(names), size, size, c), np.float16)
                  for k, c in ALL_INPUTS.items()}
        masks = np.empty((len(names), size, size), np.uint8)
        for i, n in enumerate(names):
            chip, lab = _read_chip(root, n)
            for k in arrays:
                arrays[k][i] = chip[k]
            masks[i] = lab
        if split == "train":
            stats = {"mean": {}, "std": {}}
            for k, a in arrays.items():
                flat = a.reshape(-1, a.shape[-1]).astype(np.float32)
                stats["mean"][k] = np.nanmean(flat, 0).tolist()
                stats["std"][k] = (np.nanstd(flat, 0) + 1e-6).tolist()
        save_split(cache, split, arrays, masks, names)
        valid = masks != 255
        summary[split] = {"n": len(names), "water_frac": float((masks[valid] == 1).mean()),
                          "ignored_frac": float(1 - valid.mean()),
                          "nan_frac": {k: float(np.isnan(v.astype(np.float32)).mean())
                                       for k, v in arrays.items()}}
    (cache / "meta.json").write_text(json.dumps({**stats, "summary": summary}, indent=1))
    return summary
