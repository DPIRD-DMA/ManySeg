"""Change-detection datasets: before / after images as inputs ``t1`` and ``t2`` on one grid.

* ``levircd`` — LEVIR-CD (Chen & Shi 2020): building change, 0.5 m RGB, 637 pairs of 1024x1024
  (official 445 / 64 / 128), tiled into non-overlapping 256x256 tiles (7,120 / 1,024 / 2,048).
  Binary change (task ``change``, primary metric change-class F1). Licence: research use
  (see the LEVIR-CD terms). Source: the torchange parquet copy (``EVER-Z/torchange_levircd``).
* ``second`` — SECOND (Yang et al. 2021): semantic change, 0.5–3 m aerial RGB, 512x512 pairs,
  land cover at both dates labelled where change happened (ids 1-6: water, ground, low
  vegetation, tree, building, playground; 0 = unchanged) plus a change mask. torchange's
  train / test (2,968 / 1,694); val = the sha1 10% of train. Labels are a ``(3, H, W)`` stack
  [land cover t1, land cover t2, change] (task ``semantic_change``, primary: SECOND Score).
  Source: ``EVER-Z/torchange_second``.
* ``oscd`` — OSCD (Daudt et al. 2018): urban change from Sentinel-2 L1C, 13 bands resampled to
  10 m (``imgs_*_rect``), 24 city pairs (official 14 train / 10 test cities); val = 3 train
  cities (``OSCD_VAL``, chosen by sha1, fixed before any run). Cities are tiled into 128x128
  tiles (edges padded, padding ignored: 255). Labels 1 / 2 -> 0 / 1. Reflectance = DN / 1e4.
  Licence: CC BY-NC-SA 4.0. Source: the original release (``hkristen/oscd`` mirror).
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from .base import DatasetInfo, channel_stats, check_native_size, save_split

RGB = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def is_val(name: str) -> bool:
    return int(hashlib.sha1(name.encode()).hexdigest(), 16) % 10 == 0


def tile_positions(h: int, w: int, size: int) -> list[tuple[int, int]]:
    return [(y, x) for y in range(0, h, size) for x in range(0, w, size)]


def cut(a: np.ndarray, y: int, x: int, size: int, fill) -> np.ndarray:
    """``a[y:y+size, x:x+size]`` (H, W[, C]), padded with ``fill`` at the image edge."""
    t = a[y:y + size, x:x + size]
    pad = [(0, size - t.shape[0]), (0, size - t.shape[1])] + [(0, 0)] * (a.ndim - 2)
    return np.pad(t, pad, constant_values=fill) if any(p[1] for p in pad[:2]) else t


def _png(d) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(d["bytes"])))


def _parquet_rows(raw: Path, split: str):
    import pyarrow.parquet as pq

    for f in sorted((Path(raw) / "data").glob(f"{split}-*.parquet")):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=16):
            yield from batch.to_pylist()


# --- LEVIR-CD --------------------------------------------------------------------------------

LEVIR_INFO = DatasetInfo(
    name="levircd", num_classes=2, in_channels=6, class_names=("no change", "change"),
    mean={"t1": RGB["mean"], "t2": RGB["mean"]}, std={"t1": RGB["std"], "t2": RGB["std"]},
    license="LEVIR-CD terms (research use)", source="https://justchenhao.github.io/LEVIR/",
    inputs={"t1": 3, "t2": 3}, notes=("1024x1024 pairs tiled into 256x256",))
LEVIR_SPLITS = {"train": "train", "val": "val", "test": "test"}
LEVIR_EXPECTED = {"train": 445, "val": 64, "test": 128}


def prepare_levircd(raw: Path, cache: Path, size: int = 256, expected=LEVIR_EXPECTED) -> dict:
    summary = {}
    for split, src in LEVIR_SPLITS.items():
        a, b, m, names = [], [], [], []
        pairs = 0
        for row in _parquet_rows(raw, src):
            pairs += 1
            t1, t2 = _png(row["t1_image"])[..., :3], _png(row["t2_image"])[..., :3]
            cm = (_png(row["change_mask"]) > 127).astype(np.uint8)
            for y, x in tile_positions(*cm.shape, size):
                a.append(cut(t1, y, x, size, 0))
                b.append(cut(t2, y, x, size, 0))
                m.append(cut(cm, y, x, size, 255))
                names.append(f"{Path(row['image_name']).stem}_{y}_{x}")
        if expected and pairs != expected[split]:
            raise RuntimeError(f"integrity: {split} has {pairs} pairs, expected {expected[split]}")
        masks = np.stack(m)
        save_split(Path(cache), split, {"t1": np.stack(a), "t2": np.stack(b)}, masks, names)
        summary[split] = {"pairs": pairs, "tiles": len(names),
                          "change_frac": float((masks == 1).mean())}
    (Path(cache) / "meta.json").write_text(json.dumps({"summary": summary}, indent=1))
    return summary


# --- SECOND ----------------------------------------------------------------------------------

SECOND_CLASSES = ("water", "ground", "low vegetation", "tree", "building", "playground")
SECOND_INFO = DatasetInfo(
    name="second", num_classes=len(SECOND_CLASSES), in_channels=6, class_names=SECOND_CLASSES,
    mean={"t1": RGB["mean"], "t2": RGB["mean"]}, std={"t1": RGB["std"], "t2": RGB["std"]},
    license="SECOND terms (research use)", source="https://captain-whu.github.io/SCD/",
    inputs={"t1": 3, "t2": 3},
    notes=("labels: (3, H, W) = land cover t1, land cover t2 (0 = unchanged), change",
           "val = sha1 10% of torchange train"))


def prepare_second(raw: Path, cache: Path, size: int | None = None) -> dict:
    """Cache SECOND's 512 x 512 pairs as published; another ``size`` raises."""
    check_native_size(size, 512, "second")
    from numpy.lib.format import open_memmap

    Path(cache).mkdir(parents=True, exist_ok=True)
    summary = {}
    for src in ("train", "test"):
        rows = [(r["image_name"], r) for r in _parquet_rows(raw, src)]
        groups = ({"train": [r for r in rows if not is_val(r[0])],
                   "val": [r for r in rows if is_val(r[0])]} if src == "train"
                  else {"test": rows})
        for split, items in groups.items():
            n = len(items)
            imgs = {k: open_memmap(Path(cache) / f"{split}_images.{k}.npy", "w+", np.uint8,
                                   (n, 512, 512, 3)) for k in ("t1", "t2")}
            labels = open_memmap(Path(cache) / f"{split}_masks.npy", "w+", np.uint8,
                                 (n, 3, 512, 512))
            for i, (_, r) in enumerate(items):
                imgs["t1"][i] = _png(r["t1_image"])[..., :3]
                imgs["t2"][i] = _png(r["t2_image"])[..., :3]
                lc1, lc2 = _png(r["t1_mask"]), _png(r["t2_mask"])
                labels[i] = np.stack([lc1, lc2, (_png(r["change_mask"]) > 127)])
            for v in (*imgs.values(), labels):
                v.flush()
            (Path(cache) / f"{split}_names.json").write_text(
                json.dumps([Path(nm).stem for nm, _ in items]))
            changed = labels[:, 2].astype(bool)
            agree = float(((labels[:, 0] > 0) == changed).mean())
            summary[split] = {"n": n, "change_frac": float(changed.mean()),
                              "lc_vs_change_agreement": agree}
            del imgs, labels
    (Path(cache) / "meta.json").write_text(json.dumps({"summary": summary}, indent=1))
    return summary


# --- OSCD ------------------------------------------------------------------------------------

OSCD_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10",
              "B11", "B12")
OSCD_TRAIN = ("aguasclaras", "bercy", "bordeaux", "nantes", "paris", "rennes", "saclay_e",
              "abudhabi", "cupertino", "pisa", "beihai", "hongkong", "beirut", "mumbai")
OSCD_TEST = ("brasilia", "montpellier", "norcia", "rio", "saclay_w", "valencia", "dubai",
             "lasvegas", "milano", "chongqing")
OSCD_VAL = tuple(sorted(OSCD_TRAIN, key=lambda c: hashlib.sha1(c.encode()).hexdigest())[:3])
OSCD_INFO = DatasetInfo(
    name="oscd", num_classes=2, in_channels=26, class_names=("no change", "change"),
    mean={}, std={},  # filled from meta.json
    ignore_index=255, license="CC BY-NC-SA 4.0",
    source="https://rcdaudt.github.io/oscd/", inputs={"t1": 13, "t2": 13},
    notes=(f"val cities {', '.join(OSCD_VAL)}", "128x128 tiles, padding ignored"))
_IMG = "Onera Satellite Change Detection dataset - Images/"


def _oscd_city(zi, zl, label_root: str, city: str):
    import tifffile

    def bands(d):
        return np.stack([tifffile.imread(io.BytesIO(zi.read(f"{_IMG}{city}/{d}/{b}.tif")))
                         for b in OSCD_BANDS], -1).astype(np.float32) / 1e4

    t1, t2 = bands("imgs_1_rect"), bands("imgs_2_rect")
    cm = tifffile.imread(io.BytesIO(zl.read(f"{label_root}/{city}/cm/{city}-cm.tif")))
    if t1.shape != t2.shape or t1.shape[:2] != cm.shape:
        raise RuntimeError(f"integrity: {city} shapes {t1.shape} {t2.shape} {cm.shape}")
    return t1, t2, (cm == 2).astype(np.uint8)


def prepare_oscd(raw: Path, cache: Path, size: int = 128) -> dict:
    raw = Path(raw)
    zi = zipfile.ZipFile(raw / "Onera Satellite Change Detection dataset - Images.zip")
    labels = {
        "train": (zipfile.ZipFile(raw / "Onera Satellite Change Detection dataset - Train "
                                  "Labels.zip"),
                  "Onera Satellite Change Detection dataset - Train Labels"),
        "test": (zipfile.ZipFile(raw / "Onera Satellite Change Detection dataset - Test "
                                 "Labels.zip"),
                 "Onera Satellite Change Detection dataset - Test Labels")}
    cities = {"train": [c for c in OSCD_TRAIN if c not in OSCD_VAL], "val": list(OSCD_VAL),
              "test": list(OSCD_TEST)}
    summary, stats = {}, {}
    for split, cs in cities.items():
        zl, root = labels["test" if split == "test" else "train"]
        a, b, m, names = [], [], [], []
        for c in cs:
            t1, t2, cm = _oscd_city(zi, zl, root, c)
            for y, x in tile_positions(*cm.shape, size):
                a.append(cut(t1, y, x, size, 0).astype(np.float16))
                b.append(cut(t2, y, x, size, 0).astype(np.float16))
                m.append(cut(cm, y, x, size, 255))
                names.append(f"{c}_{y}_{x}")
        imgs = {"t1": np.stack(a), "t2": np.stack(b)}
        masks = np.stack(m)
        if split == "train":
            stats["mean"], stats["std"] = {}, {}
            for k, v in imgs.items():
                stats["mean"][k], stats["std"][k] = channel_stats(v)
        save_split(Path(cache), split, imgs, masks, names)
        valid = masks != 255
        summary[split] = {"cities": cs, "tiles": len(names),
                          "change_frac": float((masks[valid] == 1).mean())}
    (Path(cache) / "meta.json").write_text(json.dumps({**stats, "summary": summary}, indent=1))
    return summary


class _Dataset:
    """A dataset entry (``INFO``, ``TASK``, ``prepare``, loader settings) without a module."""

    def __init__(self, **attrs):
        self.__dict__.update(attrs)


LEVIRCD = _Dataset(INFO=LEVIR_INFO, TASK="change", CACHE_NAME="levircd", prepare=prepare_levircd,
                   LOADER_KWARGS={"vflip": True}, EVAL_BATCH_SIZE=16)
SECOND = _Dataset(INFO=SECOND_INFO, TASK="semantic_change", CACHE_NAME="second",
                  prepare=prepare_second, EVAL_BATCH_SIZE=8,
                  LOADER_KWARGS={"vflip": True, "device_resident": False})
OSCD = _Dataset(INFO=OSCD_INFO, TASK="change", CACHE_NAME="oscd", prepare=prepare_oscd,
                LOADER_KWARGS={"vflip": True}, EVAL_BATCH_SIZE=32)
