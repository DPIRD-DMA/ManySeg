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

Metadata (``sen1floods11_meta*`` variants, read from the raw download; nothing is cached):

* per chip: ``latlon``, the chip centre from the GeoTIFF georeferencing (EPSG:4326 tie point
  and pixel scale);
* per flood event, from ``Sen1Floods11_Metadata.geojson``: ``doy`` (day of year of the S2
  acquisition) and ``orbit`` (S1 pass, 0 = ascending, 1 = descending). The catalogue has dates
  but no times, and neither it nor the GeoTIFF tags have sun or view angles.

Every field except ``latlon`` is constant within an event, and ``latlon`` clusters by event, so
on the official splits (which share events) metadata can identify the event. Use
``event_folds`` (leave-events-out) whenever metadata is an input.
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
# For remote-sensing weights (manyseg.weights): S2 L1C cached as DN / 1e4 in this order
# (checked on the cache: index 9 mean 0.032 = B09 water vapour, index 10 mean 0.005 = B10
# cirrus); S1 in dB.
BAND_NAMES = {"s2": ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09",
                     "B10", "B11", "B12"), "s1": ("VV", "VH")}
DN_PER_UNIT = {"s2": 1e4, "s1": 1.0}
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
INFO_S1 = DatasetInfo(**{**INFO.to_dict(), "name": "sen1floods11_s1", "inputs": {"s1": 2},
                         "in_channels": 2})
INFO_S2 = DatasetInfo(**{**INFO.to_dict(), "name": "sen1floods11_s2", "inputs": {"s2": 13},
                         "in_channels": 13})
META_FIELDS = {"latlon": {"type": "coords"}, "doy": {"type": "cyclic", "period": 365.25},
               "orbit": {"type": "categorical", "num": 2}}
INFO_META = DatasetInfo(**{**INFO.to_dict(), "name": "sen1floods11_meta",
                           "vectors": {"meta": {"fields": META_FIELDS}}})
INFO_META_NOGEO = DatasetInfo(**{
    **INFO.to_dict(), "name": "sen1floods11_meta_nogeo",
    "vectors": {"meta": {"fields": {k: v for k, v in META_FIELDS.items() if k != "latlon"}}}})
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


# --- metadata and event-level splits ----------------------------------------------------------

EVENT_ALIASES = {"Mekong": "Cambodia"}  # chip-name prefix -> catalogue "location"


def event_of(name: str) -> str:
    """Flood event of a chip name such as ``Sri-Lanka_120804``."""
    return name.rsplit("_", 1)[0]


def event_table(root: Path) -> dict[str, dict]:
    """``{event: catalogue properties}`` from ``Sen1Floods11_Metadata.geojson``."""
    feats = json.loads((root / "Sen1Floods11_Metadata.geojson").read_text())["features"]
    by_loc = {f["properties"]["location"]: f["properties"] for f in feats}
    return {**by_loc, **{a: by_loc[t] for a, t in EVENT_ALIASES.items() if t in by_loc}}


def chip_centre(path: Path) -> tuple[float, float]:
    """(lat, lon) of a GeoTIFF's centre from its tie point and pixel scale (EPSG:4326)."""
    import tifffile

    with tifffile.TiffFile(path) as t:
        p = t.pages[0]
        sx, sy = p.tags["ModelPixelScaleTag"].value[:2]
        tie = p.tags["ModelTiepointTag"].value
        h, w = p.shape[-2:]
    i, j, lon0, lat0 = tie[0], tie[1], tie[3], tie[4]
    return lat0 - (h / 2 - j) * sy, lon0 + (w / 2 - i) * sx


def _doy(date: str) -> float:
    import datetime

    return float(datetime.date(*map(int, date.split("/"))).timetuple().tm_yday)


def _field_values(field: str, name: str, root: Path, events: dict) -> list[float]:
    ev = events.get(event_of(name))
    if field == "latlon":
        return list(chip_centre(root / "data" / "S2L1CHand" / f"{name}_S2Hand.tif"))
    if ev is None:
        return [float("nan")]
    if field == "doy":
        return [_doy(ev["s2_date"])]
    if field == "orbit":
        return [{"ASCENDING": 0.0, "DESCENDING": 1.0}.get(ev["orbit"], float("nan"))]
    raise ValueError(f"no extractor for Sen1Floods11 metadata field {field!r}")


def load_vectors(raw: Path, names: list[str], vectors: dict) -> dict[str, np.ndarray]:
    """``{vector input: (N, columns) float32}`` for chips ``names`` (columns in field order;
    NaN where the catalogue has no entry)."""
    root = _root(Path(raw))
    events = event_table(root)
    out = {}
    for vname, cfg in vectors.items():
        rows = [[v for f in cfg["fields"] for v in _field_values(f, n, root, events)]
                for n in names]
        out[vname] = np.asarray(rows, dtype=np.float32)
    return out


def event_folds(names: list[str], k: int = 5, fold: int = 0) -> dict[str, list[int]]:
    """Leave-events-out split of ``names`` (the pooled official train+val+test chips).

    Events are sorted by name; fold ``f`` tests on events ``f, f+k, ...`` and validates on the
    next fold's events; the rest train. Across ``k`` folds every event is tested exactly once,
    and no event ever contributes chips to two of train/val/test.
    """
    events = sorted({event_of(n) for n in names})
    if not 0 <= fold < k or k < 3 or len(events) < k:
        raise ValueError(f"need 0 <= fold < k, k >= 3 and >= k events; got fold={fold}, k={k}, "
                         f"{len(events)} events")
    test, val = set(events[fold::k]), set(events[(fold + 1) % k::k])
    part = {"train": [], "val": [], "test": []}
    for i, n in enumerate(names):
        e = event_of(n)
        part["test" if e in test else "val" if e in val else "train"].append(i)
    return part
