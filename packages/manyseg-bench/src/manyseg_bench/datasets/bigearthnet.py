"""BigEarthNet v2.0 / reBEN (Clasen et al. 2024). License: CDLA-Permissive-1.0.

Multi-label land-cover classification (19 classes), Sentinel-2 L2A (12 bands) + Sentinel-1
(VV, VH), 120x120 px patches at 10 m, with pixel reference maps (CORINE 2018 level-3 codes),
10 European countries, acquisition dates.

Raw data:

* ``metadata.parquet`` (official; patches without seasonal snow, cloud or shadow, 480,038)
  with the official ``train`` / ``validation`` / ``test`` split and labels;
* ``Reference_Maps/`` (official, Zenodo);
* ``BENv2.lmdb``: S1 and S2 patches as safetensors, keyed by patch id / ``s1_name``. This is
  the community LMDB conversion on Hugging Face (``hackelle/BigEarthNetV2-LMDB``, made with the
  authors' rico-hdl tool), used because Zenodo serves the 118 GB of GeoTIFFs at ~0.2 MB/s.

Subset (fixed, pre-registered in E020): the first ``SUBSET[split]`` patches of each official
split ordered by sha1(patch id): 60,000 train, 15,000 val, 30,000 test.

Preprocessing (identical for every model): S2 reflectance / 1e4 and S1 backscatter (dB) as
float16, all bands on the 10 m grid (20 m / 60 m bands repeated 2x / 6x, so a block average
by the same factor recovers the native pixels exactly: ``band_groups``); per-band mean/std
from the training subset. Masks: the 19 classes (``CLC_TO_CLASS``), 255 for CORINE codes
outside the nomenclature (roads, ports, airports, mines, dumps, construction, urban green,
sport, bare rock, burnt areas; 999 = no data). Training augmentation: h/v flips only, so
image-level labels stay valid.

Labels: the official labels equal the classes present in the mapped reference map (checked
at preparation and recorded as ``label_agreement`` in ``meta.json``). ``bigearthnet*``
classification variants use the official labels; ``bigearthnet_seg`` (segmentation +
classification) derives them from the mask; ``bigearthnet_frac`` regresses class fractions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .base import DatasetInfo, channel_stats

CLASSES = (
    "Urban fabric", "Industrial or commercial units", "Arable land", "Permanent crops",
    "Pastures", "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas", "Broad-leaved forest", "Coniferous forest", "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation", "Transitional woodland, shrub",
    "Beaches, dunes, sands", "Inland wetlands", "Coastal wetlands", "Inland waters",
    "Marine waters",
)
# CORINE level-3 code -> class (BigEarthNet 19-class nomenclature; 241 -> permanent crops and
# 423 intertidal flats -> marine waters, verified against the official labels).
_CLC = ((111, 112), (121,), (211, 212, 213), (221, 222, 223, 241), (231,), (242,), (243,),
        (244,), (311,), (312,), (313,), (321, 333), (322, 323), (324,), (331,), (411, 412),
        (421, 422), (511, 512), (423, 521, 522, 523))
CLC_TO_CLASS = {code: i for i, codes in enumerate(_CLC) for code in codes}
S2_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
S2_NATIVE = {"B01": 60, "B09": 60, "B02": 10, "B03": 10, "B04": 10, "B08": 10, "B05": 20,
             "B06": 20, "B07": 20, "B8A": 20, "B11": 20, "B12": 20}
S1_BANDS = ("VV", "VH")
# Resolution groups of the S2 stack for the provider's ``band_groups`` option (E015 style).
BAND_GROUPS = {"s2_10m": {"bands": [1, 2, 3, 7]},
               "s2_20m": {"bands": [4, 5, 6, 8, 10, 11], "factor": 2},
               "s2_60m": {"bands": [0, 9], "factor": 6}}
COUNTRIES = ("Austria", "Belgium", "Finland", "Ireland", "Kosovo", "Lithuania", "Luxembourg",
             "Portugal", "Serbia", "Switzerland")
SPLITS = {"train": "train", "val": "validation", "test": "test"}
SUBSET = {"train": 60000, "val": 15000, "test": 30000}
SIZE = 120
CACHE_NAME = "bigearthnet"

INFO = DatasetInfo(
    name="bigearthnet",
    num_classes=19,
    in_channels=12,
    class_names=CLASSES,
    mean={}, std={},  # filled from meta.json
    license="CDLA-Permissive-1.0",
    source=("https://bigearth.net (reBEN); LMDB: "
            "huggingface.co/datasets/hackelle/BigEarthNetV2-LMDB"),
    inputs={"s2": 12},
    notes=("fixed sha1-ordered subset of the official splits: 60k / 15k / 30k",
           "S2 bands " + " ".join(S2_BANDS) + " on the 10 m grid"),
)
TASK = "multilabel"
LOADER_KWARGS = {"scale": (1.0, 1.0), "ratio": (1.0, 1.0), "vflip": True,
                 "device_resident": False}
EVAL_BATCH_SIZE = 256
META_FIELDS = {"doy": {"type": "cyclic", "period": 365.25},
               "country": {"type": "categorical", "num": len(COUNTRIES)}}


def _variant(name, **kw):
    return DatasetInfo(**{**INFO.to_dict(), "name": name, **kw})


INFO_S1S2 = _variant("bigearthnet_s1s2", inputs={"s2": 12, "s1": 2}, in_channels=14)
INFO_META = _variant("bigearthnet_meta", vectors={"meta": {"fields": META_FIELDS}})
INFO_SEG = _variant("bigearthnet_seg", ignore_index=255)
INFO_FRAC = _variant("bigearthnet_frac")
VARIANTS = (INFO_S1S2, INFO_META, INFO_SEG, INFO_FRAC)


def subset_ids(meta, split: str, n: int | None = None) -> list[str]:
    """The fixed subset of an official split: first ``n`` patch ids by sha1, then sorted by id
    (LMDB key order, for sequential reads)."""
    ids = meta.loc[meta["split"] == SPLITS[split], "patch_id"].tolist()
    ids.sort(key=lambda p: hashlib.sha1(p.encode()).hexdigest())
    return sorted(ids[:SUBSET[split] if n is None else n])


def reference_map_path(raw: Path, pid: str) -> Path:
    tile = pid.rsplit("_", 2)[0]
    return Path(raw) / "Reference_Maps" / tile / pid / f"{pid}_reference_map.tif"


_LUT = np.full(1000, 255, np.uint8)
for _code, _cls in CLC_TO_CLASS.items():
    _LUT[_code] = _cls


def map_codes(codes: np.ndarray) -> np.ndarray:
    """CORINE codes -> class ids (255 outside the nomenclature)."""
    return _LUT[np.clip(codes, 0, 999)]


def presence(masks: np.ndarray, k: int = 19) -> np.ndarray:
    """``(N, H, W)`` class masks -> ``(N, k)`` multi-hot of the classes present."""
    out = np.zeros((len(masks), k), np.uint8)
    for i, m in enumerate(masks):
        u = np.unique(m)
        out[i, u[u < k]] = 1
    return out


def fractions(masks: np.ndarray, k: int = 19, chunk: int = 512) -> np.ndarray:
    """``(N, H, W)`` class masks -> ``(N, k)`` fraction of valid pixels in each class."""
    out = np.zeros((len(masks), k), np.float32)
    for s in range(0, len(masks), chunk):
        m = np.asarray(masks[s:s + chunk]).reshape(-1, masks.shape[1] * masks.shape[2])
        counts = np.stack([(m == c).sum(1) for c in range(k)], 1).astype(np.float32)
        out[s:s + chunk] = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    return out


def multi_hot(labels, k: int = 19) -> np.ndarray:
    out = np.zeros((len(labels), k), np.uint8)
    for i, ls in enumerate(labels):
        out[i, [CLASSES.index(c) for c in ls]] = 1
    return out


def _to_grid(a: np.ndarray) -> np.ndarray:
    """A band at its native size (20, 60 or 120 px) -> 120x120 by pixel repetition."""
    f = SIZE // a.shape[0]
    if a.shape != (SIZE // f, SIZE // f) or SIZE % a.shape[0]:
        raise RuntimeError(f"unexpected band shape {a.shape}")
    return a if f == 1 else np.repeat(np.repeat(a, f, 0), f, 1)


def _read_meta(raw: Path):
    import pandas as pd

    return pd.read_parquet(Path(raw) / "metadata.parquet")


def prepare(raw: Path, cache: Path, size: int = SIZE) -> dict:
    """Stream the subset from the LMDB into memory-mapped ``.npy`` files (bounded RAM)."""
    import lmdb
    import tifffile
    from numpy.lib.format import open_memmap
    from safetensors.numpy import load as st_load

    raw, cache = Path(raw), Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    meta = _read_meta(raw).set_index("patch_id")
    env = lmdb.open(str(raw / "BENv2.lmdb"), readonly=True, lock=False, readahead=False,
                    max_readers=4)
    summary, stats = {}, {"mean": {}, "std": {}}
    for split in SUBSET:
        ids = subset_ids(meta.reset_index(), split)
        n = len(ids)
        s2 = open_memmap(cache / f"{split}_images.s2.npy", "w+", np.float16, (n, SIZE, SIZE, 12))
        s1 = open_memmap(cache / f"{split}_images.s1.npy", "w+", np.float16, (n, SIZE, SIZE, 2))
        masks = open_memmap(cache / f"{split}_masks.npy", "w+", np.uint8, (n, SIZE, SIZE))
        with env.begin() as txn:
            for i, pid in enumerate(ids):
                t2 = st_load(txn.get(pid.encode()))
                t1 = st_load(txn.get(meta.at[pid, "s1_name"].encode()))
                s2[i] = np.stack([_to_grid(t2[b]) for b in S2_BANDS], -1).astype(np.float32) / 1e4
                s1[i] = np.stack([_to_grid(t1[b]) for b in S1_BANDS], -1)
                masks[i] = map_codes(tifffile.imread(reference_map_path(raw, pid)))
        for a in (s2, s1, masks):
            a.flush()
        (cache / f"{split}_names.json").write_text(json.dumps(ids))
        official = multi_hot(meta.loc[ids, "labels"])
        agree = float((presence(masks) == official).all(1).mean())
        if split == "train":
            for k, a in (("s2", s2), ("s1", s1)):
                stats["mean"][k], stats["std"][k] = channel_stats(a, chunk=256)
        summary[split] = {"n": n, "label_agreement": agree,
                          "label_freq": official.mean(0).round(4).tolist(),
                          "ignored_frac": float((masks == 255).mean()),
                          "s1_nonfinite_frac": float((~np.isfinite(s1[:256].astype(np.float32))
                                                      ).mean())}
        del s2, s1, masks
    (cache / "meta.json").write_text(json.dumps({**stats, "summary": summary}, indent=1))
    return summary


def split_labels(raw: Path, split) -> np.ndarray | None:
    """Official multi-hot labels for classification variants (``None``: keep the masks)."""
    meta = _read_meta(raw).set_index("patch_id")
    return multi_hot(meta.loc[split.names, "labels"])


def load_vectors(raw: Path, names: list[str], vectors: dict) -> dict[str, np.ndarray]:
    """``doy`` (day of year of the S2 acquisition, from the patch id) and ``country`` (index
    into ``COUNTRIES``), columns in field order."""
    from datetime import date

    meta = _read_meta(raw).set_index("patch_id")
    out = {}
    for vname, cfg in vectors.items():
        cols = []
        for f in cfg["fields"]:
            if f == "doy":
                d = [n.split("_")[2][:8] for n in names]  # S2A_MSIL2A_YYYYMMDDT..._...
                cols.append([date(int(s[:4]), int(s[4:6]), int(s[6:])).timetuple().tm_yday
                             for s in d])
            elif f == "country":
                cols.append([COUNTRIES.index(c) for c in meta.loc[names, "country"]])
            else:
                raise ValueError(f"unknown BigEarthNet metadata field {f!r}")
        out[vname] = np.asarray(cols, np.float32).T
    return out


def fraction_labels(raw: Path, split) -> np.ndarray:
    """Per-patch class fractions of the valid (mapped) pixels: ``(N, 19)`` regression targets."""
    return fractions(split.masks)


VARIANT_OVERRIDES = {
    "bigearthnet_seg": {"TASK": "seg_multilabel", "split_labels": None},
    "bigearthnet_frac": {"TASK": "regression", "split_labels": fraction_labels},
}
