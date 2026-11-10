"""AI4Boundaries, aerial orthophoto subset (d'Andrimont et al. 2023). Licence: CC BY 4.0, (c) EU.

Field delineation as boundary + distance instances. Tiles are 512x512 RGB at 1 m (EPSG:3035)
from seven regions (AT, ES-Catalonia, FR, LU, NL, SE, SI); labels come from 2019 GSAA parcel
declarations (incomplete in places: unlabelled fields count as background, as in the paper).

Masks have 4 bands: extent, boundary, distance, field enumeration. We keep only the field
enumeration (instance ids; -10000 = no field -> 0, ids relabelled 1..K per tile) and generate
extent/boundary/distance on the fly with ``manyseg.instances.instance_targets`` so definitions
match the library; the published extent band is used as an integrity check.

Splits: official train 5,319 / val 1,140 / test 1,139. The official split is **random, not
spatially blocked** (neighbouring 4x4 km cells can fall in different splits), so tile names keep
their country prefix for country-held-out analyses.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .base import DatasetInfo, check_native_size

URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/AI4BOUNDARIES/orthophoto/orthos.zip"
MD5 = "c54bdab203a9732a5d928b14585f75c8"
EXPECTED = {"train": 5319, "val": 1140, "test": 1139}

INFO = DatasetInfo(
    name="ai4boundaries",
    num_classes=1,
    in_channels=3,
    class_names=("field",),
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    license="CC BY 4.0, (c) European Union (JRC)",
    source="http://data.europa.eu/89h/0e79ce5d-e4c8-4721-8773-59a4acf2c9c9",
    splits=("train", "val", "test"),
    test_splits=("test",),
    notes=("official split is random, not spatial", "labels from 2019 GSAA parcels"),
)
TASK = "instances"
TASK_KWARGS = {"boundary_width": 2.0}
LOADER_KWARGS = {"device_resident": False, "vflip": True, "scale": (0.5, 1.0)}
EVAL_BATCH_SIZE = 8
SIZE = 512  # tiles as published (prepare does not resize)


def _root(raw: Path) -> Path:
    for cand in (raw / "orthos", raw):
        if (cand / "images").exists():
            return cand
    raise FileNotFoundError(f"no images/ under {raw}; unzip orthos.zip there first")


def _read(args):
    import tifffile

    img_path, mask_path = args
    img = tifffile.imread(img_path)
    m = tifffile.imread(mask_path)
    ids = np.nan_to_num(m[..., 3], nan=-10000)
    ids = np.where(ids > 0, ids, 0).astype(np.int64)
    u, inv = np.unique(ids, return_inverse=True)
    seq = inv.reshape(ids.shape) + (0 if u[0] == 0 else 1)
    agree = float(((m[..., 0] > 0.5) == (seq > 0)).mean())
    return img[..., :3].astype(np.uint8), seq.astype(np.int32), agree


def prepare(raw: Path, cache: Path, size: int | None = None, workers: int = 4,
            expected: dict | None = EXPECTED) -> dict:
    """Cache the tiles at their published ``SIZE`` px; another ``size`` raises (before any
    download)."""
    check_native_size(size, SIZE, "ai4boundaries")
    size = SIZE
    raw, cache = Path(raw), Path(cache)
    try:
        root = _root(raw)
    except FileNotFoundError:
        import subprocess

        z = raw / "orthos.zip"
        if not z.exists():
            subprocess.run(["curl", "-sSfL", "-o", str(z), URL], check=True)
        subprocess.run(["unzip", "-q", "-o", str(z), "-d", str(raw / "orthos")], check=True)
        root = _root(raw)
    cache.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ("train", "val", "test"):
        imgs = sorted((root / "images" / split).glob("*_ortho_1m_512.tif"))
        names = [p.name.replace("_ortho_1m_512.tif", "") for p in imgs]
        masks = [root / "masks" / split / f"{n}_ortholabel_1m_512.tif" for n in names]
        missing = [str(m) for m in masks if not m.exists()]
        if missing:
            raise RuntimeError(f"integrity: {len(missing)} masks missing in {split}, "
                               f"e.g. {missing[0]}")
        if expected is not None and split in expected and len(imgs) != expected[split]:
            raise RuntimeError(f"integrity: {split} has {len(imgs)} tiles, "
                               f"expected {expected[split]}")
        n = len(imgs)
        out_i = np.lib.format.open_memmap(cache / f"{split}_images.npy", "w+", np.uint8,
                                          (n, size, size, 3))
        out_m = np.lib.format.open_memmap(cache / f"{split}_masks.npy", "w+", np.int32,
                                          (n, size, size))
        agree, n_fields = [], 0
        with ProcessPoolExecutor(workers) as ex:
            for j, (im, m, a) in enumerate(ex.map(_read, zip(imgs, masks), chunksize=16)):
                if im.shape[:2] != (size, size):
                    raise RuntimeError(f"integrity: {names[j]} is {im.shape}, expected {size}")
                out_i[j], out_m[j] = im, m
                agree.append(a)
                n_fields += int(m.max())
        out_i.flush()
        out_m.flush()
        (cache / f"{split}_names.json").write_text(json.dumps(names))
        summary[split] = {"n": n, "fields": n_fields,
                          "extent_agreement": float(np.mean(agree)) if agree else 1.0,
                          "countries": {c: sum(nm.startswith(c + "_") for nm in names)
                                        for c in sorted({nm.split("_")[0] for nm in names})}}
        if agree and min(agree) < 0.95:
            raise RuntimeError(f"integrity: published extent disagrees with field ids on "
                               f"{split} (min agreement {min(agree):.3f})")
    (cache / "meta.json").write_text(json.dumps({"summary": summary}, indent=1))
    return summary
