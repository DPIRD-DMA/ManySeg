"""E046: FTW Netherlands chips -> cache (README).

    uv run python prepare46.py   # writes $CACHE/{split}_img.npy (N,256,256,8 uint16: window_a B04 B03 B02 B08,
                                 # window_b ...), {split}_inst.npy (N,256,256 int32), {split}_poly.npz, stats.json
"""

from __future__ import annotations

import os
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "E030_whu_polygons"))
from common import save_polygons  # noqa: E402
from rasterio.features import shapes  # noqa: E402
from scipy import ndimage  # noqa: E402
from shapely.geometry import Polygon, shape  # noqa: E402

MIN_AREA = 16
SIMPLIFY = 1.0

DATA = Path(os.environ.get("MANYSEG_BENCH_DATA", "data"), "ftw/netherlands")
CACHE = Path(os.environ.get("MANYSEG_BENCH_DATA", "data"), "cache/e046_ftw_nl")


def instances(m: np.ndarray):
    """One accepted set of fields for both targets (Codex E046 #2-4): every 4-connected piece of
    an FTW id is its own instance; pieces under 16 px are dropped; holes holding only background are
    filled; each instance's polygon is the exterior of its final pixels, Douglas-Peucker 1 px; an
    instance whose polygon is degenerate is removed from the mask too. Returns (ids, polygons) with
    polygon k-1 <-> id k."""
    out = np.zeros(m.shape, np.int32)
    k = 0
    for v in np.unique(m[m > 0]):
        lab, n = ndimage.label(m == v)
        for j in range(1, n + 1):
            c = lab == j
            if c.sum() < MIN_AREA:
                continue
            k += 1
            out[ndimage.binary_fill_holes(c) & ((m == 0) | c) & (out == 0)] = k
    polys, ids = [], np.zeros_like(out)
    for geom, v in sorted(shapes(out, mask=out > 0, connectivity=4), key=lambda g: g[1]):
        p = Polygon(shape(geom).exterior).simplify(SIMPLIFY)
        if not p.is_valid:
            p = p.buffer(0)
        if p.geom_type == "Polygon" and p.area >= MIN_AREA:
            polys.append(p)
            ids[out == int(v)] = len(polys)
    return ids, polys


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(DATA / "chips_netherlands.parquet")
    stats = None
    for split in ("train", "val", "test"):
        ids = sorted(d[d.split == split].aoi_id)
        ok = [a for a in ids if all((DATA / w / f"{a}.tif").exists() for w in ("window_a", "window_b", "instance"))]
        if len(ok) != len(ids):
            raise SystemExit(f"{split}: {len(ids) - len(ok)} of {len(ids)} chips incomplete; finish the download")
        img = np.lib.format.open_memmap(CACHE / f"{split}_img.npy", "w+", np.uint16, (len(ok), 256, 256, 8))
        inst = np.lib.format.open_memmap(CACHE / f"{split}_inst.npy", "w+", np.int32, (len(ok), 256, 256))
        polys = []
        for i, a in enumerate(ok):
            x = np.concatenate([rasterio.open(DATA / w / f"{a}.tif").read() for w in ("window_a", "window_b")])
            img[i] = x.transpose(1, 2, 0)
            ids_i, polys_i = instances(rasterio.open(DATA / "instance" / f"{a}.tif").read(1).astype(np.int64))
            inst[i] = ids_i
            polys.append(polys_i)
        img.flush(), inst.flush()
        save_polygons(CACHE / f"{split}_poly.npz", polys)
        (CACHE / f"{split}_ids.json").write_text(json.dumps(ok))
        if split == "train":
            s = np.asarray(img[:: max(1, len(ok) // 300)], np.float32).reshape(-1, 8)
            stats = {"mean": s.mean(0).tolist(), "std": s.std(0).tolist()}
            (CACHE / "stats.json").write_text(json.dumps(stats))
        print(split, len(ids), "chips,", len(ok), "complete", flush=True)


if __name__ == "__main__":
    main()
