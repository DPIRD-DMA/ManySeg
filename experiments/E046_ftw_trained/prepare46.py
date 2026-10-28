"""E046: FTW Netherlands chips -> cache (README).

    uv run python prepare46.py   # writes $CACHE/{split}_img.npy (N,256,256,8 uint16: window_a B04 B03 B02 B08,
                                 # window_b ...), {split}_inst.npy (N,256,256 int32), {split}_poly.npz, stats.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "E030_whu_polygons"))
sys.path.insert(0, str(HERE.parent / "E045_field_oracle"))
from common import save_polygons  # noqa: E402
from e045 import chip_polygons  # noqa: E402

DATA = Path("$MANYSEG_BENCH_DATA/ftw/netherlands")
CACHE = Path("$MANYSEG_BENCH_DATA/cache/e046_ftw_nl")


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(DATA / "chips_netherlands.parquet")
    stats = None
    for split in ("train", "val", "test"):
        ids = sorted(d[d.split == split].aoi_id)
        ok = [a for a in ids if all((DATA / w / f"{a}.tif").exists() for w in ("window_a", "window_b", "instance"))]
        img = np.lib.format.open_memmap(CACHE / f"{split}_img.npy", "w+", np.uint16, (len(ok), 256, 256, 8))
        inst = np.lib.format.open_memmap(CACHE / f"{split}_inst.npy", "w+", np.int32, (len(ok), 256, 256))
        polys = []
        for i, a in enumerate(ok):
            x = np.concatenate([rasterio.open(DATA / w / f"{a}.tif").read() for w in ("window_a", "window_b")])
            img[i] = x.transpose(1, 2, 0)
            m = rasterio.open(DATA / "instance" / f"{a}.tif").read(1).astype(np.int64)
            u, inv = np.unique(m, return_inverse=True)
            inv = inv.reshape(m.shape) + (0 if u[0] == 0 else 1)  # background 0, fields 1..K
            inst[i] = np.where(m > 0, inv, 0).astype(np.int32)
            polys.append(chip_polygons(DATA / "instance" / f"{a}.tif"))
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
