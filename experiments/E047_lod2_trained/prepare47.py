"""E047 data (README): 3DBAG roof faces + PDOK 2023 ortho (25 cm) + AHN4 nDSM, 512 px tiles.

    uv run python prepare47.py fetch3dbag   # new train / val areas into $RAW (test areas exist from E031)
    uv run python prepare47.py build        # tiles, inputs and targets into $CACHE
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import pickle
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.features import rasterize
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "E031_lod2_oracle"))
import e031 as E  # noqa: E402
import fetch_3dbag as F  # noqa: E402

RAW = Path("$MANYSEG_BENCH_DATA/raw/3dbag")
WEB = Path("$MANYSEG_BENCH_DATA/raw/e047_web")  # cached WMS / WCS responses
CACHE = Path("$MANYSEG_BENCH_DATA/cache/e047_lod2")
UA = {"User-Agent": "ManySeg-E047/0.1 (research)"}
NEW_AREAS = {  # RD New boxes; first N 3DBAG tiles of each
    "amersfoort": ((157000, 465000, 159000, 467000), 8), "zwolle": ((202000, 501000, 204000, 503000), 8),
    "eindhoven": ((160000, 383000, 162000, 385000), 8), "groningen": ((233000, 581000, 235000, 583000), 8),
    "haarlem": ((103000, 487000, 105000, 489000), 6)}
SPLITS = {"train": [("amersfoort", 300), ("zwolle", 300), ("eindhoven", 300), ("groningen", 300)],
          "val": [("haarlem", 150)], "test": [("leidscherijn", 40), ("staphorst", 40)]}
WMS = ("https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0?service=WMS&version=1.3.0&request=GetMap"
       "&layers=2023_ortho25&styles=&crs=EPSG:28992&bbox={x0},{y0},{x1},{y1}&width=512&height=512&format=image/jpeg")
WCS = ("https://service.pdok.nl/rws/ahn/wcs/v1_0?service=WCS&version=2.0.1&request=GetCoverage"
       "&coverageId={cov}&format=image/tiff&subset=x({x0},{x1})&subset=y({y0},{y1})")


def fetch3dbag():
    for area, (bbox, n) in NEW_AREAS.items():
        d = RAW / area
        d.mkdir(parents=True, exist_ok=True)
        for p in F.tiles(bbox)[:n]:
            path = d / (p["tile_id"].replace("/", "-") + ".city.json")
            if path.exists():
                continue
            gzp = path.with_suffix(".gz.part")
            for attempt in range(6):  # curl: urllib hangs on this object store's TLS handshake
                r = subprocess.run(["curl", "-s", "-f", "--max-time", "600", "-o", str(gzp), p["cj_download"]])
                if r.returncode == 0 and hashlib.sha256(gzp.read_bytes()).hexdigest() == p["cj_sha256"]:
                    path.write_bytes(gzip.decompress(gzp.read_bytes()))
                    gzp.unlink()
                    break
                print(area, p["tile_id"], "retry", r.returncode, flush=True)
                time.sleep(5 * (attempt + 1))
            print(area, p["tile_id"], path.exists(), flush=True)


def _get(url: str, path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                b = r.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b)
            return b
        except OSError:
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"failed: {url}")


def inputs(t) -> tuple[np.ndarray, np.ndarray]:
    """(512, 512, 3) uint8 RGB and (512, 512) float32 nDSM (m) for one tile window."""
    s = E.TILE * E.GSD
    b = dict(x0=t.x0, y0=t.y1 - s, x1=t.x0 + s, y1=t.y1)
    key = f"{t.area}/{int(t.x0)}_{int(t.y1)}"
    rgb = np.asarray(Image.open(io.BytesIO(_get(WMS.format(**b), WEB / f"{key}_rgb.jpg"))).convert("RGB"))
    rasters = {}
    for cov in ("dsm_05m", "dtm_05m"):
        with rasterio.MemoryFile(_get(WCS.format(cov=cov, **b), WEB / f"{key}_{cov}.tif")) as m, m.open() as r:
            rasters[cov] = r.read(1).astype(np.float32)
    dsm, dtm = rasters["dsm_05m"], rasters["dtm_05m"]
    bad = ~np.isfinite(dtm) | (np.abs(dtm) > 1e5)
    if bad.all():  # no terrain in the window: fill from a padded DTM window (Codex E047 #3)
        pad = 96.0
        pb = dict(x0=b["x0"] - pad, y0=b["y0"] - pad, x1=b["x1"] + pad, y1=b["y1"] + pad)
        with rasterio.MemoryFile(_get(WCS.format(cov="dtm_05m", **pb), WEB / f"{key}_dtm_pad.tif")) as m, m.open() as r:
            big = r.read(1).astype(np.float32)
        bb = ~np.isfinite(big) | (np.abs(big) > 1e5)
        if bb.all():
            raise RuntimeError(f"{key}: no valid DTM within {pad} m")
        big = big[tuple(ndimage.distance_transform_edt(bb, return_distances=False, return_indices=True))]
        k = int(round(pad / 0.5))
        dtm = big[k:k + dtm.shape[0], k:k + dtm.shape[1]]
    elif bad.any():
        dtm = dtm[tuple(ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True))]
    nd = np.where(~np.isfinite(dsm) | (np.abs(dsm) > 1e5), 0, dsm - dtm)
    nd = np.clip(ndimage.zoom(nd, 2, order=1, mode="nearest", grid_mode=True), 0, 60).astype(np.float32)
    return rgb, nd


def build():
    CACHE.mkdir(parents=True, exist_ok=True)
    for split, areas in SPLITS.items():
        tiles = []
        for area, n in areas:
            ts = E.make_tiles(area, E.load_faces(RAW / area), max_tiles=n)
            for t in ts:
                t.height = t.mask = None  # recomputed from faces / planes when needed
            tiles += ts
            print(split, area, len(ts), "tiles", flush=True)
        img = np.lib.format.open_memmap(CACHE / f"{split}_img.npy", "w+", np.uint8, (len(tiles), 512, 512, 3))
        nd = np.lib.format.open_memmap(CACHE / f"{split}_ndsm.npy", "w+", np.float16, (len(tiles), 512, 512))
        inst = np.lib.format.open_memmap(CACHE / f"{split}_inst.npy", "w+", np.int32, (len(tiles), 512, 512))
        with ThreadPoolExecutor(8) as ex:
            for i, (rgb, h) in enumerate(ex.map(inputs, tiles)):
                img[i], nd[i] = rgb, h
                f = tiles[i].faces
                inst[i] = rasterize([(p, k + 1) for k, p in enumerate(f)], out_shape=(512, 512),
                                    transform=E.IDENT, dtype="int32") if f else 0
                if i % 100 == 0:
                    print(split, i, "/", len(tiles), flush=True)
        img.flush(), nd.flush(), inst.flush()
        (CACHE / f"{split}_tiles.pkl").write_bytes(pickle.dumps(tiles))
        (CACHE / f"{split}_meta.json").write_text(json.dumps([[t.area, t.index, t.x0, t.y1] for t in tiles]))


def repair():
    """Recompute nDSM for tiles whose cached DTM was entirely invalid (built before the fix)."""
    for split in SPLITS:
        tiles = pickle.loads((CACHE / f"{split}_tiles.pkl").read_bytes())
        nd = np.load(CACHE / f"{split}_ndsm.npy", mmap_mode="r+")
        n = 0
        for i, t in enumerate(tiles):
            p = WEB / f"{t.area}/{int(t.x0)}_{int(t.y1)}_dtm_05m.tif"
            with rasterio.open(p) as r:
                d = r.read(1)
            if (~np.isfinite(d) | (np.abs(d) > 1e5)).all():
                nd[i] = inputs(t)[1]
                n += 1
        nd.flush()
        print(split, "repaired", n, "of", len(tiles), flush=True)


if __name__ == "__main__":
    {"fetch3dbag": fetch3dbag, "build": build, "repair": repair}[sys.argv[1]]()
