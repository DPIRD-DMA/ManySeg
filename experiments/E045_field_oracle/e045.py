"""E045: Fields of The World (FTW) field polygons as E028-style oracle tiles (README).

Each 1024 px tile is a 3 x 3 mosaic of 256 px FTW chips (10 m) at offsets 0 / 384 / 768 px, so
128 px empty gutters separate chips (no fake contacts across chip edges). Ground truth = each
field instance's traced outline, holes filled, simplified at 1 px (a vector-like outline, not the
raster staircase: the E038 lesson). Tiles carry gsd = 0.5 so E028's occlusion rate (per 100 m of
outline, disks of 4-12 px) gives the same disks per pixel of outline as the building oracle
(Codex E045 #1); metrics are reported in px.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import Polygon, shape

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "E028_vector_oracle"))
import e028 as E  # noqa: E402

DATA = Path(os.environ.get("E045_DATA", "ftw"))
OFFS = (0, 384, 768)
SIMPLIFY = 1.0
MIN_AREA = 16.0
CORRUPT_GSD = 0.5


def chip_polygons(path: Path) -> list[Polygon]:
    a = rasterio.open(path).read(1).astype(np.int64)
    out = []
    for geom, v in shapes(a.astype(np.int32), mask=a > 0, connectivity=4):
        p = Polygon(shape(geom).exterior).simplify(SIMPLIFY)
        if not p.is_valid:
            p = p.buffer(0)
        if p.geom_type == "Polygon" and p.area >= MIN_AREA:
            out.append(p)
    return out


def field_tiles(country: str, split: str, min_fields: int = 5) -> list[E.Tile]:
    ids = pd.read_parquet(DATA / country / f"chips_{country}.parquet")
    ids = sorted(ids[ids.split == split].aoi_id)
    tiles = []
    for k in range(0, len(ids) - 8, 9):
        polys = []
        for (oy, ox), aoi in zip([(y, x) for y in OFFS for x in OFFS], ids[k:k + 9]):
            for p in chip_polygons(DATA / country / "instance" / f"{aoi}.tif"):
                polys.append(Polygon(np.asarray(p.exterior.coords) + [ox, oy]))
        if len(polys) >= min_fields:
            tiles.append(E.Tile(f"{country}_{split}", "buildings", len(tiles), 0.0, 0.0, CORRUPT_GSD, polygons=polys))
    return tiles


def chip_boxes():
    from shapely.geometry import box

    return [box(x, y, x + 256, y + 256).exterior for y in OFFS for x in OFFS]


_EDGES = None


def boundary_counts(gt, pred, tol: float = 1.0, step: float = 0.5):
    """E038's boundary F counts; points within 1 px of a chip edge (artificial cuts) excluded."""
    import shapely

    global _EDGES
    if _EDGES is None:
        _EDGES = shapely.union_all(chip_boxes())

    def pts(polys):
        out = [shapely.line_interpolate_point(p.exterior, np.arange(0, p.exterior.length, step)) for p in polys]
        if not out:
            return np.array([])
        a = np.concatenate(out)
        return a[shapely.distance(a, _EDGES) > 1.0]

    pp, pg = pts(pred), pts(gt)
    bg = shapely.union_all([p.exterior for p in gt]) if gt else None
    bp = shapely.union_all([p.exterior for p in pred]) if pred else None
    tp_p = int((shapely.distance(pp, bg) <= tol).sum()) if len(pp) and bg is not None else 0
    tp_g = int((shapely.distance(pg, bp) <= tol).sum()) if len(pg) and bp is not None else 0
    return tp_p, len(pp), tp_g, len(pg)


def irregular(p: Polygon) -> bool:
    """Secondary subset: outlines far from their minimum rotated rectangle (curved / ragged)."""
    return p.area / p.minimum_rotated_rectangle.area < 0.8
