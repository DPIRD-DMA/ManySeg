"""E030 sanity tests: dihedral transforms keep polygons and rasters aligned; WKB round trip."""

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import Polygon, box

from common import dihedral_array, dihedral_polygon, load_polygons, save_polygons


def raster(polys, size=64):
    return rasterize([(p, i + 1) for i, p in enumerate(polys)], out_shape=(size, size),
                     transform=Affine.identity(), dtype="int32")


def test_dihedral_consistency():
    # vertices off the half-pixel grid, so no edge passes exactly through a pixel centre (where
    # rasterization tie-breaking is not rotation-invariant)
    polys = [box(3, 5, 20, 12), Polygon([(30.3, 30.1), (60.2, 34.3), (50.1, 60.3), (33.3, 50.2)])]
    base = raster(polys)
    for d in range(8):
        moved = raster([dihedral_polygon(p, d, 64) for p in polys])
        np.testing.assert_array_equal(moved, dihedral_array(base, d), err_msg=f"d={d}")
    assert len({dihedral_array(base, d).tobytes() for d in range(8)}) == 8


def test_wkb_round_trip(tmp_path):
    tiles = [[box(0, 0, 1, 1), box(2, 2, 4, 5)], [], [Polygon([(0, 0), (3, 0), (0, 3)])]]
    save_polygons(tmp_path / "p.npz", tiles)
    back = load_polygons(tmp_path / "p.npz")
    assert [len(t) for t in back] == [2, 0, 1]
    assert all(a.equals(b) for ta, tb in zip(tiles, back) for a, b in zip(ta, tb))
