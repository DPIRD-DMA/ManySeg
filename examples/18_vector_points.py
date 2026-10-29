"""Tree points and counts over a scene (experimental; needs ``pip install 'manyseg[vectors]'``).

Tree locations in a CRS -> heatmap and density targets -> point + density heads -> a scene
decoded block by block -> GeoJSON points with scores (and a GeoPackage with geopandas).

``manyseg.points`` works in ``(y, x)`` with pixel centres at integers; ``manyseg.vectors`` and
GIS in ``(x, y)`` with pixel corners at integers. ``points_from_xy`` / ``points_to_xy`` are the
only conversions needed (``ScenePoints.xy`` and ``to_geojson`` do the second for you). Here the
targets of the whole scene stand in for a trained model's stitched predictions;
``point_counting.py`` trains the heads briefly.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from manyseg import create_model

TRANSFORM = (0.5, 0.0, 391_000.0, 0.0, -0.5, 6_466_000.0)  # 0.5 m, EPSG:28350, rasterio order
CRS = "EPSG:28350"
SHAPE = (400, 400)


def trees():
    """Tree positions in metres (a GeoDataFrame of points or GeoJSON works too)."""
    import shapely

    rng = np.random.default_rng(0)
    pts = []
    while len(pts) < 60:  # at least 4 m apart
        p = rng.uniform((391_002, 6_465_802), (391_198, 6_465_998))
        if all(np.hypot(*(p - q)) >= 4 for q in pts):
            pts.append(p)
    return [shapely.Point(p) for p in pts]


def main():
    try:
        import rasterio  # noqa: F401
        import shapely  # noqa: F401
    except ImportError as e:  # CI (MANYSEG_REQUIRE_TRAINERS=1) fails on "is not installed"
        print(f"skipping: {e.name} is not installed (pip install 'manyseg[vectors]')")
        return
    from manyseg.points import (
        DensityBlockDecoder,
        PointBlockDecoder,
        density_head,
        density_map,
        gaussian_heatmap,
        point_heads,
        points_from_xy,
    )
    from manyseg.vectors import Scene, decode_scene, to_pixels

    # 1. Points in a CRS -> (N, 2) pixel (x, y) inside the tile -> (y, x) targets.
    world = trees()
    xy = to_pixels(world, TRANSFORM, SHAPE, crs=CRS, raster_crs=CRS, kind="points")
    yx = points_from_xy(xy)
    heatmap = gaussian_heatmap([yx], SHAPE, sigma=2.0)[0].numpy()  # (1, H, W), peak 1
    density = density_map([yx], SHAPE, sigma=2.0)[0].numpy()  # sums to the count
    print(f"{len(xy)} trees; heatmap {heatmap.shape}, density sums to {density.sum():.2f}")

    # 2. The model: a regression heatmap (MSE) and a softplus density head; both are values.
    model = create_model(backbone="resnet18", pretrained=False, in_channels=3,
                         heads={**point_heads(), **density_head(activation="softplus")}).eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 128, 128))
    print("model fields:", {k: tuple(v[0].shape) for k, v in out.items()})
    fields = {"heatmap": heatmap, "density": density}  # stand-in: perfect outputs

    # 3. Block by block: peaks whose pixel lies in a block's core; halo >= dec.reach.
    #    (block_decoder("points", ...) / ("density", ...) build the same decoders.)
    scene = Scene(*SHAPE, transform=TRANSFORM, crs=CRS)
    dec = PointBlockDecoder(threshold=0.5, kernel=3)
    pts = decode_scene(fields, dec, block=128, halo=16, scene=scene)  # ScenePoints
    count = decode_scene(fields, DensityBlockDecoder(), block=128, halo=0, scene=scene)
    err = np.hypot(*(pts.xy[:, None] - xy[None]).transpose(2, 0, 1)).min(1)
    print(f"{len(pts)} peaks (max {err.max():.2f} px from a tree: peaks are pixels; add an "
          f"offset head for sub-pixel points); density count {count.total:.2f} over "
          f"{len(count.counts)} blocks")

    # 4. GeoJSON points in metres, with each peak's score.
    out_dir = Path(tempfile.mkdtemp())
    fc = pts.to_geojson()  # through scene.transform; it does not name the CRS, so add it
    fc["crs"] = {"type": "name", "properties": {"name": CRS}}  # as write_records does
    (out_dir / "trees.geojson").write_text(json.dumps(fc))
    print("first tree:", fc["features"][0]["geometry"]["coordinates"],
          fc["features"][0]["properties"])
    try:
        import geopandas as gpd
    except ImportError:
        print("wrote trees.geojson (a GeoPackage needs geopandas)")
        return
    gdf = gpd.GeoDataFrame.from_features(fc["features"], crs=CRS)
    gdf.to_file(out_dir / "trees.gpkg")
    print(f"wrote trees.geojson, trees.gpkg ({gdf.crs}, {len(gdf)} points)")


if __name__ == "__main__":
    main()
