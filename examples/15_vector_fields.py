"""Field boundaries as polygons (experimental; needs ``pip install 'manyseg[vectors]'``).

Field parcels in a CRS -> training targets -> the instance heads -> a scene decoded block by
block -> GeoJSON (and a GeoPackage with geopandas).

Fields use the instance heads only (``instance_heads``: extent, boundary, distance), decoded with
the outline of each instance (B0; E046 on Fields of The World, where corners and walls lost to
it). Here the targets of the whole scene stand in for a trained model's stitched predictions;
``instance_fields.py`` trains the heads briefly.
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
from manyseg import create_model

# A 480 x 480 px scene at 10 m (Sentinel-2) in UTM zone 50S (EPSG:32750), rasterio order.
TRANSFORM = (10.0, 0.0, 400_000.0, 0.0, -10.0, 6_500_000.0)
CRS = "EPSG:32750"
SHAPE = (480, 480)


def parcels():
    """Touching fields on a jittered 6 x 6 grid in metres (a cadastre or crop-map layer); a few
    have a track or fence line along one side, so they do not touch their neighbour there."""
    import shapely

    rng = np.random.default_rng(0)
    n, step = 6, 800.0  # 80 px per field
    gx = 400_000 + step * np.arange(n + 1)[None, :] + rng.uniform(-150, 150, (n + 1, n + 1))
    gy = 6_500_000 - step * np.arange(n + 1)[:, None] + rng.uniform(-150, 150, (n + 1, n + 1))
    gx[:, 0], gx[:, -1], gy[0], gy[-1] = 400_000, 404_800, 6_500_000, 6_495_200  # scene edge
    out = []
    for i in range(n):
        for j in range(n):
            ring = [(gx[i, j], gy[i, j]), (gx[i, j + 1], gy[i, j + 1]),
                    (gx[i + 1, j + 1], gy[i + 1, j + 1]), (gx[i + 1, j], gy[i + 1, j])]
            p = shapely.Polygon(ring)
            out.append(p.buffer(-15, join_style="mitre") if rng.random() < 0.3 else p)  # 30 m gap
    return out


def main():
    try:
        import rasterio  # noqa: F401
        import shapely
        import skimage  # noqa: F401
    except ImportError as e:  # CI (MANYSEG_REQUIRE_TRAINERS=1) fails on "is not installed"
        print(f"skipping: {e.name} is not installed (pip install 'manyseg[vectors]')")
        return
    from manyseg.instances import instance_heads
    from manyseg.vectors import (
        Scene,
        block_decoder,
        decode_scene,
        field_targets,
        to_pixels,
    )

    # 1. Parcels -> the tile's pixel geometry -> targets (each 4-connected piece is a field,
    #    pieces < 16 px dropped, holes holding only background filled: E046's single label set).
    world = parcels()
    polys = to_pixels(world, TRANSFORM, SHAPE, crs=CRS, raster_crs=CRS)
    t = field_targets(polys, SHAPE)
    print("targets:", {k: v.shape for k, v in t.items()})

    # 2. The model: the instance heads (13 Sentinel-2 bands). extent and boundary are logits.
    #    field_targets' distance is absolute (px; E052): train with
    #    InstanceTask(boundary_width=2, distance="absolute") from manyseg-bench.
    model = create_model(backbone="resnet18", pretrained=False, in_channels=13,
                         heads=instance_heads()).eval()
    with torch.no_grad():
        out = model(torch.rand(1, 13, 128, 128))
    fields = {k: (v if k == "distance" else v.sigmoid())[0].numpy() for k, v in out.items()}
    print("model fields:", {k: v.shape for k, v in fields.items()})
    fields = {k: t[k] for k in ("extent", "boundary", "distance")}  # stand-in: perfect outputs

    # 3. Block by block; every field here crosses a seam of the 3 x 3 grid of 160 px blocks.
    scene = Scene(*SHAPE, transform=TRANSFORM, crs=CRS)
    dec = block_decoder("fields")  # outline decoding, E052 thresholds, 0.5 px simplification
    records = decode_scene(fields, dec, block=160, halo=32, scene=scene)
    whole = decode_scene(fields, dec, scene=scene)  # default block 4096 px: one block
    same = len(records) == len(whole) and all(
        shapely.equals_exact(a["geometry"], b["geometry"], 0) for a, b in zip(records, whole))
    print(f"{len(records)} fields from {len(world)} parcels; {records.stats}; "
          f"equal to the whole-scene decode: {same}")
    # Even from perfect targets one extra small field appears here: a parcel's narrow tip, cut
    # off from the rest by the 2 px boundary band, gets a seed of its own. Raise min_size (px)
    # or filter by area if that matters for your data.

    out_dir = Path(tempfile.mkdtemp())
    records.write(out_dir / "fields.geojson")  # the scene's transform and CRS
    try:
        import geopandas as gpd
    except ImportError:
        print("wrote fields.geojson (a GeoPackage needs geopandas)")
        return
    gdf = gpd.read_file(records.write(out_dir / "fields.gpkg"))
    iou = [max(p.intersection(b).area / p.union(b).area for b in world) for p in gdf.geometry]
    print(f"wrote fields.geojson, fields.gpkg ({gdf.crs}, {gdf.area.sum() / 1e4:.0f} ha); "
          f"IoU with the input parcels: median {np.median(iou):.3f}, min {min(iou):.3f}")


if __name__ == "__main__":
    main()
