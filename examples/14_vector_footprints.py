"""Building footprints as polygons (experimental; needs ``pip install 'manyseg[vectors]'``).

Building outlines in a CRS -> training targets for one tile -> the model's heads -> a scene
decoded block by block -> GeoJSON (and a GeoPackage with geopandas).

The model is the corner-line preset (``corner_line_heads``: instance heads + a corner heatmap +
wall directions), decoded with corners and walls (B3, E038). Training it needs a dataset and a
GPU (losses: ``CornerLineTask`` in manyseg-bench, or the short versions in
``vector_presets.py``), so here the targets of the whole scene stand in for a trained model's
stitched predictions. Swap in your model's outputs: the decoding and writing are the same.
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
from manyseg import create_model

# A 512 x 512 px scene at 0.5 m in MGA zone 50 (EPSG:28350), as rasterio's dataset.transform
# gives it: (pixel width, 0, left, 0, -pixel height, top). GDAL order converts with from_gdal.
TRANSFORM = (0.5, 0.0, 391_000.0, 0.0, -0.5, 6_466_000.0)
CRS = "EPSG:28350"
SHAPE = (512, 512)
LOGIT_HEADS = ("extent", "boundary", "corner")  # sigmoid these; distance, direction are values


def buildings():
    """Building outlines in metres (a cadastre layer; a GeoDataFrame or GeoJSON works too)."""
    import shapely

    rng = np.random.default_rng(0)
    out = [shapely.Polygon([(391_050, 6_465_940), (391_080, 6_465_940), (391_080, 6_465_925),
                            (391_065, 6_465_925), (391_065, 6_465_905), (391_050, 6_465_905)])]
    while len(out) < 30:  # the L-shape above crosses the blocks' seams; so do some of these
        x, y = 391_000 + rng.uniform(4, 240), 6_466_000 - rng.uniform(4, 240)
        w, h = rng.uniform(7, 22, 2)
        b = shapely.affinity.rotate(shapely.box(x, y - h, x + w, y), rng.uniform(0, 90))
        if not any(b.buffer(2).intersects(o) for o in out):
            out.append(b)
    return out


def main():
    try:
        import networkx  # noqa: F401
        import rasterio  # noqa: F401
        import shapely
        import skimage  # noqa: F401
    except ImportError:
        print("skipping: needs pip install 'manyseg[vectors]'")
        return
    from manyseg.vectors import (
        Scene,
        block_decoder,
        corner_line_heads,
        decode_scene,
        footprint_targets,
        to_pixels,
        write_records,
    )

    # 1. Vector labels -> pixel geometry of the tile ((x, y), pixel corners at integers),
    #    reprojected when crs and raster_crs differ -> every head's target as numpy arrays.
    world = buildings()
    polys = to_pixels(world, TRANSFORM, SHAPE, crs=CRS, raster_crs=CRS)
    t = footprint_targets(polys, SHAPE)
    print("targets:", {k: v.shape for k, v in t.items()})

    # 2. The model and what the decoder reads: probabilities for the logit heads.
    model = create_model(backbone="resnet18", pretrained=False, in_channels=3,
                         heads=corner_line_heads()).eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 256, 256))
    fields = {k: (v.sigmoid() if k in LOGIT_HEADS else v)[0].numpy() for k, v in out.items()}
    print("model fields:", {k: v.shape for k, v in fields.items()})
    # untrained: decode the scene's targets instead, as a perfect model's stitched outputs
    fields = {k: t[k] for k in ("extent", "boundary", "distance", "corner", "direction")}

    # 3. Block by block (block 256 px + 64 px halo here; 4096 + 256 on real scenes), as
    #    ManyPred's BlockPostprocess runs it while predicting. Equal to a whole-scene decode.
    scene = Scene(*SHAPE, transform=TRANSFORM, crs=CRS)
    dec = block_decoder("footprints")  # B3 defaults; halo must be >= dec.min_halo
    records = decode_scene(fields, dec, block=256, halo=64, scene=scene)
    whole = decode_scene(fields, dec, block=max(SHAPE), scene=scene)  # one block
    same = all(shapely.equals_exact(a["geometry"], b["geometry"], 0)
               for a, b in zip(records, whole)) and len(records) == len(whole)
    print(f"{len(records)} polygons from {len(polys)} buildings; {records.stats}; "
          f"equal to the whole-scene decode: {same}")
    walls = sorted(len(r["geometry"].exterior.coords) - 1 for r in records)
    print("corners per polygon:", walls)  # straight walls: 4 for rectangles, 6 for the L

    # 4. Records -> world coordinates through scene.transform, CRS kept.
    out_dir = Path(tempfile.mkdtemp())
    path = write_records(records, out_dir / "buildings.geojson", scene)
    print("wrote", path.name, "in metres; properties:", [k for k in records[0] if k != "geometry"])
    try:
        import geopandas as gpd
    except ImportError:
        print("GeoPackage: needs geopandas")
        return
    gdf = gpd.read_file(write_records(records, out_dir / "buildings.gpkg", scene))
    iou = [max(p.intersection(b).area / p.union(b).area for b in world) for p in gdf.geometry]
    print(f"wrote buildings.gpkg ({gdf.crs}); IoU with the input layer: min {min(iou):.3f}")


if __name__ == "__main__":
    main()
