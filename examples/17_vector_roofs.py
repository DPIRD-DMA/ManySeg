"""LoD2 roof faces with heights (experimental; needs ``pip install 'manyseg[vectors]'``).

Roof faces in a CRS -> training targets -> the corner-line heads on RGB + nDSM -> a scene
decoded block by block into planar faces with a fitted plane each and a 3-D mesh -> GeoJSON
(and a GeoPackage with geopandas).

Roofs need a height input: a normalised DSM (nDSM, metres above ground, e.g. from LiDAR or
photogrammetry) is a 4th input channel next to RGB, and each decoded face's plane is fitted to
it (E047b; RGB alone was not viable, E049). Faces are decoded with corners and walls, joined
into a planar partition (``planar_faces``, E048b) and lifted onto their planes. Here the
targets of the whole scene stand in for a trained model's stitched predictions.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from manyseg import create_model

# A 384 x 384 px scene at 0.25 m in the Dutch RD grid (EPSG:28992), rasterio order.
PX = 0.25
TRANSFORM = (PX, 0.0, 136_000.0, 0.0, -PX, 456_000.0)
CRS = "EPSG:28992"
SHAPE = (384, 384)
LOGIT_HEADS = ("extent", "boundary", "corner")


def houses():
    """Roof faces in metres (as in a 3D BAG / CityGML LoD2 layer) and, to make an nDSM, each
    face's plane z = a X + b Y + c (metres above ground, X / Y in metres)."""
    import shapely

    faces = []
    x0, y0 = 136_000.0, 456_000.0

    def gable(left, top, w, d, eave=3.0, slope=0.6):  # ridge along x, metres from the corner
        xa, xb, ya, ym, yb = x0 + left, x0 + left + w, y0 - top, y0 - top - d / 2, y0 - top - d
        ridge = eave + slope * d / 2
        faces.append((shapely.box(xa, ym, xb, ya), (0.0, -slope, ridge + slope * ym)))  # north
        faces.append((shapely.box(xa, yb, xb, ym), (0.0, slope, ridge - slope * ym)))  # south

    def hip(left, top, w, d, eave=3.5, slope=0.6):
        xa, xb, ya, yb = x0 + left, x0 + left + w, y0 - top, y0 - top - d
        h = d / 2
        a, b, c, e = (xa, ya), (xb, ya), (xb, yb), (xa, yb)
        r1, r2 = (xa + h, ya - h), (xb - h, ya - h)
        faces.append((shapely.Polygon([a, b, r2, r1]), (0.0, -slope, eave + slope * ya)))
        faces.append((shapely.Polygon([r1, r2, c, e]), (0.0, slope, eave - slope * yb)))
        faces.append((shapely.Polygon([a, r1, e]), (slope, 0.0, eave - slope * xa)))
        faces.append((shapely.Polygon([b, c, r2]), (-slope, 0.0, eave + slope * xb)))

    gable(6, 6, 16, 10)
    gable(28, 40, 14, 9)  # across the blocks' seam at 48 m
    hip(52, 52, 20, 12)
    hip(8, 66, 18, 10)
    faces.append((shapely.box(x0 + 60, y0 - 30, x0 + 80, y0 - 14), (0.0, 0.0, 9.0)))  # flat
    return faces


def ndsm(faces, transform, shape):
    """Heights above ground on the pixel grid (pixel centres), 0 off the roofs."""
    import shapely

    a, _, c, _, e, f = transform
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]] + 0.5
    X, Y = c + a * xx, f + e * yy
    z = np.zeros(shape, np.float32)
    for face, (pa, pb, pc) in faces:
        inside = shapely.contains_xy(face, X, Y)
        z[inside] = (pa * X + pb * Y + pc)[inside]
    return z


def main():
    try:
        import networkx  # noqa: F401
        import rasterio  # noqa: F401
        import shapely
        import skimage  # noqa: F401
    except ImportError as e:  # CI (MANYSEG_REQUIRE_TRAINERS=1) fails on "is not installed"
        print(f"skipping: {e.name} is not installed (pip install 'manyseg[vectors]')")
        return
    from manyseg.vectors import (
        Scene,
        block_decoder,
        corner_line_heads,
        decode_scene,
        roof_face_targets,
        to_pixels,
        write_records,
    )

    # 1. Faces -> pixel geometry -> targets (faces sharing an edge stay separate instances).
    truth = houses()
    faces = to_pixels([f for f, _ in truth], TRANSFORM, SHAPE, crs=CRS, raster_crs=CRS)
    t = roof_face_targets(faces, SHAPE)
    height = ndsm(truth, TRANSFORM, SHAPE)  # in practice: DSM - DTM, resampled to the image grid
    print(f"{len(faces)} faces; targets:", sorted(t), f"nDSM up to {height.max():.1f} m")

    # 2. The model sees RGB + nDSM (4 channels); the decoder reads its heads plus the nDSM.
    model = create_model(backbone="resnet18", pretrained=False, in_channels=4,
                         heads=corner_line_heads()).eval()
    x = torch.cat([torch.rand(1, 3, 128, 128), torch.from_numpy(height[None, None, :128, :128])], 1)
    with torch.no_grad():
        out = model(x)
    fields = {k: (v.sigmoid() if k in LOGIT_HEADS else v)[0].numpy() for k, v in out.items()}
    print("model fields:", {k: v.shape for k, v in fields.items()}, "+ height")
    fields = {k: t[k] for k in ("extent", "boundary", "distance", "corner", "direction")}
    fields["height"] = height  # the input nDSM, not a prediction

    # 3. Block by block: faces, planar join, a robust plane per face, a J2 mesh.
    scene = Scene(*SHAPE, transform=TRANSFORM, crs=CRS)
    dec = block_decoder("roofs")  # halo >= dec.min_halo
    records = decode_scene(fields, dec, block=192, halo=64, scene=scene)
    whole = decode_scene(fields, dec, block=max(SHAPE), scene=scene)
    same = len(records) == len(whole) and all(
        shapely.equals_exact(a["geometry"], b["geometry"], 0) and a["plane"] == b["plane"]
        for a, b in zip(records, whole))
    print(f"{len(records)} faces decoded; {records.stats}; equal to the whole-scene decode: {same}")
    # planes are in scene pixels: z = a * col + b * row + c (metres); slope in degrees:
    a, b, c = records[0]["plane"]
    print(f"first face: {np.degrees(np.arctan(np.hypot(a, b) / PX)):.0f} deg roof pitch")

    # 4. Faces as 2-D polygons (plane_a / _b / _c attributes) and as a 3-D mesh (z in metres).
    out_dir = Path(tempfile.mkdtemp())
    write_records(records, out_dir / "roof_faces.geojson", scene)
    path = write_records(records, out_dir / "roof_mesh.geojson", scene, mesh=True)
    ring = json.loads(path.read_text())["features"][0]["geometry"]["coordinates"][0]
    first = [tuple(round(v, 2) for v in p) for p in ring[:2]]
    print("3-D ring of the first face (X, Y, z):", first, "...")
    try:
        import geopandas as gpd
    except ImportError:
        print("wrote roof_faces.geojson, roof_mesh.geojson (a GeoPackage needs geopandas)")
        return
    gdf = gpd.read_file(write_records(records, out_dir / "roofs.gpkg", scene, mesh=True))
    print(f"wrote roofs.gpkg ({gdf.crs}, has z: {bool(gdf.has_z.all())}), {len(gdf)} faces")


if __name__ == "__main__":
    main()
