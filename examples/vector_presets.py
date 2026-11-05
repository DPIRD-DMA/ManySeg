"""Training the vector-output presets (runs on CPU with random weights).

Corner-line instances (footprints, fields, roof faces) and unified roads: head preset -> targets
encoded from vector data in world coordinates (building polygons and road lines in a projected
CRS, cut to one tile with ``to_pixels``) -> a random dihedral transform per step, with the
direction fields transformed too -> a few training steps -> vector decoding.

The losses are the trained models' (E036, E044): ``corner_line_losses`` and ``road_losses``, whose
terms are summed here. With manyseg-bench, a dataset whose cache holds these targets as a label
dict trains either preset through ``run_training`` (``TASK = "corner_lines"`` or ``"roads"``), which
augments with the same exact dihedral transforms (``GPULoader(dihedral=True,
directions=("direction",))``).
"""

import torch
from manyseg import create_model


def augment(d, image, targets, directions=("direction",)):
    """One dihedral transform ``d`` (0..7) of an image and its target dict."""
    from manyseg.vectors import dihedral

    return dihedral(image, d), {k: dihedral(v, d, direction=k in directions)
                                for k, v in targets.items()}


# The tile: 0.5 m pixels, top-left corner at (391000, 6466000) in MGA zone 50 (EPSG:28350), as
# rasterio's dataset.transform gives it (a GDAL geotransform converts with from_gdal).
TRANSFORM = (0.5, 0.0, 391_000.0, 0.0, -0.5, 6_466_000.0)


def corner_lines(size=64, steps=15):
    import shapely
    from manyseg.instances import instance_targets, instances_from_outputs
    from manyseg.vectors import (
        corner_line_heads,
        corner_line_losses,
        footprint_targets,
        polygons_from_corners,
        to_pixels,
    )

    x0, y0 = 391_000.0, 6_466_000.0  # building outlines in metres (e.g. a cadastre layer)
    buildings = [shapely.affinity.rotate(shapely.box(x0 + 4, y0 - 11, x0 + 13, y0 - 5), 20),
                 shapely.Polygon([(x0 + 18, y0 - 4), (x0 + 28, y0 - 4), (x0 + 28, y0 - 15),
                                  (x0 + 23, y0 - 15), (x0 + 23, y0 - 9), (x0 + 18, y0 - 9)]),
                 shapely.box(x0 + 6, y0 - 28, x0 + 15, y0 - 19),
                 shapely.box(x0 + 20, y0 - 27, x0 + 34, y0 - 20)]  # cut by the tile's edge
    # GeoJSON features, a GeoDataFrame (crs= / raster_crs= reproject) or shapely geometries
    polys = to_pixels(buildings, TRANSFORM, (size, size))
    t = footprint_targets(polys, (size, size))  # instances, corner, direction(_mask), extent...
    ids = torch.from_numpy(t["instances"]).long()
    image = torch.rand(3, size, size) * 0.2 + (ids > 0).float() * (0.5 + 0.1 * (ids % 3))
    targets = {k: torch.from_numpy(t[k]) for k in ("corner", "direction", "direction_mask")}
    targets["instances"] = ids  # what CornerLineTask reads (it derives extent etc. itself)
    # Perfect outputs round-trip: the instances and the target maps decode to straight walls
    exact = polygons_from_corners(t["instances"], t["corner"], t["direction"])
    print("corner lines: vertices per polygon from the targets:",
          [len(q.exterior.coords) - 1 for q in exact])

    model = create_model(backbone="resnet18", pretrained=False, heads=corner_line_heads())
    bias = float(model.heads.corner.conv.bias.detach())
    print(f"corner head bias {bias:.3f}: starts at p = 0.1 (CenterNet prior)")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(0)
    for step in range(steps):
        batch = [augment(int(torch.randint(8, (1,), generator=g)), image, targets)
                 for _ in range(2)]
        x = torch.stack([b[0] for b in batch])
        t = {k: torch.stack([b[1][k] for b in batch]) for k in targets}
        it = instance_targets(t["instances"], boundary_width=2)
        loss = sum(corner_line_losses(model(x), {**t, **it}).values())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 5 == 0:
            print(f"  step {step} loss {loss.item():.2f}")
    model.eval()
    with torch.no_grad():
        out = model(image[None])
    labels = instances_from_outputs(out["extent"], out["boundary"], out["distance"],
                                    logits=True)[0]
    pred = polygons_from_corners(labels, out["corner"][0, 0].sigmoid(), out["direction"][0])
    print(f"  {steps} steps (not converged): {len(pred)} polygons, true {len(polys)}")


def roads(size=64, steps=15):
    import numpy as np
    import shapely
    from manyseg.vectors import (
        graph_from_road_heads,
        road_graph,
        road_heads,
        road_losses,
        road_targets,
        to_pixels,
    )

    def world(*xy):  # pixel positions of this toy network -> metres (EPSG:28350)
        return shapely.LineString([(391_000 + 0.5 * x, 6_466_000 - 0.5 * y) for x, y in xy])

    # OSM-style ways: the junction is a shared vertex; the overpass shares none
    ways = [world((0, 30), (30, 31)), world((30, 31), (64, 26)), world((30, 31), (31, 64)),
            world((30, 31), (40, 8)), world((50, 0), (58, 64))]
    lines, src = to_pixels(ways, TRANSFORM, (size, size), return_index=True)
    widths = np.array([3.0, 3.0, 3.0, 3.0, 4.0])[src] / 0.5  # metres -> pixels
    g, crossing = road_graph(lines, widths=widths)  # junctions + grade-separated crossings
    rt = road_targets(g, (size, size), crossings=crossing)  # mask, line, keypoints, direction
    targets = {k: torch.from_numpy(v) for k, v in rt.items()}
    image = torch.rand(3, size, size) * 0.2 + targets["mask"] * 0.6
    exact = graph_from_road_heads(rt["mask"], rt["keypoints"], rt["direction"], min_length=5)
    at_crossing = np.hypot(*(exact.xy - crossing[0]).T) < 6
    print(f"roads: graph decoded from the targets: {len(exact.edges)} edges, "
          f"{int((exact.degree[at_crossing] >= 3).sum())} junctions at the overpass")

    model = create_model(backbone="resnet18", pretrained=False, heads=road_heads())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(0)
    for step in range(steps):
        batch = [augment(int(torch.randint(8, (1,), generator=gen)), image, targets)
                 for _ in range(2)]
        x = torch.stack([b[0] for b in batch])
        t = {k: torch.stack([b[1][k] for b in batch]) for k in targets}
        loss = sum(road_losses(model(x), t).values())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 5 == 0:
            print(f"  step {step} loss {loss.item():.2f}")
    model.eval()
    with torch.no_grad():
        out = model(image[None])
    iou = ((out["mask"][0, 0] > 0) & (targets["mask"] > 0.5)).sum() / (
        (out["mask"][0, 0] > 0) | (targets["mask"] > 0.5)).sum().clamp(min=1)
    print(f"  {steps} steps (not converged): road mask IoU {float(iou):.2f}")


def main():
    try:
        import networkx  # noqa: F401
        import rasterio  # noqa: F401
        import shapely  # noqa: F401
    except ImportError:
        print("skipping: needs pip install 'manyseg[vectors]'")
        return
    torch.manual_seed(0)
    corner_lines()
    roads()


if __name__ == "__main__":
    main()
