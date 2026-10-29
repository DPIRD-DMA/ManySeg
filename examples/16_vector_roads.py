"""Road networks as graphs (experimental; needs ``pip install 'manyseg[vectors]'``).

Road lines in a CRS (OSM-style ways) -> a road graph with junctions and overpasses -> training
targets -> the unified road heads -> a scene decoded block by block -> GeoJSON LineStrings (and
a GeoPackage with geopandas).

The model is the unified road preset (``road_heads``: surface, centreline, junction / end /
crossing heatmaps, two direction modes), decoded with ``graph_from_road_heads`` (R4; E044b:
same APLS and junction F1 as a mask skeleton, about 23 points more of the overpasses kept).
Here the targets of the whole scene stand in for a trained model's stitched predictions;
``vector_presets.py`` shows the training losses.
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
from manyseg import create_model

# A 512 x 512 px scene at 1 m in MGA zone 50 (EPSG:28350), rasterio order.
TRANSFORM = (1.0, 0.0, 391_000.0, 0.0, -1.0, 6_466_000.0)
CRS = "EPSG:28350"
SHAPE = (512, 512)
LOGIT_HEADS = ("mask", "line", "keypoints")  # sigmoid these; direction is a value


def ways():
    """OSM-style ways in metres with a width attribute. Ways that share a vertex meet at a
    junction; the motorway shares none with the street it passes over (a bridge)."""
    import shapely

    def way(*xy):  # positions in metres from the scene's top-left corner
        return shapely.LineString([(391_000 + x, 6_466_000 - y) for x, y in xy])

    streets = [way((0, 120), (130, 122), (300, 118), (512, 125)),
               way((300, 0), (300, 118)),
               way((300, 118), (310, 300), (305, 512)),
               way((130, 122), (200, 330), (260, 400)),
               way((60, 300), (200, 512))]
    motorway = [way((0, 500), (512, 230))]
    return streets + motorway, [6.0] * len(streets) + [12.0]


def main():
    try:
        import networkx  # noqa: F401
        import rasterio  # noqa: F401
        import shapely  # noqa: F401
        import skimage  # noqa: F401
    except ImportError:
        print("skipping: needs pip install 'manyseg[vectors]'")
        return
    from manyseg.vectors import (
        Scene,
        block_decoder,
        decode_scene,
        graph_from_road_heads,
        road_graph,
        road_heads,
        road_targets,
        to_pixels,
        write_records,
    )

    # 1. Ways -> pixel lines (shared vertices kept, so junctions survive) -> road graph with
    #    per-edge widths in pixels and the grade-separated crossings -> every head's target.
    lines_world, width_m = ways()
    lines, src = to_pixels(lines_world, TRANSFORM, SHAPE, crs=CRS, raster_crs=CRS,
                           return_index=True)
    graph, crossings = road_graph(lines, widths=np.asarray(width_m)[src] / TRANSFORM[0])
    t = road_targets(graph, SHAPE, crossings=crossings)
    print(f"road graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, "
          f"{len(crossings)} overpasses; targets:", {k: v.shape for k, v in t.items()})

    # 2. The model and what the decoder reads (probabilities for the logit heads).
    model = create_model(backbone="resnet18", pretrained=False, in_channels=3,
                         heads=road_heads()).eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 256, 256))
    fields = {k: (v.sigmoid() if k in LOGIT_HEADS else v)[0].numpy() for k, v in out.items()}
    print("model fields:", {k: v.shape for k, v in fields.items()})
    fields = {k: t[k] for k in ("mask", "keypoints", "direction")}  # stand-in: perfect outputs

    # 3. Block by block. The halo must cover dec.min_halo and the skeleton's reach: about the
    #    half-width of the widest road blob (more for blobs with holes, such as car parks);
    #    combine warns when a block's skeleton depended on its halo.
    scene = Scene(*SHAPE, transform=TRANSFORM, crs=CRS)
    dec = block_decoder("roads")  # R4 with E044's City-Scale tuning (1 m pixels)
    g = decode_scene(fields, dec, block=256, halo=64, scene=scene)
    whole = graph_from_road_heads(fields["mask"], fields["keypoints"], fields["direction"])
    same = np.array_equal(g.xy, whole.xy) and np.array_equal(g.edges, whole.edges)
    junctions = g.junctions()
    bridges = sum(np.hypot(*(junctions - c).T).min() > 15 for c in crossings)
    print(f"decoded: {len(g.edges)} segments, {len(junctions)} junctions, "
          f"{bridges} of {len(crossings)} overpasses kept (no junction there); "
          f"equal to the whole-scene decode: {same}")

    # 4. LineStrings in metres (one per chain between junctions and ends).
    out_dir = Path(tempfile.mkdtemp())
    write_records(g, out_dir / "roads.geojson", scene)
    try:
        import geopandas as gpd
    except ImportError:
        print("wrote roads.geojson (a GeoPackage needs geopandas)")
        return
    gdf = gpd.read_file(write_records(g, out_dir / "roads.gpkg", scene))
    true_len = sum(ln.length for ln in lines) * TRANSFORM[0]
    print(f"wrote roads.geojson, roads.gpkg ({gdf.crs}): {len(gdf)} lines, "
          f"{gdf.length.sum():.0f} m of road ({true_len:.0f} m in the input)")


if __name__ == "__main__":
    main()
