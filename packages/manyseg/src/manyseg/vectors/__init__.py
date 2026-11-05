"""Vector outputs: line graphs and polygons decoded from dense heads, and GeoJSON export.

Decoding runs outside the exported model, like :mod:`manyseg.instances`. Metrics (APLS, C-IoU,
PoLiS) live in ``manyseg-bench``. Needs the ``vectors`` extra (shapely, networkx, rasterio,
scipy, scikit-image). Experimental in 0.1.0.

Recommended API: head presets (:func:`corner_line_heads`, :func:`road_heads`), target encoders
(:func:`to_pixels` -> :func:`footprint_targets` / :func:`field_targets` /
:func:`roof_face_targets` / :func:`road_graph` + :func:`road_targets`), decoding with
``decode_scene(fields, block_decoder(task), scene=Scene(...))`` (whole scene or block by
block), and output through :class:`SceneRecords` (``.write`` / ``.to_geojson``),
:func:`write_records` and :func:`records_to_geojson`. The rest (:func:`polygons_from_corners`,
:func:`polygons_from_labels`, :func:`graph_from_road_heads`, :func:`graph_from_mask`, ...) are
building blocks with generic defaults; roof assembly lives in :mod:`manyseg.vectors.roofs`.
"""

from .augment import dihedral, dihedral_geometry, dihedral_shape, dihedral_xy
from .blocks import (
    Block,
    BlockDecoder,
    InstanceBlockDecoder,
    RoofBlockDecoder,
    Scene,
    SceneRecords,
    block_decoder,
    block_grid,
    decode_scene,
    records_to_geojson,
    write_records,
)
from .corners import corner_targets, polygons_from_corners
from .georef import from_gdal, pixel_to_world, to_pixels, world_to_pixel
from .graph import VectorGraph, bridge_ends, graph_chains, graph_from_mask, graph_from_mask_blocks
from .polygons import planar_faces, polygons_from_labels, to_geojson
from .presets import corner_line_heads, road_heads
from .road_targets import road_targets
from .roads import RoadBlockDecoder, graph_from_road_heads
from .targets import (
    field_targets,
    footprint_targets,
    rasterize_instances,
    road_graph,
    roof_face_targets,
)

__all__ = [
    "Block",
    "BlockDecoder",
    "InstanceBlockDecoder",
    "RoadBlockDecoder",
    "RoofBlockDecoder",
    "Scene",
    "SceneRecords",
    "VectorGraph",
    "block_decoder",
    "block_grid",
    "bridge_ends",
    "corner_line_heads",
    "corner_targets",
    "decode_scene",
    "dihedral",
    "dihedral_geometry",
    "dihedral_shape",
    "dihedral_xy",
    "field_targets",
    "footprint_targets",
    "from_gdal",
    "graph_chains",
    "graph_from_mask",
    "graph_from_mask_blocks",
    "graph_from_road_heads",
    "pixel_to_world",
    "planar_faces",
    "polygons_from_corners",
    "polygons_from_labels",
    "rasterize_instances",
    "records_to_geojson",
    "road_graph",
    "road_heads",
    "road_targets",
    "roof_face_targets",
    "to_geojson",
    "to_pixels",
    "world_to_pixel",
    "write_records",
]
