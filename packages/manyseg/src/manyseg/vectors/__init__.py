"""Vector outputs: line graphs and polygons decoded from dense heads, and GeoJSON export.

Decoding runs outside the exported model, like :mod:`manyseg.instances`. Metrics (APLS, C-IoU,
PoLiS) live in ``manyseg-bench``. Needs the ``vectors`` extra (shapely, networkx, scipy,
scikit-image). See ``experiments/E028_vector_oracle`` for how these baselines were chosen.
"""

from .corners import corner_targets, polygons_from_corners
from .graph import VectorGraph, bridge_ends, graph_chains, graph_from_mask
from .graph_tensor import (
    encode_graph_tensor,
    graph_from_tensor,
    graph_tensor_heads,
    offset_field,
    offset_field_from_mask,
    polygons_from_tensor,
    snap_points,
    vertex_graph,
)
from .polygons import planar_faces, polygons_from_labels, to_geojson
from .roads import graph_from_road_heads

__all__ = [
    "VectorGraph",
    "bridge_ends",
    "corner_targets",
    "encode_graph_tensor",
    "graph_chains",
    "graph_from_mask",
    "graph_from_road_heads",
    "graph_from_tensor",
    "graph_tensor_heads",
    "offset_field",
    "offset_field_from_mask",
    "planar_faces",
    "polygons_from_corners",
    "polygons_from_labels",
    "polygons_from_tensor",
    "snap_points",
    "to_geojson",
    "vertex_graph",
]

# --- training presets (heads, road targets, dihedral augmentation) ---
from .augment import dihedral, dihedral_geometry, dihedral_xy  # noqa: E402
from .presets import corner_line_heads, road_heads  # noqa: E402
from .road_targets import road_targets  # noqa: E402

__all__ += ["corner_line_heads", "dihedral", "dihedral_geometry", "dihedral_xy", "road_heads",
            "road_targets"]
# --- end training presets ---

# --- training-target encoders (vector data in a CRS -> tile pixels -> every head's target) ---
from .georef import from_gdal, pixel_to_world, to_pixels, world_to_pixel  # noqa: E402
from .targets import (  # noqa: E402
    field_targets,
    footprint_targets,
    rasterize_instances,
    road_graph,
    roof_face_targets,
)

__all__ += ["field_targets", "footprint_targets", "from_gdal", "pixel_to_world",
            "rasterize_instances", "road_graph", "roof_face_targets", "to_pixels",
            "world_to_pixel"]
# --- end training-target encoders ---

# --- block decoders for large scenes (vec/block-api) ---
from .blocks import (  # noqa: E402
    Block,
    BlockDecoder,
    InstanceBlockDecoder,
    Records,
    RoofBlockDecoder,
    Scene,
    block_decoder,
    block_grid,
    decode_scene,
    records_to_geojson,
    write_records,
)
from .roofs import face_plane, roof_faces, roof_mesh, roof_planes, roof_records  # noqa: E402

__all__ += [
    "Block", "BlockDecoder", "InstanceBlockDecoder", "Records", "RoofBlockDecoder", "Scene",
    "block_decoder", "block_grid", "decode_scene", "face_plane", "records_to_geojson",
    "roof_faces", "roof_mesh", "roof_planes", "roof_records", "write_records",
]
# --- end block decoders ---

# --- block-wise road graphs (vec/road-blocks) ---
from .roads import RoadBlockDecoder  # noqa: E402

__all__ += ["RoadBlockDecoder"]
# --- end block-wise road graphs ---
