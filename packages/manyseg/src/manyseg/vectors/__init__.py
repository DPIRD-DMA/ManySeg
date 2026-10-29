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

# --- block-wise road graphs (vec/road-blocks) ---
from .roads import RoadBlockDecoder  # noqa: E402

__all__ += ["RoadBlockDecoder"]
# --- end block-wise road graphs ---
