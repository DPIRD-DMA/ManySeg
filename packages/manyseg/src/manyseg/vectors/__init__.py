"""Vector outputs: line graphs and polygons decoded from dense heads, and GeoJSON export.

Decoding runs outside the exported model, like :mod:`manyseg.instances`. Metrics (APLS, C-IoU,
PoLiS) live in ``manyseg-bench``. Needs the ``vectors`` extra (shapely, networkx, scipy,
scikit-image). See ``experiments/E028_vector_oracle`` for how these baselines were chosen.
"""

from .graph import VectorGraph, bridge_ends, graph_chains, graph_from_mask
from .graph_tensor import (
    encode_graph_tensor,
    graph_from_tensor,
    graph_tensor_heads,
    polygons_from_tensor,
    vertex_graph,
)
from .polygons import polygons_from_labels, to_geojson

__all__ = ["VectorGraph", "bridge_ends", "encode_graph_tensor", "graph_chains", "graph_from_mask",
           "graph_from_tensor", "graph_tensor_heads", "polygons_from_labels", "polygons_from_tensor",
           "to_geojson", "vertex_graph"]
