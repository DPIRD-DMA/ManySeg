"""Vector outputs: line graphs and polygons decoded from dense heads, and GeoJSON export.

Decoding runs outside the exported model, like :mod:`manyseg.instances`. Metrics (APLS, C-IoU,
PoLiS) live in ``manyseg-bench``. Needs the ``vectors`` extra (shapely, networkx, scipy,
scikit-image). See ``experiments/E028_vector_oracle`` for how these baselines were chosen.
"""

from .graph import VectorGraph, bridge_ends, graph_chains, graph_from_mask
from .polygons import polygons_from_labels, to_geojson

__all__ = ["VectorGraph", "bridge_ends", "graph_chains", "graph_from_mask", "polygons_from_labels",
           "to_geojson"]
