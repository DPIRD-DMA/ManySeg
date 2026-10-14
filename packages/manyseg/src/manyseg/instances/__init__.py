"""Boundary + distance instance outputs: presets, targets, post-processing.

Losses and metrics live in ``manyseg-bench`` (training orchestration is not part of the model
library). Post-processing runs outside the exported graph.
"""

from .postprocess import (
    connected_components,
    grow_labels,
    instances_from_outputs,
    relabel_sequential,
)
from .presets import instance_heads
from .targets import edge_pixels, edt, instance_distance, instance_targets

__all__ = ["connected_components", "edge_pixels", "edt", "grow_labels", "instance_distance",
           "instance_heads", "instance_targets", "instances_from_outputs", "relabel_sequential"]
