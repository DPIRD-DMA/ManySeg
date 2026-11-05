"""Boundary + distance instance outputs: presets, targets, post-processing.

The loss terms are :func:`instance_losses`. Object metrics and the training loops live in
``manyseg-bench`` (training orchestration is not part of the model library). Post-processing runs
outside the exported graph.
"""

from .losses import instance_losses
from .postprocess import (
    anchor_labels,
    connected_components,
    grow_labels,
    instances_from_outputs,
    relabel_sequential,
)
from .presets import instance_heads
from .targets import edge_pixels, edt, instance_distance, instance_targets

__all__ = ["anchor_labels", "connected_components", "edge_pixels", "edt", "grow_labels",
           "instance_distance", "instance_heads", "instance_losses", "instance_targets",
           "instances_from_outputs",
           "relabel_sequential"]
