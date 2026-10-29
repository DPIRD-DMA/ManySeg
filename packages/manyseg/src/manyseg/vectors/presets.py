"""Head presets for the trained vector-output models (corner-line instances, unified roads).

Both presets use the built-in head types only. The heatmap heads trained with the CenterNet focal
loss (``corner``, ``keypoints``) start at the prior ``p = 0.1`` (``bias_init = logit(0.1)``), so
the loss is not swamped by negatives at initialisation (E036: about 200 at ``p = 0.5``). Head
names are those the vector decoders read (``polygons_from_corners``, ``graph_from_road_heads``).
Losses and metrics live in ``manyseg-bench`` (``CornerLineTask``, ``RoadGraphTask``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

#: Bias of a focal-loss heatmap head: ``logit(0.1)`` (CenterNet; also ``point_heads``).
HEATMAP_PRIOR = math.log(0.1 / 0.9)


def _attach(heads: dict, decoders) -> dict:
    if decoders is None:
        return heads
    if isinstance(decoders, str):
        decoders = {k: ("unet" if decoders == "separate" else decoders) for k in heads}
    unknown = set(decoders) - set(heads)
    if unknown:
        raise ValueError(f"unknown heads in decoders: {sorted(unknown)}")
    for k, d in decoders.items():
        heads[k]["decoder"] = d
    return heads


def corner_line_heads(decoders: None | str | Mapping[str, Any] = None) -> dict[str, dict]:
    """Head configs for ``create_model(heads=...)`` of the corner-line instance model (B3:
    footprints, fields, roof faces; E036, E038, E047): the instance heads of
    :func:`manyseg.instances.instance_heads` (``extent``, ``boundary``, ``distance``) plus
    ``corner`` (1 channel, logits of a Gaussian corner heatmap, bias ``logit(0.1)``) and
    ``direction`` (2 channels, regression of the wall direction as ``(cos 2t, sin 2t)``).

    Targets: :func:`manyseg.instances.instance_targets` and
    :func:`manyseg.vectors.corner_targets`; decoding: instances, then
    :func:`manyseg.vectors.polygons_from_corners`. ``decoders`` as in ``instance_heads``.
    """
    from ..instances import instance_heads

    heads = {**instance_heads(),
             "corner": {"type": "segmentation", "out_channels": 1, "bias_init": HEATMAP_PRIOR},
             "direction": {"type": "regression", "out_channels": 2}}
    return _attach(heads, decoders)


def road_heads(decoders: None | str | Mapping[str, Any] = None) -> dict[str, dict]:
    """Head configs for the unified road model (R4; E039, E044): ``mask`` (1, road-surface
    logits), ``line`` (1, logits of a 3 px centreline band), ``keypoints`` (3, logits of the
    junction / end / crossing heatmaps, bias ``logit(0.1)``) and ``direction`` (4, regression
    of two ``(cos 2t, sin 2t)`` modes; the second is set where roads cross).

    Targets: :func:`manyseg.vectors.road_targets`; decoding:
    :func:`manyseg.vectors.graph_from_road_heads` (reads ``mask``, ``keypoints``, ``direction``;
    ``line`` is a training aid). ``decoders`` as in :func:`manyseg.instances.instance_heads`.
    """
    heads = {"mask": {"type": "segmentation", "out_channels": 1},
             "line": {"type": "segmentation", "out_channels": 1},
             "keypoints": {"type": "segmentation", "out_channels": 3, "bias_init": HEATMAP_PRIOR},
             "direction": {"type": "regression", "out_channels": 4}}
    return _attach(heads, decoders)
