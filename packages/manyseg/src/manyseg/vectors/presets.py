"""Head presets for the trained vector-output models (corner-line instances, unified roads).

Both presets use the built-in head types only. The heatmap heads trained with the CenterNet focal
loss (``corner``, ``keypoints``) start at the prior ``p = 0.1`` (``bias_init = logit(0.1)``), so
the loss is not swamped by negatives at initialisation (E036: about 200 at ``p = 0.5``; the
experiments set the bias to -2.19, ``p = 0.1007``). Head
names are those the vector decoders read (``polygons_from_corners``, ``graph_from_road_heads``).
Losses are in :mod:`manyseg.vectors.losses` and metrics in :mod:`manyseg.vectors.metrics`;
``manyseg-bench`` has training tasks that use them (``CornerLineTask``, ``RoadGraphTask``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..heads._presets import HEATMAP_PRIOR, attach_decoders


def corner_line_heads(decoders: None | str | Mapping[str, Any] = None) -> dict[str, dict]:
    """Head configs for ``create_model(heads=...)`` of the corner-line instance model, for
    footprints and roof faces (E036, E038, E047; fields use ``instance_heads``, E046, E052): the
    instance heads of :func:`manyseg.instances.instance_heads` (``extent``, ``boundary``,
    ``distance``) plus ``corner`` (1 channel, logits of a Gaussian corner heatmap, bias
    ``logit(0.1)``) and ``direction`` (2 channels, regression of the wall direction as ``(cos 2t,
    sin 2t)``).

    Targets: :func:`manyseg.instances.instance_targets` and
    :func:`manyseg.vectors.corner_targets`; decoding: instances, then
    :func:`manyseg.vectors.polygons_from_corners`. ``decoders`` as in ``instance_heads``.
    """
    from ..instances import instance_heads

    heads = {**instance_heads(),
             "corner": {"type": "segmentation", "out_channels": 1, "bias_init": HEATMAP_PRIOR},
             "direction": {"type": "regression", "out_channels": 2}}
    return attach_decoders(heads, decoders)


def road_heads(decoders: None | str | Mapping[str, Any] = None) -> dict[str, dict]:
    """Head configs for the unified road model (E039, E044): ``mask`` (1, road-surface
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
    return attach_decoders(heads, decoders)
