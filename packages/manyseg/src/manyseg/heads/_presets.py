"""What the head presets (instances, points, vectors) share."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

#: Bias of a focal-loss heatmap head: ``logit(0.1)``, CenterNet's prior, so the loss is not swamped
#: by negatives at initialisation.
HEATMAP_PRIOR = math.log(0.1 / 0.9)


def attach_decoders(heads: dict, decoders: None | str | Mapping[str, Any]) -> dict:
    """Give ``heads`` their own decoders, in place: ``None`` -> they all share the model decoder;
    ``"separate"`` -> each gets its own ``"unet"``; a decoder name -> each gets its own of that
    type; a mapping ``{head: decoder}`` -> only those heads get one."""
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
