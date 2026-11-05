"""Head presets for point (centre heatmap [+ offsets]) and density outputs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


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


def point_heads(heatmap: str = "regression", offsets: bool = False,
                decoders: None | str | Mapping[str, Any] = None) -> dict[str, dict]:
    """``heatmap`` (1 channel) and optional ``offset`` (2 channels, ``(dy, dx)`` to the centre).

    ``heatmap="regression"`` pairs with MSE on Gaussian targets; ``"segmentation"`` treats it
    as logits (focal / penalty-reduced focal loss, CenterNet). ``decoders`` as in
    :func:`manyseg.instances.instance_heads`.
    """
    if heatmap not in ("regression", "segmentation"):
        raise ValueError(f"heatmap must be 'regression' or 'segmentation', got {heatmap!r}")
    heads = {"heatmap": {"type": heatmap, "out_channels": 1}}
    if heatmap == "segmentation":
        # CenterNet: start at prior p=0.1 so the focal loss is not swamped by negatives.
        heads["heatmap"]["bias_init"] = math.log(0.1 / 0.9)
    if offsets:
        heads["offset"] = {"type": "regression", "out_channels": 2}
    return _attach(heads, decoders)


def density_head(decoder: Any = None, *, activation: str | None = "softplus") -> dict[str, dict]:
    """A 1-channel density map whose integral is the count (:func:`count_from_density`).
    Train with :func:`density_loss`.

    ``activation``: ``"softplus"`` (default) or ``None`` (raw values). The model applies it, so
    training and inference use the same one. Softplus never reaches 0, so a mostly empty tile
    still sums to a small positive count; the head's bias starts at -4 to keep that low, and
    ``bias_init`` overrides it. Don't clamp a raw head at 0 after training: that made counts worse.

    For a few large objects per image, count peaks of a heatmap head instead: a small background
    bias summed over every pixel adds up. For many small objects a density head does well.
    ``docs/api_design.md`` has the measurements. Validate on your data.

    ``decoder`` as one entry of :func:`point_heads` ``decoders``.
    """
    # ReLU was tried and removed: with bias 0 the head died (its bias drifted below 0 and the map
    # went to 0). Don't re-add it without a fix for dead units.
    heads = {"density": {"type": "regression", "out_channels": 1}}
    if activation is not None:
        heads["density"]["activation"] = activation
        # start low (softplus(-4) = 0.018) while keeping a gradient (sigmoid(-4) = 0.018);
        # softplus(0) = 0.69 per pixel would start the count far too high
        heads["density"]["bias_init"] = -4.0
    return _attach(heads, None if decoder is None else {"density": decoder})
