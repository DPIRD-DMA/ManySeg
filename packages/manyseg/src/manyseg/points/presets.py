"""Head presets for point (centre heatmap [+ offsets]) and density outputs."""

from __future__ import annotations

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
    heads = {"heatmap": {"type": heatmap, "channels": 1}}
    if offsets:
        heads["offset"] = {"type": "regression", "channels": 2}
    return _attach(heads, decoders)


def density_head(decoder: Any = None) -> dict[str, dict]:
    """A 1-channel density map whose integral is the count."""
    heads = {"density": {"type": "regression", "channels": 1}}
    return _attach(heads, None if decoder is None else {"density": decoder})
