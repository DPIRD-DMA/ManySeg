"""Head presets for boundary + distance instance segmentation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..heads._presets import attach_decoders


def instance_heads(
    num_classes: int = 1,
    distance: str = "normalized",
    decoders: None | str | Mapping[str, Any] = None,
) -> dict[str, dict]:
    """Head configs for ``create_model(heads=...)``: ``extent``, ``boundary``, ``distance``.

    Args:
        num_classes: Foreground classes. 1 -> binary extent (1 channel, sigmoid/BCE);
            >1 -> ``num_classes + 1`` channels with background at index 0 (softmax/CE).
        distance: ``"normalized"`` or ``"absolute"``; must match the targets
            (:func:`manyseg.instances.instance_targets`). Heads output raw values either way.
        decoders: ``None`` -> all heads share the model decoder; ``"separate"`` -> each head
            gets its own ``"unet"`` decoder (whatever the model's ``decoder=``); a decoder name ->
            each head gets its own decoder of that type; a mapping ``{head: decoder}`` ->
            only those heads get their own decoder (name or ``{"type": ..., **kwargs}``).
    """
    if distance not in ("normalized", "absolute"):
        raise ValueError(f"distance must be 'normalized' or 'absolute', got {distance!r}")
    if num_classes < 1:
        raise ValueError("num_classes must be >= 1")
    heads = {
        "extent": {"type": "segmentation",
                   "out_channels": 1 if num_classes == 1 else num_classes + 1},
        "boundary": {"type": "segmentation", "out_channels": 1},
        "distance": {"type": "regression", "out_channels": 1},
    }
    return attach_decoders(heads, decoders)
