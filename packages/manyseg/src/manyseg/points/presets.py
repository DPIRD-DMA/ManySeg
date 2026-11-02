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

    ``activation``: ``"softplus"`` (default since E011c) or ``None`` (raw values, as in E011).
    The model applies it, so training and inference use the same one.
    Results on a downstream drone dataset (792 x 792 tiles, 1-20 plants of about 115 px,
    sigma 12, 45 val images, ConvNeXtV2-T U-Net, density head on the shared decoder,
    :func:`density_loss` MSE with the default ``value_scale``, 25 epochs, one seed):

    * ``None``: count MAE 7.5 summed raw, 10.1 clamped at 0 (clamping a model trained without
      an activation makes counts worse); mean signed error +10.1, correlation 0.80.
    * ``"softplus"`` (bias starts at -4): count MAE 2.9. It leaves a background floor of about
      +1.75 objects per image (2.4 on an empty tile), because softplus never reaches 0
      (softplus(-4) = 0.018 per pixel is about 12 objects over a 792 x 792 tile at the start
      of training); correlation 0.72. Override the start with the head's ``bias_init``.
    * ReLU was tried and removed: with bias 0 the head died (its bias drifted to -0.005 and
      the map went to 0; the largest count on any image was 0.06 against true counts up to
      27; MAE 3.5, correlation 0.37). Don't re-add it without a fix for dead units.

    Peak counting on a heatmap head scored count MAE 1.0 on the same data: for a few large
    objects per image, count peaks. On OAM-TCD (many small trees per tile, sigma 2 px) the
    plain density head halved count MAE against peak counting at a fixed threshold (E011).
    E011c reran that density arm with softplus (3 seeds): count MAE val 20.4 vs 20.4, test
    19.8 vs 19.3 (no worse), with the mean signed count error closer to 0 (val -1.2 vs -4.9).
    Validate on your data.

    ``decoder`` as one entry of :func:`point_heads` ``decoders``.
    """
    heads = {"density": {"type": "regression", "out_channels": 1}}
    if activation is not None:
        heads["density"]["activation"] = activation
        # start low (softplus(-4) = 0.018) while keeping a gradient (sigmoid(-4) = 0.018);
        # softplus(0) = 0.69 per pixel would start the count far too high
        heads["density"]["bias_init"] = -4.0
    return _attach(heads, None if decoder is None else {"density": decoder})
