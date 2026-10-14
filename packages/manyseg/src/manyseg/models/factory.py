"""``create_model``: argument parsing over component composition."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from torch import nn

from ..backbones import TimmBackbone
from ..decoders import DECODERS
from ..heads import create_head
from .dense import DenseModel


def create_model(
    backbone: str | nn.Module = "resnet34",
    *,
    pretrained: bool = True,
    in_channels: int = 3,
    channel_adapter: str = "repeat",
    out_indices: Sequence[int] | None = None,
    backbone_kwargs: Mapping[str, Any] | None = None,
    decoder: str | Callable[..., nn.Module] = "unet",
    decoder_kwargs: Mapping[str, Any] | None = None,
    num_classes: int | None = None,
    heads: Mapping[str, Mapping[str, Any]] | None = None,
    pad_to_stride: bool = True,
) -> DenseModel:
    """Build a dense prediction model.

    Exactly one of ``num_classes`` (returns a Tensor) or ``heads`` (returns a dict) is required.
    See ``docs/api_design.md``.
    """
    if (num_classes is None) == (heads is None):
        raise ValueError("pass exactly one of num_classes= or heads=")

    if isinstance(backbone, str):
        bb = TimmBackbone(
            backbone,
            pretrained=pretrained,
            in_channels=in_channels,
            channel_adapter=channel_adapter,
            out_indices=out_indices,
            **dict(backbone_kwargs or {}),
        )
    else:
        bb = backbone

    dec_cls = DECODERS.get_or_raise(decoder) if isinstance(decoder, str) else decoder
    dec = dec_cls(bb.feature_specs, **dict(decoder_kwargs or {}))

    single = None
    if num_classes is not None:
        heads = {"logits": {"type": "segmentation", "channels": num_classes}}
        single = "logits"
    deepest = max(bb.feature_specs, key=lambda s: s.stride)
    built, sources = {}, {}
    for name, cfg in heads.items():
        built[name], sources[name] = create_head(dec.out_spec, deepest, cfg)
    return DenseModel(bb, dec, built, sources, single_output=single, pad_to_stride=pad_to_stride)
