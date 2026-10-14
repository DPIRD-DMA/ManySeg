"""``create_model``: argument parsing over component composition."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from torch import nn

from ..adapters import ProjectionAdapter
from ..backbones import TimmBackbone
from ..decoders import DECODERS
from ..features import FeatureSpec
from ..heads import create_head
from .dense import DenseModel

FUSION_TYPES = ("early", "intermediate", "late")
_INPUT_KEYS = {"channels", "backbone", "pretrained", "channel_adapter", "alignment",
               "backbone_kwargs", "out_indices", "rgb_indices"}


def _build_inputs(inputs, fusion, *, backbone, pretrained, channel_adapter, out_indices,
                  backbone_kwargs):
    fusion = dict(fusion or {"type": "intermediate"})
    ftype = fusion.pop("type", "intermediate")
    if ftype not in FUSION_TYPES:
        raise ValueError(f"unknown fusion type {ftype!r}; choose from {FUSION_TYPES}")
    for n, cfg in inputs.items():
        unknown = set(cfg) - _INPUT_KEYS
        if unknown:
            raise ValueError(f"input {n!r}: unknown keys {sorted(unknown)}")
        if "channels" not in cfg:
            raise ValueError(f"input {n!r}: 'channels' is required")
    from ..fusion import EarlyFusionEncoder, MultiEncoder

    def make_bb(cfg, channels):
        name = cfg.get("backbone", backbone)
        if not isinstance(name, str):
            return name
        return TimmBackbone(name, pretrained=cfg.get("pretrained", pretrained),
                            in_channels=channels,
                            channel_adapter=cfg.get("channel_adapter", channel_adapter),
                            out_indices=cfg.get("out_indices", out_indices),
                            rgb_indices=cfg.get("rgb_indices"),
                            **dict(cfg.get("backbone_kwargs", backbone_kwargs) or {}))

    if ftype == "early":
        if fusion:
            raise ValueError(f"early fusion takes no options, got {sorted(fusion)}")
        total = sum(cfg["channels"] for cfg in inputs.values())
        return EarlyFusionEncoder(list(inputs), make_bb({}, total)), ftype, {}

    encoders, adapters = {}, {}
    for n, cfg in inputs.items():
        bname = cfg.get("backbone", backbone)
        if isinstance(bname, str) and bname.startswith("shared:"):
            target = bname.split(":", 1)[1]
            if target not in encoders:
                raise ValueError(f"input {n!r} shares {target!r}, which must be defined earlier")
            encoders[n] = encoders[target]
            if cfg["channels"] != encoders[target].in_channels:
                adapters[n] = ProjectionAdapter(cfg["channels"], encoders[target].in_channels)
        else:
            encoders[n] = make_bb(cfg, cfg["channels"])
    alignment = {n: cfg.get("alignment", "same_grid") for n, cfg in inputs.items()}
    op = fusion.pop("op", "concat")
    strides = fusion.pop("strides", None)
    op_kwargs = fusion.pop("op_kwargs", None)
    if fusion:
        raise ValueError(f"unknown fusion options {sorted(fusion)}")
    enc = MultiEncoder(encoders, mode=ftype, op=op, strides=strides, alignment=alignment,
                       adapters=adapters, op_kwargs=op_kwargs)
    return enc, ftype, {"op": op}


def create_model(
    backbone: str | nn.Module = "resnet34",
    *,
    inputs: Mapping[str, Mapping[str, Any]] | None = None,
    fusion: Mapping[str, Any] | None = None,
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

    ftype, fopts = None, {}
    if inputs is not None:
        bb, ftype, fopts = _build_inputs(
            inputs, fusion, backbone=backbone, pretrained=pretrained,
            channel_adapter=channel_adapter, out_indices=out_indices,
            backbone_kwargs=backbone_kwargs)
    elif fusion is not None:
        raise ValueError("fusion= requires inputs=")
    elif isinstance(backbone, str):
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
    specs = tuple(bb.feature_specs)
    dkw = dict(decoder_kwargs or {})
    if ftype == "late":
        from ..fusion import LateFusionDecoder

        dec = LateFusionDecoder(specs, bb.groups, lambda sp: dec_cls(sp, **dkw), op=fopts["op"])
        specs = specs[bb.groups[0][0]:bb.groups[0][1]]
    else:
        if getattr(dec_cls, "wants_input", False):
            if inputs is not None:
                raise ValueError(f"decoder {decoder!r} needs the input image; not supported "
                                 "with multiple inputs")
            specs = (FeatureSpec(bb.in_channels, 1, "input"), *specs)
        dec = dec_cls(specs, **dkw)

    single = None
    if num_classes is not None:
        heads = {"logits": {"type": "segmentation", "channels": num_classes}}
        single = "logits"
    deepest = max((s for s in specs if s.name != "input"), key=lambda s: s.stride)
    built, sources = {}, {}
    for name, cfg in heads.items():
        built[name], sources[name] = create_head(dec.out_spec, deepest, cfg)
    return DenseModel(bb, dec, built, sources, single_output=single, pad_to_stride=pad_to_stride)
