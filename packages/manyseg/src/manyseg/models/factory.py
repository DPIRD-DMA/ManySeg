"""``create_model``: argument parsing over component composition."""

from __future__ import annotations

import functools
import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from torch import nn

from ..adapters import ProjectionAdapter
from ..backbones import TimmBackbone
from ..decoders import DECODERS
from ..features import FeatureSpec
from ..heads import create_head
from ..registry import check_module_names
from .dense import DenseModel

FUSION_TYPES = ("early", "intermediate", "late")
_INPUT_KEYS = {"in_channels", "backbone", "pretrained", "channel_adapter", "alignment",
               "share_norm", "backbone_kwargs", "out_indices", "rgb_indices", "kind", "temporal"}


def check_input_keys(name: str, cfg: Mapping) -> None:
    """A raster or time-series input config: known keys, ``in_channels`` given."""
    if "channels" in cfg:
        raise ValueError(f"input {name!r}: 'channels' was renamed 'in_channels' (heads: "
                         "'out_channels'; vector inputs: 'num_features')")
    unknown = set(cfg) - _INPUT_KEYS
    if unknown:
        raise ValueError(f"input {name!r}: unknown keys {sorted(unknown)}")
    if "in_channels" not in cfg:
        raise ValueError(f"input {name!r}: 'in_channels' is required")


def share_weights(encoder: nn.Module) -> nn.Module:
    """A copy of ``encoder`` that shares every parameter (the same ``Parameter`` objects) except
    those of its BatchNorm layers, which get their own affine weights and running statistics."""
    import copy

    clone = copy.deepcopy(encoder)
    for (_, mine), (_, theirs) in zip(clone.named_modules(), encoder.named_modules()):
        if isinstance(mine, nn.modules.batchnorm._BatchNorm):
            continue
        for name, p in theirs.named_parameters(recurse=False):
            setattr(mine, name, p)
    return clone


def _build_inputs(inputs, fusion, *, backbone, pretrained, channel_adapter, out_indices,
                  backbone_kwargs):
    if isinstance(fusion, str):
        raise TypeError(f"fusion takes a dict, e.g. fusion={{'type': {fusion!r}}}")
    fusion = dict(fusion or {"type": "intermediate"})
    ftype = fusion.pop("type", "intermediate")
    if ftype not in FUSION_TYPES:
        raise ValueError(f"unknown fusion type {ftype!r}; choose from {FUSION_TYPES}")
    for n, cfg in inputs.items():
        check_input_keys(n, cfg)
        if cfg.get("kind", "raster") not in ("raster", "timeseries"):
            raise ValueError(f"input {n!r}: kind {cfg['kind']!r} is not a raster kind")
        if "temporal" in cfg and cfg.get("kind") != "timeseries":
            raise ValueError(f"input {n!r}: 'temporal' applies to "
                             "{'kind': 'timeseries'} inputs only")
        for suffix in ("mask", "dates"):
            if cfg.get("kind") == "timeseries" and f"{n}_{suffix}" in inputs:
                raise ValueError(f"input name {n}_{suffix} is reserved for the {suffix} of "
                                 f"time series {n!r}")
    series = {n for n, cfg in inputs.items() if cfg.get("kind") == "timeseries"}
    from ..fusion import EarlyFusionEncoder, MultiEncoder
    from ..temporal import TemporalEncoder, parse_temporal

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

    if ftype == "early" and series:
        raise ValueError(f"early fusion stacks channels and cannot take the time series "
                         f"{sorted(series)}; use intermediate or late fusion")
    if ftype == "early":
        if fusion:
            raise ValueError(f"early fusion takes no options, got {sorted(fusion)}")
        # One encoder sees the stacked inputs, so per-input encoder keys must agree (they then
        # override the top-level arguments); silently dropping them built the wrong backbone.
        shared_cfg = {}
        for key in ("backbone", "pretrained", "channel_adapter", "out_indices",
                    "backbone_kwargs"):
            vals = {n: cfg[key] for n, cfg in inputs.items() if key in cfg}
            if len({repr(v) for v in vals.values()}) > 1:
                raise ValueError(f"early fusion stacks all inputs into one encoder; per-input "
                                 f"{key!r} values differ: {vals}")
            if vals:
                shared_cfg[key] = next(iter(vals.values()))
        for n, cfg in inputs.items():
            if str(cfg.get("backbone", "")).startswith("shared:") or "share_norm" in cfg:
                raise ValueError(f"early fusion stacks all inputs into one encoder, so input "
                                 f"{n!r} cannot share another input's encoder ('shared:' / "
                                 "'share_norm'); use intermediate or late fusion")
            if "rgb_indices" in cfg:
                raise ValueError(f"early fusion: input {n!r} sets 'rgb_indices', which would "
                                 "index the stacked channels; adapt the stem with intermediate "
                                 "fusion or stack the inputs yourself")
            if cfg.get("alignment", "same_grid") != "same_grid":
                raise ValueError(f"early fusion stacks channels, so every input must be "
                                 f"same_grid; input {n!r} is {cfg['alignment']!r}")
        total = sum(cfg["in_channels"] for cfg in inputs.values())
        channels = {n: cfg["in_channels"] for n, cfg in inputs.items()}
        return EarlyFusionEncoder(list(inputs), make_bb(shared_cfg, total), channels), ftype, {}

    encoders, adapters = {}, {}
    for n, cfg in inputs.items():
        bname = cfg.get("backbone", backbone)
        if isinstance(bname, str) and bname.startswith("shared:"):
            target = bname.split(":", 1)[1]
            if n in series or target in series:
                raise ValueError(f"input {n!r}: a 'shared:' backbone cannot involve a time series "
                                 "yet; give each input its own encoder")
            if target not in encoders:
                raise ValueError(f"input {n!r} shares {target!r}, which must be defined earlier")
            # Weights shared, BatchNorm per input by default: one BN shared by inputs with
            # different statistics blends its running statistics, and evaluation then applies
            # the blend to every input (E015: val 0.50-0.65 vs 0.89).
            encoders[n] = (encoders[target] if cfg.get("share_norm", False)
                           else share_weights(encoders[target]))
            if cfg["in_channels"] != encoders[target].in_channels:
                adapters[n] = ProjectionAdapter(cfg["in_channels"], encoders[target].in_channels)
        else:
            if "share_norm" in cfg:
                raise ValueError(f"input {n!r}: 'share_norm' only applies to a "
                                 "'shared:<input>' backbone")
            encoders[n] = make_bb(cfg, cfg["in_channels"])
            if n in series:
                kind, opts = parse_temporal(n, cfg.get("temporal", "mean"))
                encoders[n] = TemporalEncoder(encoders[n], kind, opts)
    alignment = {n: cfg.get("alignment", "same_grid") for n, cfg in inputs.items()}
    if len(inputs) == 1 and set(fusion) & {"op", "op_kwargs", "strides"}:
        raise ValueError(f"fusion options {sorted(set(fusion) & {'op', 'op_kwargs', 'strides'})} "
                         f"need at least two raster inputs; got only {list(inputs)}")
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
    pretrained: bool | str = True,
    in_channels: int | None = None,
    channel_adapter: str = "repeat",
    out_indices: Sequence[int] | None = None,
    backbone_kwargs: Mapping[str, Any] | None = None,
    decoder: str | Mapping[str, Any] | Callable[..., nn.Module] = "unet",
    decoder_kwargs: Mapping[str, Any] | None = None,
    num_classes: int | None = None,
    heads: Mapping[str, Mapping[str, Any]] | None = None,
    pad_to_stride: bool = True,
) -> DenseModel:
    """Build a dense prediction model (segmentation, regression, classification, ...).

    Exactly one of ``num_classes`` or ``heads`` is required. With ``num_classes`` the model
    returns a Tensor; with ``heads`` a ``dict[str, Tensor]`` keyed by head name, even for a
    single head. Full reference: ``docs/api_design.md``.

    Args:
        backbone: A timm model name, or an ``nn.Module`` with ``feature_specs`` and
            ``input_divisor``.
        inputs: Several inputs, ``{name: config}`` (the first raster input is the primary).
            Raster or time-series configs: ``in_channels`` (required), ``kind``
            (``"raster"`` / ``"timeseries"``), ``backbone`` (or ``"shared:<input>"``),
            ``pretrained``, ``channel_adapter``, ``alignment`` (``"same_grid"`` /
            ``"same_extent"``), ``share_norm``, ``backbone_kwargs``, ``out_indices``,
            ``rgb_indices``, ``temporal`` (time series). Vector (metadata) configs:
            ``{"kind": "vector"}`` with ``fields`` or ``num_features``, ``dim``, ``dropout``,
            ``conditioning`` (``"film"`` / ``"head"`` / ``"channels"``).
        fusion: ``{"type": "early" | "intermediate" | "late", "op": ..., "op_kwargs": ...,
            "strides": [...]}`` for several raster inputs.
        pretrained: ``True`` (timm weights), ``False``, or a ``manyseg.weights`` tag such as
            ``"ssl4eo_s2_dino"``.
        in_channels: Input bands of a single-input model (default 3; with ``inputs=``, set it
            per input instead).
        channel_adapter: How the RGB stem is adapted to ``in_channels``.
        out_indices: Encoder levels to return (timm ``out_indices``).
        backbone_kwargs: Passed to ``timm.create_model``; also takes ``rgb_indices`` (input
            channels carrying R, G, B for the ``rgb_plus_*`` / ``projection`` adapters).
        decoder: A decoder name, a decoder class or callable, or one dict
            ``{"type": name, **options}``.
        decoder_kwargs: Options for the decoder (not together with the same keys in a dict
            ``decoder``).
        num_classes: One segmentation head with this many output channels.
        heads: ``{name: config}`` with ``type`` (``"segmentation"``: pixel scope,
            ``"classification"``: image scope, ``"regression"``: either), ``out_channels``
            (required), ``scope`` (``"pixel"`` / ``"image"``), ``source``, ``scale``,
            ``uncertainty``, ``decoder`` (a per-head decoder), ``input`` and ``share``
            (per-input heads), and head options (``kernel_size``, ``bias_init``,
            ``dropout``, ``init_std``).
        pad_to_stride: Pad inputs to the encoder stride and crop outputs back (default);
            ``False`` resizes outputs to the input size instead.
    """
    if (num_classes is None) == (heads is None):
        raise ValueError("pass exactly one of num_classes= or heads=")
    if num_classes is not None and (isinstance(num_classes, bool)
                                    or not isinstance(num_classes, int) or num_classes < 1):
        raise ValueError(f"num_classes must be a positive int, got {num_classes!r}")
    if isinstance(decoder, Mapping):  # {"type": "unet", **kwargs}, as for per-head decoders
        d = dict(decoder)
        if "type" not in d:
            raise ValueError("decoder={...} needs a 'type', e.g. {'type': 'unet', 'out_stride': 2}")
        both = set(d) & set(decoder_kwargs or {})
        if both:
            raise ValueError(f"decoder keys {sorted(both)} given both in decoder={{...}} and "
                             "decoder_kwargs")
        decoder = d.pop("type")
        decoder_kwargs = {**dict(decoder_kwargs or {}), **d}
    if heads is not None:
        check_module_names("head", heads)
    if inputs is not None:
        check_module_names("input", inputs)
        if in_channels is not None:
            raise ValueError("in_channels= is for a single-input model; with inputs= give each "
                             "input its own 'in_channels'")
    if inputs is not None:
        from ..conditioning.build import build_conditioned_model, has_kinds

        if has_kinds(inputs):  # vector (metadata) inputs: see manyseg.conditioning
            return build_conditioned_model(
                create_model, inputs=inputs, fusion=fusion, backbone=backbone,
                pretrained=pretrained, channel_adapter=channel_adapter, out_indices=out_indices,
                backbone_kwargs=backbone_kwargs, decoder=decoder, decoder_kwargs=decoder_kwargs,
                num_classes=num_classes, heads=heads, pad_to_stride=pad_to_stride)

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
            in_channels=3 if in_channels is None else in_channels,
            channel_adapter=channel_adapter,
            out_indices=out_indices,
            **dict(backbone_kwargs or {}),
        )
    else:
        bb = backbone

    def resolve(d):
        return DECODERS.get_or_raise(d) if isinstance(d, str) else d

    def build_decoder(dec_cls, dkw, specs=None):
        per_input = specs is not None
        specs = tuple(specs if per_input else bb.feature_specs)
        wants = (dec_cls.wants_input_for(dkw) if hasattr(dec_cls, "wants_input_for")
                 else getattr(dec_cls, "wants_input", False))
        if per_input and wants:
            raise ValueError(f"decoder {dec_cls.__name__} reads the raw input image, which "
                             "per-input heads do not support (e.g. fastai_unet, input_cross); "
                             "choose a decoder that reads features only")
        if wants and inputs is not None:
            raise ValueError(f"decoder {dec_cls.__name__} needs the input image (fastai_unet, "
                             "input_cross), which models built with inputs= (several inputs, "
                             "a time series or metadata) don't have; use a decoder that reads "
                             "features only, e.g. 'unet'")
        if ftype == "late" and not per_input:
            from ..fusion import LateFusionDecoder

            return LateFusionDecoder(specs, bb.groups, lambda sp: dec_cls(sp, **dkw),
                                     op=fopts["op"], alignment=[bb.alignment[n] for n in bb.names])
        if wants:
            specs = (FeatureSpec(bb.in_channels, 1, "input"), *specs)
        return dec_cls(specs, **dkw)

    single = None
    if num_classes is not None:
        heads = {"logits": {"type": "segmentation", "out_channels": num_classes}}
        single = "logits"
    heads = {n: dict(cfg) for n, cfg in heads.items()}
    # Per-input heads ("input": <raster input>) read that input's own encoder features through
    # their own decoder (default: the `decoder` type), e.g. land cover at each date for semantic
    # change; "share": <head> reuses another per-input head's decoder and head modules.
    for n, cfg in heads.items():
        if cfg.get("source") == "encoder" and ("decoder" in cfg or "input" in cfg):
            raise ValueError(f"head {n!r}: 'source': 'encoder' reads the encoder directly, so it "
                             "cannot also have its own decoder ('decoder' / 'input')")
    head_inputs = {n: cfg.pop("input") for n, cfg in heads.items() if "input" in cfg}
    shares = {n: cfg.pop("share") for n, cfg in heads.items() if "share" in cfg}
    for n, inp in head_inputs.items():
        if not hasattr(bb, "encoders"):
            raise ValueError(f"head {n!r}: 'input' needs a multi-input model (inputs=...)")
        if ftype != "intermediate":
            raise ValueError(f"head {n!r}: 'input' heads need intermediate fusion, not {ftype!r} "
                             "(late fusion already decodes each input)")
        if inp not in bb.names:
            raise ValueError(f"head {n!r}: unknown input {inp!r}; raster inputs are "
                             f"{list(bb.names)}")
        if heads[n].get("scope", "pixel") != "pixel":
            raise ValueError(f"head {n!r}: 'input' heads are pixel heads")
        if getattr(bb.encoders[inp], "temporal", False):
            raise ValueError(f"head {n!r}: 'input' cannot be a time series yet")
        if bb.alignment[inp] != "same_grid":
            raise ValueError(f"head {n!r}: 'input' heads need an input on the primary grid "
                             f"(same_grid); {inp!r} is {bb.alignment[inp]!r}")
    for n, other in shares.items():
        if n not in head_inputs or other not in head_inputs or other in shares:
            raise ValueError(f"head {n!r}: 'share' links two per-input heads ('input' set on "
                             f"both); {other!r} must be one that does not share itself")
        if {k: v for k, v in heads[n].items() if k != "decoder"} != \
                {k: v for k, v in heads[other].items() if k != "decoder"}:
            raise ValueError(f"head {n!r}: 'share' needs the same head config as {other!r}")
        if bb.encoders[head_inputs[n]].feature_specs != \
                bb.encoders[head_inputs[other]].feature_specs:
            raise ValueError(f"head {n!r}: 'share' needs inputs with the same feature specs")
    reads_kwargs = any("decoder" not in heads[n] for n in head_inputs)
    for n in head_inputs:  # default: the model's decoder type and kwargs, on this input
        heads[n].setdefault("decoder", {"type": decoder, **dict(decoder_kwargs or {})})
    task_cfgs = {n: cfg.pop("decoder") for n, cfg in heads.items() if "decoder" in cfg}
    for n in task_cfgs:
        if heads[n].get("scope", "pixel") != "pixel":
            raise ValueError(f"head {n!r}: task-specific decoders are only for pixel heads")
    # pixel heads read the decoder by default, image heads only with "source": "decoder"
    needs_shared = any(n not in task_cfgs and heads[n].get(
        "source", "decoder" if heads[n].get("scope", "pixel") == "pixel" else "encoder")
        == "decoder" for n in heads)
    shared = build_decoder(resolve(decoder), dict(decoder_kwargs or {})) if needs_shared else None
    if not needs_shared:
        resolve(decoder)  # an unknown name is an error even when no shared decoder is built
        if decoder_kwargs and not reads_kwargs:
            raise ValueError("decoder_kwargs given but no head reads the shared decoder (image "
                             "heads, or every pixel head has its own 'decoder'), so they would "
                             "be ignored")
    task_decoders = {}
    for n, dcfg in task_cfgs.items():
        if n in shares:
            continue
        specs = bb.encoders[head_inputs[n]].feature_specs if n in head_inputs else None
        if isinstance(dcfg, Mapping):
            dcfg = dict(dcfg)
            if "type" not in dcfg:
                raise ValueError(f"head {n!r}: 'decoder': {{...}} needs a 'type', e.g. "
                                 "{'type': 'fpn'}")
            task_decoders[n] = build_decoder(resolve(dcfg.pop("type")), dcfg, specs)
        else:
            task_decoders[n] = build_decoder(resolve(dcfg), {}, specs)
    for n, other in shares.items():
        task_decoders[n] = task_decoders[other]

    enc_specs = bb.feature_specs
    if ftype == "late":
        enc_specs = enc_specs[bb.groups[0][0]:bb.groups[0][1]]
    deepest = max(enc_specs, key=lambda s: s.stride)
    built, sources = {}, {}
    for name, cfg in heads.items():
        if name in shares:
            continue
        dec_for_head = task_decoders.get(name, shared)
        dspec = dec_for_head.out_spec if dec_for_head is not None else deepest
        built[name], sources[name] = create_head(dspec, deepest, cfg)
        if name in task_decoders:
            sources[name] = "task"
    for name, other in shares.items():
        built[name], sources[name] = built[other], "task"
    built = {n: built[n] for n in heads}  # keep the configured head order
    model = DenseModel(bb, shared, built, sources, single_output=single,
                       pad_to_stride=pad_to_stride, task_decoders=task_decoders,
                       head_inputs=head_inputs)
    model._restrict_to_read_levels()
    return model


_build = create_model


@functools.wraps(_build)
def _create_and_record(*args: Any, **kwargs: Any) -> DenseModel:
    model = _build(*args, **kwargs)
    # Record the arguments, so manyseg.save can rebuild the model. Only plain data can be
    # stored: models built from modules or callables get build_config None.
    bound = inspect.signature(_build).bind(*args, **kwargs)
    bound.apply_defaults()
    try:
        model.build_config = json.loads(json.dumps(dict(bound.arguments)))
    except TypeError:
        model.build_config = None
    return model


if TYPE_CHECKING:  # type checkers see the full signature
    create_model = _build
else:
    create_model = _create_and_record
