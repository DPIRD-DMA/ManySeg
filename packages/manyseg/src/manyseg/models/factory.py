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
from ..heads.heads import head_scope
from ..registry import check_module_names
from .dense import DenseModel

FUSION_TYPES = ("early", "intermediate", "late")
_INPUT_KEYS = {"in_channels", "backbone", "pretrained", "channel_adapter", "alignment",
               "share_norm", "backbone_kwargs", "out_indices", "rgb_indices", "kind", "temporal"}


def check_input_keys(name: str, cfg: Mapping) -> None:
    """A raster or time-series input config: known keys, ``in_channels`` given."""
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


def _check_shared_adapter(name: str, cfg: Mapping, encoder_channels: int) -> None:
    """A shared input reaches the encoder unchanged, or through a learned projection it asks for
    itself. Only its own setting counts: the top-level ``channel_adapter`` adapts a backbone's
    stem, and a shared input has no stem of its own."""
    channels, adapter = cfg["in_channels"], cfg.get("channel_adapter")
    if adapter not in (None, "projection"):
        raise ValueError(f"input {name!r}: a shared encoder's stem is already built, so "
                         f"channel_adapter {adapter!r} can't apply; only 'projection' can")
    if adapter is None and channels != encoder_channels:
        raise ValueError(
            f"input {name!r} has {channels} channels (metadata in 'channels' mode included) and "
            f"the encoder it shares expects {encoder_channels}. Either give it its own encoder, "
            f"or add a learned 1x1 projection in front of the shared one with "
            f"{{'channel_adapter': 'projection'}} on this input.")
    if adapter == "projection" and channels == encoder_channels:
        raise ValueError(f"input {name!r}: channel_adapter 'projection' has nothing to adapt: it "
                         f"has the {channels} channels the encoder it shares expects")


def backbone_kwargs_with_rgb(name: str, cfg: Mapping, backbone_kwargs) -> dict:
    """An input's backbone kwargs: its own ``backbone_kwargs`` (else the top-level ones), with
    its ``rgb_indices``, which win over an ``rgb_indices`` in the top-level ``backbone_kwargs``
    (giving it both in the input's own ``backbone_kwargs`` and as its key is an error)."""
    bkw = dict(cfg.get("backbone_kwargs", backbone_kwargs) or {})
    if cfg.get("rgb_indices") is not None:
        if "rgb_indices" in dict(cfg.get("backbone_kwargs") or {}):
            raise ValueError(f"input {name!r}: 'rgb_indices' given both as a key and in its "
                             "'backbone_kwargs'; give it once")
        bkw["rgb_indices"] = cfg["rgb_indices"]
    return bkw


_MODULE_IGNORES = ("pretrained", "channel_adapter", "out_indices", "backbone_kwargs",
                   "rgb_indices")


def check_module_input(name: str, cfg: Mapping, module: nn.Module, channels: int) -> None:
    """An input encoded by a backbone given as an ``nn.Module``: the timm options can't apply,
    and its channel count must be the module's (when the module exposes ``in_channels``)."""
    given = sorted(k for k in _MODULE_IGNORES if k in cfg)
    if given:
        raise ValueError(f"input {name!r}: {given} configure a timm backbone, but its backbone "
                         "is an nn.Module (already built); configure the module itself")
    _check_module_channels(module, channels, f"input {name!r} has")


def _check_module_channels(module: nn.Module, channels: int, what: str) -> None:
    expected = getattr(module, "in_channels", None)
    if isinstance(expected, int) and expected != channels:
        raise ValueError(f"{what} {channels} channels but the backbone module "
                         f"{type(module).__name__} takes {expected} (its in_channels)")


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

    built_timm = False

    def make_bb(n, cfg, channels):
        nonlocal built_timm
        name = cfg.get("backbone", backbone)
        if not isinstance(name, str):
            check_module_input(n, cfg, name, channels)
            return name
        built_timm = True
        return TimmBackbone(name, pretrained=cfg.get("pretrained", pretrained),
                            in_channels=channels,
                            channel_adapter=cfg.get("channel_adapter", channel_adapter),
                            out_indices=cfg.get("out_indices", out_indices),
                            **backbone_kwargs_with_rgb(n, cfg, backbone_kwargs))

    def check_top_level():  # top-level timm options with no timm backbone to apply them to
        if not built_timm:
            check_module_options(pretrained=pretrained, channel_adapter=channel_adapter,
                                 out_indices=out_indices, backbone_kwargs=backbone_kwargs)

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
        bb = make_bb(" + ".join(inputs), shared_cfg, total)
        check_top_level()
        return EarlyFusionEncoder(list(inputs), bb, channels), ftype, {}

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
            target_ch = getattr(encoders[target], "in_channels", None)
            if target_ch is None and target not in adapters:
                # a custom nn.Module backbone need not declare in_channels: it was built for
                # (and is fed) the target input as declared
                target_ch = inputs[target]["in_channels"]
            if target_ch is None:
                raise ValueError(f"input {n!r} shares the backbone of {target!r}, which does "
                                 "not declare 'in_channels' (a custom nn.Module backbone); "
                                 "set its in_channels attribute so the shared input's "
                                 "channels can be checked")
            _check_shared_adapter(n, cfg, target_ch)
            # Weights shared, BatchNorm per input by default: one BN shared by inputs with
            # different statistics blends its running statistics, and evaluation then applies
            # the blend to every input (Sen1Floods11 val mIoU 0.50-0.65 vs 0.89).
            encoders[n] = (encoders[target] if cfg.get("share_norm", False)
                           else share_weights(encoders[target]))
            if cfg.get("channel_adapter") == "projection":
                adapters[n] = ProjectionAdapter(cfg["in_channels"], target_ch)
        else:
            if "share_norm" in cfg:
                raise ValueError(f"input {n!r}: 'share_norm' only applies to a "
                                 "'shared:<input>' backbone")
            encoders[n] = make_bb(n, cfg, cfg["in_channels"])
            if n in series:
                kind, opts = parse_temporal(n, cfg.get("temporal", "mean"))
                encoders[n] = TemporalEncoder(encoders[n], kind, opts)
    check_top_level()
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
    return enc, ftype, {"op": op, "op_kwargs": op_kwargs}


def check_module_options(*, in_channels=None, module=None, pretrained=True,
                         channel_adapter="repeat", out_indices=None, backbone_kwargs=None) -> None:
    """A backbone given as an ``nn.Module`` is already built: options that configure a timm
    backbone would be silently ignored, so they are errors. ``pretrained=True`` (the default)
    and ``False`` are accepted (there is nothing to load); a weights tag is not.
    ``in_channels`` must equal the module's ``in_channels`` when it exposes one."""
    given = []
    if isinstance(pretrained, str):
        given.append(f"pretrained={pretrained!r}")
    if channel_adapter != "repeat":
        given.append(f"channel_adapter={channel_adapter!r}")
    if out_indices is not None:
        given.append(f"out_indices={out_indices!r}")
    if backbone_kwargs:
        given.append(f"backbone_kwargs={dict(backbone_kwargs)!r}")
    if given:
        raise ValueError(f"{', '.join(given)} configure a timm backbone, but backbone is an "
                         "nn.Module (already built), so they would be ignored; configure the "
                         "module itself, or pass a timm model name")
    if in_channels is not None and module is not None:
        if not isinstance(getattr(module, "in_channels", None), int):
            raise ValueError(f"in_channels={in_channels} can't change a backbone given as an "
                             f"nn.Module ({type(module).__name__} has no 'in_channels' to "
                             "check it against); leave it out")
        _check_module_channels(module, in_channels, f"in_channels={in_channels}: the input has")


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
    single head. Full reference: https://manyseg.readthedocs.io/en/latest/docs/api_design/.

    Args:
        backbone: A timm model name, or an ``nn.Module`` with ``feature_specs`` and
            ``input_divisor``. A module is used as built: the timm options (``pretrained``
            other than True/False, ``channel_adapter``, ``out_indices``, ``backbone_kwargs``,
            ``rgb_indices``) are then errors, and ``in_channels`` must equal the module's
            ``in_channels`` if given. With ``inputs=``, every input without its own
            ``backbone`` is encoded by this same module object (weights and BatchNorm
            shared, as with ``"shared:<input>"`` plus ``share_norm``), and each such input's
            ``in_channels`` must match the module's; give inputs separate module instances
            (``"backbone": module`` per input) for separate encoders.
        inputs: Several inputs, ``{name: config}`` (the first raster input is the primary).
            Raster or time-series configs: ``in_channels`` (required), ``kind``
            (``"raster"`` / ``"timeseries"``), ``backbone`` (or ``"shared:<input>"``),
            ``pretrained``, ``channel_adapter``, ``alignment`` (``"same_grid"`` /
            ``"same_extent"``), ``share_norm``, ``backbone_kwargs``, ``out_indices``,
            ``rgb_indices`` (wins over an ``rgb_indices`` in the top-level
            ``backbone_kwargs``), ``temporal`` (time series). Vector (metadata) configs:
            ``{"kind": "vector"}`` with ``fields`` or ``num_features``, ``dim``, ``dropout``,
            ``conditioning`` (``"film"`` / ``"head"`` / ``"channels"``).
        fusion: ``{"type": "early" | "intermediate" | "late", "op": ..., "op_kwargs": ...,
            "strides": [...]}`` for several raster inputs (``strides``: intermediate only).
        pretrained: ``True`` (timm weights), ``False``, or a ``manyseg.weights`` tag such as
            ``"ssl4eo_s2_dino"``.
        in_channels: Input bands of a single-input model (default 3; with ``inputs=``, set it
            per input instead).
        channel_adapter: How the RGB stem is adapted to ``in_channels``.
        out_indices: Encoder levels to return (timm ``out_indices``).
        backbone_kwargs: Passed to ``timm.create_model``, so the keywords depend on the
            architecture (``drop_path_rate``, for one, exists only where timm implements
            stochastic depth); also takes ``rgb_indices`` (input channels carrying R, G, B for
            the ``rgb_plus_*`` / ``projection`` adapters; with ``inputs=``, for every input,
            unless the input sets its own ``rgb_indices``).
        decoder: A decoder name, a decoder class or callable, or one dict
            ``{"type": name, **options}``.
        decoder_kwargs: Options for the decoder (not together with the same keys in a dict
            ``decoder``).
        num_classes: One segmentation head with this many output channels.
        heads: ``{name: config}`` with ``type`` (``"segmentation"``: pixel scope,
            ``"classification"``: image scope, ``"regression"``: either), ``out_channels``
            (required), ``scope`` (``"pixel"`` / ``"image"``; default ``"image"`` for
            classification, else ``"pixel"``), ``source``, ``scale``,
            ``uncertainty``, ``decoder`` (a per-head decoder), ``input`` and ``share``
            (per-input heads), and head options (pixel: ``kernel_size``, ``bias_init``,
            ``hidden``, ``dilation``, ``norm`` (``"batch"`` / ``"group"`` / None),
            ``activation``; image: ``dropout``,
            ``init_std``; see :class:`manyseg.heads.PixelHead`). A ``type`` registered with
            ``manyseg.heads.HEADS.register`` builds a custom head class.
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
        check_module_options(in_channels=in_channels, module=backbone, pretrained=pretrained,
                             channel_adapter=channel_adapter, out_indices=out_indices,
                             backbone_kwargs=backbone_kwargs)
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
                                     op=fopts["op"], alignment=[bb.alignment[n] for n in bb.names],
                                     op_kwargs=fopts["op_kwargs"])
        if wants:
            ch = getattr(bb, "in_channels", None)
            if not isinstance(ch, int):
                raise ValueError(f"decoder {dec_cls.__name__} reads the input image, so it "
                                 "needs the input's channel count, but the backbone module "
                                 f"{type(bb).__name__} has no 'in_channels' attribute; set "
                                 "one on it, or use a decoder that reads features only")
            specs = (FeatureSpec(ch, 1, "input"), *specs)
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
        if head_scope(heads[n]) != "pixel":
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
        if "decoder" in heads[n]:  # it reuses the other head's decoder: its own would be lost

            def _dec(cfg):
                d = cfg.get("decoder", {"type": decoder, **dict(decoder_kwargs or {})})
                return {"type": d} if isinstance(d, str) else (
                    dict(d) if isinstance(d, Mapping) else d)

            if _dec(heads[n]) != _dec(heads[other]):
                raise ValueError(f"head {n!r}: a head with 'share' uses the decoder of "
                                 f"{other!r}, so it cannot set a different 'decoder' (got "
                                 f"{heads[n]['decoder']!r}); remove it, or drop 'share'")
    reads_kwargs = any("decoder" not in heads[n] for n in head_inputs)
    for n in head_inputs:  # default: the model's decoder type and kwargs, on this input
        heads[n].setdefault("decoder", {"type": decoder, **dict(decoder_kwargs or {})})
    task_cfgs = {n: cfg.pop("decoder") for n, cfg in heads.items() if "decoder" in cfg}
    for n in task_cfgs:
        if head_scope(heads[n]) != "pixel":
            raise ValueError(f"head {n!r}: task-specific decoders are only for pixel heads")
    # pixel heads read the decoder by default, image heads only with "source": "decoder"
    needs_shared = any(n not in task_cfgs and heads[n].get(
        "source", "decoder" if head_scope(heads[n]) == "pixel" else "encoder")
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
