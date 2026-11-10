"""``create_model`` support for vector inputs: parse ``{"kind": "vector", ...}`` inputs, build
the raster model with the ordinary factory, then attach a :class:`Conditioner`."""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn

from .encoder import MetadataEncoder
from .inject import Conditioner, DecoderFiLM, FiLM, used_levels

VECTOR_KEYS = {"kind", "fields", "num_features", "dim", "dropout", "conditioning"}
MODES = ("film", "head", "channels")
"""How a vector input conditions the model: FiLM, a bias on the heads, or extra input
channels."""
_MODE_OPTIONS = {"film": {"on"}, "head": {"pixel"}, "channels": {"target"}}


def has_kinds(inputs: Mapping[str, Mapping]) -> bool:
    """Inputs that need the conditioning path: vector inputs, or explicit ``"raster"`` kinds
    (stripped there). Time series are rasters handled by ``create_model`` itself."""
    return any(cfg.get("kind", "timeseries") != "timeseries" for cfg in inputs.values())


def split_inputs(inputs: Mapping[str, Mapping]) -> tuple[dict, dict]:
    """``(rasters, vectors)``; ``"kind": "raster"`` is stripped from raster configs."""
    rasters, vectors = {}, {}
    for n, cfg in inputs.items():
        kind = cfg.get("kind", "raster")
        if kind == "vector":
            vectors[n] = dict(cfg)
        elif kind == "raster":
            rasters[n] = {k: v for k, v in cfg.items() if k != "kind"}
        elif kind == "timeseries":
            rasters[n] = dict(cfg)
        else:
            raise ValueError(f"input {n!r}: kind must be 'raster', 'timeseries' or 'vector', "
                             f"got {kind!r}")
    return rasters, vectors


def parse_vector(name: str, cfg: Mapping) -> tuple[Callable[[], MetadataEncoder], str, dict]:
    """``(encoder factory, mode, mode options)``. Encoders are constructed only after the
    raster model, so the raster weights do not depend on the metadata configuration."""
    unknown = set(cfg) - VECTOR_KEYS
    if unknown:
        raise ValueError(f"vector input {name!r}: unknown keys {sorted(unknown)}")
    cond = cfg.get("conditioning", "film")
    cond = {"mode": cond} if isinstance(cond, str) else dict(cond)
    mode = cond.pop("mode", "film")
    if mode == "token":
        raise NotImplementedError(
            "'token' conditioning (metadata as extra ViT tokens) is not implemented yet; "
            "use 'film', 'head' or 'channels'")
    if mode not in MODES:
        raise ValueError(f"vector input {name!r}: conditioning mode must be one of {MODES}, "
                         f"got {mode!r}")
    bad = set(cond) - _MODE_OPTIONS[mode]
    if bad:
        raise ValueError(f"vector input {name!r}: unknown {mode!r} options {sorted(bad)}")
    if mode == "channels" and "dim" in cfg:
        raise ValueError(f"vector input {name!r}: 'dim' does not apply to 'channels' mode "
                         "(the field encodings themselves become channels)")
    make = functools.partial(MetadataEncoder, cfg.get("fields"),
                             num_features=cfg.get("num_features"),
                             dim=None if mode == "channels" else cfg.get("dim", 64),
                             dropout=cfg.get("dropout", 0.0))
    with torch.random.fork_rng(devices=[]):
        make()  # validate the config now, without consuming the caller's RNG
    return make, mode, cond


def build_conditioned_model(create_model: Callable[..., nn.Module], *, inputs, fusion, backbone,
                            pretrained, channel_adapter, out_indices, backbone_kwargs,
                            heads, num_classes, **common: Any) -> nn.Module:
    from ..models.factory import backbone_kwargs_with_rgb, check_input_keys, check_module_input

    rasters, vectors = split_inputs(inputs)
    shared = dict(pretrained=pretrained, channel_adapter=channel_adapter,
                  out_indices=out_indices, heads=heads, num_classes=num_classes)
    if not vectors:
        return create_model(backbone=backbone, inputs=rasters, fusion=fusion,
                            backbone_kwargs=backbone_kwargs, **shared, **common)
    if not rasters:
        raise ValueError(f"vector inputs {sorted(vectors)} only condition a model; at least one "
                         "raster input is required")
    for n in vectors:
        if f"{n}_mask" in inputs:
            raise ValueError(f"input name {n}_mask is reserved for the mask of vector input {n!r}")
    for n, cfg in rasters.items():
        for suffix in ("mask", "dates"):
            if cfg.get("kind") == "timeseries" and f"{n}_{suffix}" in inputs:
                raise ValueError(f"input name {n}_{suffix} is reserved for the {suffix} of "
                                 f"time series {n!r}")

    makers, modes, opts, targets = {}, {}, {}, {}
    for n, cfg in vectors.items():
        makers[n], modes[n], opts[n] = parse_vector(n, cfg)
        if modes[n] == "channels":
            t = opts[n].get("target", next(iter(rasters)))
            if t not in rasters:
                raise ValueError(f"vector input {n!r}: channels target {t!r} is not a raster "
                                 f"input; choose from {list(rasters)}")
            check_input_keys(t, rasters[t])
            if rasters[t].get("kind") == "timeseries":
                raise ValueError(f"vector input {n!r}: 'channels' conditioning cannot target the "
                                 f"time series {t!r}; use 'film' or 'head'")
            with torch.random.fork_rng(devices=[]):
                width = makers[n]().width
            rasters[t] = {**rasters[t], "in_channels": rasters[t]["in_channels"] + width}
            targets[n] = t

    single = None
    if len(rasters) == 1 and fusion is None and \
            next(iter(rasters.values())).get("kind") != "timeseries":
        # One raster: build the ordinary single-input model (plain-tensor backbone), so every
        # decoder (including those that need the input image) and adapter works unchanged.
        (single, cfg), = rasters.items()
        check_input_keys(single, cfg)
        bb = cfg.get("backbone", backbone)
        channels = cfg["in_channels"]
        if not isinstance(bb, str):  # a built module: no timm options; channels must match
            check_module_input(single, cfg, bb, channels)
            channels = None
        if heads is not None:
            shared["heads"], common = _single_raster_heads(heads, single, vectors, common)
        model = create_model(
            backbone=bb, in_channels=channels,
            backbone_kwargs=backbone_kwargs_with_rgb(single, cfg, backbone_kwargs),
            **{**shared, **common,
               **{k: cfg[k] for k in ("pretrained", "channel_adapter", "out_indices")
                  if k in cfg}})
    else:
        _check_head_inputs(heads, vectors)
        model = create_model(backbone=backbone, inputs=rasters, fusion=fusion,
                             backbone_kwargs=backbone_kwargs, **shared, **common)

    encoders = {n: make() for n, make in makers.items()}
    film_names = [n for n in vectors if modes[n] == "film"]
    head_names = [n for n in vectors if modes[n] == "head"]
    film = {}
    if film_names:
        ons = {tuple(opts[n].get("on", ("features", "output"))) for n in film_names}
        if len(ons) > 1:
            raise ValueError("all 'film' vector inputs must use the same 'on' option")
        on = ons.pop()
        dim = sum(encoders[n].dim for n in film_names)
        specs = tuple(model.backbone.feature_specs)
        in_ch = getattr(model.backbone, "in_channels", None)
        decoders = {"decoder": model.decoder} if model.decoder is not None else {}
        decoders.update({f"task_{h}": d for h, d in model.task_decoders.items()})
        by_module = {}  # shared per-input decoders share their FiLM too
        for k, d in decoders.items():
            if id(d) in by_module:
                film[k] = by_module[id(d)]
                continue
            inp = model._head_inputs.get(k.removeprefix("task_")) if k.startswith("task_") \
                else None
            sp = tuple(model.backbone.encoders[inp].feature_specs) if inp else specs
            film[k] = by_module[id(d)] = DecoderFiLM(dim, sp, used_levels(d, sp, in_ch),
                                                    d.out_spec, on)
        # Heads that read the encoder directly: FiLM on the deepest feature (for image heads, a
        # per-channel affine before the mean pool, i.e. on the pooled vector).
        if any(model._head_sources[n] == "encoder" for n in model.heads):
            film["image"] = FiLM(dim, specs[model._deepest].channels)

    head_bias = {}
    if head_names:
        dim = sum(encoders[n].dim for n in head_names)
        pixel = any(opts[n].get("pixel", False) for n in head_names)
        cfgs = heads if heads is not None else {"logits": {"out_channels": num_classes}}
        bias_of = {}  # shared heads share their bias layer
        for name, head in model.heads.items():
            scope = getattr(head, "scope", "pixel")
            if id(head) in bias_of:
                head_bias[name] = bias_of[id(head)]
                continue
            if scope == "image" or (scope == "pixel" and pixel):
                cfg = cfgs[name]
                out = cfg["out_channels"] * (2 if cfg.get("uncertainty", False) else 1)
                head_bias[name] = bias_of[id(head)] = nn.Linear(dim, out)
                nn.init.zeros_(head_bias[name].weight)
                nn.init.zeros_(head_bias[name].bias)
        if not head_bias:
            raise ValueError("'head' conditioning: the model has no image-scope heads; to "
                             "condition pixel heads, give the vector input "
                             "conditioning={'mode': 'head', 'pixel': True}")

    model.conditioner = Conditioner(encoders, modes, list(rasters), single=single,
                                    temporal=[n for n, c in rasters.items()
                                              if c.get("kind") == "timeseries"],
                                    channel_targets=targets, film=film, head_bias=head_bias)
    return model


def _check_head_inputs(heads, vectors) -> None:
    """A head's ``"input"`` names a raster input, never a vector one."""
    for n, cfg in (heads or {}).items():
        if cfg.get("input") in vectors:
            raise ValueError(f"head {n!r}: 'input' {cfg['input']!r} is a vector (metadata) "
                             "input; 'input' names the raster input whose features the head "
                             "decodes")


def _single_raster_heads(heads, single: str, vectors, common: dict) -> tuple[dict, dict]:
    """``(heads, common)`` for a model with one raster input (built as a single-input model):
    a head's ``"input": <that raster>`` is unambiguous, so it becomes what a per-input head
    is there, a head with its own decoder (default: the model's ``decoder`` type and
    ``decoder_kwargs``, which move into it when no other head reads the shared decoder)."""
    from ..heads.heads import head_scope

    _check_head_inputs(heads, vectors)
    out = {n: dict(cfg) for n, cfg in heads.items()}
    moved = []
    for n, cfg in out.items():
        if "input" not in cfg:
            continue
        if cfg["input"] != single:
            raise ValueError(f"head {n!r}: unknown input {cfg['input']!r}; the raster input is "
                             f"{single!r}")
        if "share" in cfg:
            raise ValueError(f"head {n!r}: 'share' links per-input heads on different raster "
                             f"inputs; this model has one raster input, {single!r}")
        del cfg["input"]
        if cfg.get("source") == "encoder":
            raise ValueError(f"head {n!r}: 'source': 'encoder' reads the encoder directly, so it "
                             "cannot also have its own decoder ('decoder' / 'input')")
        if head_scope(cfg) != "pixel":
            raise ValueError(f"head {n!r}: 'input' heads are pixel heads")
        if "decoder" not in cfg:
            cfg["decoder"] = {"type": common.get("decoder", "unet"),
                              **dict(common.get("decoder_kwargs") or {})}
            moved.append(n)
    reads_shared = any("decoder" not in cfg and cfg.get(
        "source", "decoder" if head_scope(cfg) == "pixel" else "encoder") == "decoder"
        for cfg in out.values())
    if moved and not reads_shared:  # the kwargs are in the heads' decoders now
        common = {**common, "decoder_kwargs": None}
    return out, common
