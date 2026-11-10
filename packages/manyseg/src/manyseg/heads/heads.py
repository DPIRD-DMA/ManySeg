"""Output heads. Heads return raw logits / values; activations belong to losses or users, except
the opt-in output ``activation`` of regression pixel heads (``"softplus"``: a density map that
is positive by construction).

Custom head classes register under a new ``type`` name::

    @HEADS.register("my_head")
    class MyHead(nn.Module):
        def __init__(self, spec: FeatureSpec, channels: int, **opts): ...

They are built as ``cls(spec, out_channels, **opts)``, where ``spec`` is the feature the head
reads (``spec.channels`` at stride ``spec.stride``) and ``opts`` are the head config's other
keys (everything but ``type``, ``out_channels``, ``scope`` and ``source``). A pixel-scope head
must return ``(B, out_channels, H', W')`` covering at least the padded input (``stride`` times
the feature size); the model crops it to the input. To predict on a coarser grid, set a
``factor`` attribute (``1 / scale``) and return at least ``ceil(H / factor)`` rows and columns.
Image-scope heads return ``(B, out_channels)``. ``manyseg.save`` needs an ``activations``
entry for custom heads, and ``manyseg.load`` needs the class registered (imported) first.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch.nn.functional as F
from torch import Tensor, nn

from ..features import FeatureSpec
from ..layers import GroupNorm
from ..registry import Registry
from ..utils.resolution import scale_factor
from ..utils.spatial import pad_to_multiple

HEADS = Registry("head type")
"""Head constructors by head ``"type"``; ``HEADS.register("name")`` adds one."""


class PixelHead(nn.Module):
    """Per-pixel prediction on a grid of ``scale`` x the input resolution.

    ``f = 1 / scale`` is the output stride (1, 2, 4 or 8). From a feature of stride ``d``:

    * ``d == f``: just the ``k x k`` conv; nothing is resampled (a decoder built with
      ``out_stride=f`` feeding a head with ``scale=1/f`` pays for no upsampling at all);
    * ``d > f``: conv, then bilinear upsampling by ``d / f`` (``align_corners=False``);
    * ``d < f``: area-average by ``f / d`` (block means), then the conv on the coarse grid.

    The output covers ``feature_size * d / f`` pixels (>= the scaled padded input); the model
    crops it to ``ceil(H / f) x ceil(W / f)``.

    Optional, all off by default (the default head is the single conv, as in SMP):

    * ``hidden``: widths of ``k x k`` conv -> ``norm`` -> ReLU layers before the final conv,
      e.g. ``[64, 32, 16]``; ``dilation`` (one int, or one per layer, e.g. ``[8, 16, 32]``)
      widens their receptive field to match large objects; ``norm`` is ``"bn"`` (default),
      ``"gn"`` or None. They run on the head's grid, before upsampling.
    * ``activation="softplus"`` (regression heads): the output is ``softplus(conv)``, > 0
      (never exactly 0). Applied after resampling. ``bias_init`` stays the bias of the final
      conv, before the activation. (ReLU was tried for density heads and removed: the head
      died, see :func:`manyseg.points.density_head`.)
    """

    def __init__(self, spec: FeatureSpec, channels: int, kernel_size: int = 3,
                 bias_init: float = 0.0, scale: float = 1, *,
                 hidden: Sequence[int] | None = None, dilation: int | Sequence[int] = 1,
                 norm: str | None = "bn", activation: str | None = None) -> None:
        super().__init__()
        self.stride = spec.stride
        self.scale = scale
        self.factor = scale_factor(scale)
        self.up = max(1, self.stride // self.factor)
        self.down = max(1, self.factor // self.stride)
        if self.up * self.factor != max(self.stride, self.factor) or \
                self.down * self.stride != max(self.stride, self.factor):
            raise ValueError(f"feature stride {self.stride} and head scale {scale} are not "
                             "related by a power of two")
        in_ch = spec.channels
        self.hidden = None
        if hidden is not None:
            self.hidden = _hidden_layers(in_ch, hidden, dilation, norm, kernel_size)
            in_ch = list(hidden)[-1]
        elif dilation != 1:
            raise ValueError("'dilation' applies to the 'hidden' layers; give 'hidden' too")
        if activation not in ACTIVATIONS:
            raise ValueError(f"head activation must be one of {ACTIVATIONS}, got {activation!r}")
        self.activation = activation
        self.conv = nn.Conv2d(in_ch, channels, kernel_size, padding=kernel_size // 2)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.constant_(self.conv.bias, bias_init)  # e.g. logit(prior) for focal-loss heads

    def forward(self, x: Tensor, activate: bool = True) -> Tensor:
        """``activate=False``: skip the output activation (the model applies :meth:`activate`
        itself after a metadata head bias, so the final output stays >= 0)."""
        if self.down > 1:
            x = F.avg_pool2d(pad_to_multiple(x, self.down), self.down)
        if self.hidden is not None:
            x = self.hidden(x)
        x = self.conv(x)
        if self.up > 1:
            h, w = x.shape[-2:]
            x = F.interpolate(x, size=(h * self.up, w * self.up), mode="bilinear",
                              align_corners=False)
        return self.activate(x) if activate else x

    def activate(self, x: Tensor) -> Tensor:
        """The output activation (after resampling: bilinear weights are convex)."""
        return F.softplus(x) if self.activation == "softplus" else x

    def extra_repr(self) -> str:
        return f"activation={self.activation!r}" if self.activation else ""


ACTIVATIONS = (None, "softplus")
NORMS = (None, "bn", "gn")


def _hidden_layers(in_ch: int, hidden: Sequence[int], dilation: int | Sequence[int],
                   norm: str | None, kernel_size: int) -> nn.Sequential:
    """``k x k`` conv -> norm -> ReLU per hidden width; ``dilation`` per layer (padding keeps
    the size, so any input size works)."""
    if isinstance(hidden, (str, int)) or not hidden or not all(
            isinstance(c, int) and not isinstance(c, bool) and c > 0 for c in hidden):
        raise ValueError(f"'hidden' must be a non-empty list of positive channel counts, got "
                         f"{hidden!r} (leave it out for the single-conv head)")
    hidden = list(hidden)
    dil = [dilation] * len(hidden) if isinstance(dilation, int) else list(dilation)
    if len(dil) != len(hidden) or not all(
            isinstance(d, int) and not isinstance(d, bool) and d > 0 for d in dil):
        raise ValueError(f"'dilation' must be a positive int or one per hidden layer "
                         f"({len(hidden)}), got {dilation!r}")
    if norm not in NORMS:
        raise ValueError(f"head norm must be one of {NORMS}, got {norm!r}")
    layers: list[nn.Module] = []
    for c, d in zip(hidden, dil):
        layers.append(nn.Conv2d(in_ch, c, kernel_size, padding=d * (kernel_size // 2),
                                dilation=d, bias=norm is None))
        if norm == "bn":
            layers.append(nn.BatchNorm2d(c))
        elif norm == "gn":
            layers.append(GroupNorm(math.gcd(32, c), c))
        layers.append(nn.ReLU(inplace=True))
        in_ch = c
    return nn.Sequential(*layers)


class ImageHead(nn.Module):
    """Image-level prediction: global average pool, optional dropout, linear.

    ``init_std``: normal linear weights (timm's ``trunc_normal_``, bounds ±2) and zero bias
    instead of PyTorch's default Linear init."""

    def __init__(self, spec: FeatureSpec, channels: int, dropout: float = 0.0,
                 init_std: float | None = None) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(spec.channels, channels)
        if init_std is not None:  # timm's classifier init for ViT / ConvNeXt / PVT families
            nn.init.trunc_normal_(self.fc.weight, std=init_std)  # timm: bounds ±2 absolute
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(self.drop(x.mean(dim=(-2, -1))))


for _t in ("segmentation", "regression", "classification"):
    HEADS[_t] = _t


def create_head(spec_decoder: FeatureSpec, spec_encoder: FeatureSpec, cfg: dict) -> tuple[
    nn.Module, str
]:
    """Build a head from a config dict; returns (module, source) with source in
    {"decoder", "encoder"}. Image-scope heads default to the deepest encoder feature.

    ``cfg``: ``type`` (``"segmentation"`` for pixel heads, ``"classification"`` for image
    heads, ``"regression"`` for either), ``out_channels`` (required), ``scope``, ``source``,
    ``uncertainty`` (regression: ``2 * out_channels`` outputs, mean and log-variance),
    ``scale`` (pixel heads) and the head module's options (:class:`PixelHead`: ``kernel_size``,
    ``bias_init``, ``hidden``, ``dilation``, ``norm``, ``activation``; :class:`ImageHead`:
    ``dropout``, ``init_std``). A ``type`` registered in :data:`HEADS` as a class builds that
    class (see the module docstring)."""
    cfg = dict(cfg)
    if "out_channels" not in cfg:
        raise ValueError("a head needs 'out_channels'")
    kind_name = cfg.pop("type", "segmentation")
    kind = HEADS.get_or_raise(kind_name)
    channels = cfg.pop("out_channels")
    if not isinstance(kind, str):  # a registered head class
        return _custom_head(kind, kind_name, channels, cfg, spec_decoder, spec_encoder)
    for k in ("dilation", "norm"):
        if k in cfg and "hidden" not in cfg:
            raise ValueError(f"head {k!r} applies to the 'hidden' layers; give 'hidden' too")
    if cfg.get("activation") is not None and kind != "regression":
        raise ValueError("an output 'activation' is only for regression heads (segmentation and "
                         "classification heads return logits for the loss)")
    uncertainty = cfg.pop("uncertainty", False)
    if not isinstance(uncertainty, bool):
        raise ValueError(f"'uncertainty' must be True or False, got {uncertainty!r}")
    if uncertainty:
        # Heteroscedastic regression: [mean (C), log-variance (C)]; the loss (e.g. Gaussian NLL)
        # lives outside the model.
        if kind != "regression":
            raise ValueError("uncertainty=True is only supported for regression heads")
        if cfg.get("activation") is not None:
            raise ValueError("'activation' with uncertainty=True would also apply to the "
                             "log-variance; leave it out")
        channels = 2 * channels
    scope = cfg.pop("scope", "pixel")
    # segmentation predicts per pixel, classification per image; regression is either
    if (kind_name, scope) in (("segmentation", "image"), ("classification", "pixel")):
        right = "classification" if scope == "image" else "segmentation"
        other = "pixel" if scope == "image" else "image"
        raise ValueError(f"{kind_name!r} heads are {other}-scope: set 'scope': {other!r}, or use "
                         f"'type': {right!r} for a {scope}-scope head (scope defaults to "
                         "'pixel')")
    if scope != "pixel":
        extra = sorted(set(cfg) & {"hidden", "dilation", "norm", "activation", "kernel_size",
                                   "bias_init"})
        if extra:
            raise ValueError(f"{extra} apply to pixel heads only")
    if scope != "pixel" and "scale" in cfg:
        raise ValueError("'scale' applies to pixel heads only; image-scope heads have no grid")
    if scope == "pixel":
        source = cfg.pop("source", "decoder")
        spec = spec_decoder if source == "decoder" else spec_encoder
        head: nn.Module = PixelHead(spec, channels, **cfg)
    elif scope == "image":
        source = cfg.pop("source", "encoder")
        spec = spec_decoder if source == "decoder" else spec_encoder
        head = ImageHead(spec, channels, **cfg)
    else:
        raise ValueError(f"scope must be 'pixel' or 'image', got {scope!r}")
    if source not in ("decoder", "encoder"):
        raise ValueError(f"source must be 'decoder' or 'encoder', got {source!r}")
    head.kind = kind
    head.scope = scope
    head.uncertainty = uncertainty
    return head, source


def _custom_head(cls, kind_name: str, channels: int, cfg: dict, spec_decoder: FeatureSpec,
                 spec_encoder: FeatureSpec) -> tuple[nn.Module, str]:
    if cfg.pop("uncertainty", False):
        raise ValueError(f"uncertainty=True is for the built-in regression head, not "
                         f"{kind_name!r}")
    scope = cfg.pop("scope", "pixel")
    if scope not in ("pixel", "image"):
        raise ValueError(f"scope must be 'pixel' or 'image', got {scope!r}")
    source = cfg.pop("source", "decoder" if scope == "pixel" else "encoder")
    if source not in ("decoder", "encoder"):
        raise ValueError(f"source must be 'decoder' or 'encoder', got {source!r}")
    head = cls(spec_decoder if source == "decoder" else spec_encoder, channels, **cfg)
    if not isinstance(head, nn.Module):
        raise TypeError(f"head type {kind_name!r} built a {type(head).__name__}, not an "
                        "nn.Module")
    head.kind = kind_name
    head.scope = scope
    head.uncertainty = False
    return head, source
