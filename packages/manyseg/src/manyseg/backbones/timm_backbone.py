"""timm models as multi-scale feature extractors with a uniform NCHW contract."""

from __future__ import annotations

import difflib
import math
from collections.abc import Sequence

import timm
from torch import Tensor, nn

from ..adapters import CHANNEL_ADAPTERS, ProjectionAdapter, adapt_conv_weight
from ..features import FeatureSpec


def _timm_create(name: str, pretrained: bool = False, **kwargs) -> nn.Module:
    # Indirection point so tests can substitute weights without network access.
    return timm.create_model(name, pretrained=pretrained, **kwargs)


def _check_name(name: str) -> None:
    arch = name.split(".", 1)[0]
    if arch in timm.list_models():
        return
    close = difflib.get_close_matches(arch, timm.list_models(), n=5)
    hint = f"; did you mean {', '.join(close)}?" if close else ""
    raise ValueError(f"unknown timm backbone {name!r}{hint}")


def _norm(name: str) -> str:
    return name.replace(".", "_")


def find_stem_conv(model: nn.Module, first_conv: str | None = None) -> tuple[str, nn.Conv2d]:
    """Locate the convolution that consumes the image.

    Prefers ``pretrained_cfg['first_conv']`` (feature wrappers may rename ``a.b`` to ``a_b``);
    otherwise the first ``Conv2d`` with 3 input channels in module order.
    """
    candidates = [
        (n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d) and m.in_channels == 3
    ]
    if first_conv:
        for n, m in candidates:
            if _norm(n) == _norm(first_conv) or _norm(n).endswith(_norm(first_conv)):
                return n, m
    if not candidates:
        raise ValueError("could not find a 3-channel stem convolution to adapt")
    return candidates[0]


class TimmBackbone(nn.Module):
    """A timm model wrapped to return ``list[Tensor]`` (NCHW) with static ``feature_specs``.

    Args:
        name: timm model name, optionally with a pretrained tag (``"resnet50.a1_in1k"``).
        pretrained: Load timm pretrained weights.
        in_channels: Number of input channels.
        channel_adapter: How to adapt the RGB stem when ``in_channels != 3``; see
            :func:`manyseg.adapters.adapt_conv_weight` and :class:`ProjectionAdapter`.
        out_indices: Feature levels to return (timm ``out_indices``). Default: all.
        **timm_kwargs: Forwarded to ``timm.create_model`` (e.g. ``drop_path_rate``).
    """

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        in_channels: int = 3,
        channel_adapter: str = "repeat",
        out_indices: Sequence[int] | None = None,
        **timm_kwargs,
    ) -> None:
        super().__init__()
        _check_name(name)
        if channel_adapter not in CHANNEL_ADAPTERS:
            raise ValueError(
                f"unknown channel_adapter {channel_adapter!r}; choose from {CHANNEL_ADAPTERS}"
            )
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        kwargs = dict(features_only=True, **timm_kwargs)
        if out_indices is not None:
            kwargs["out_indices"] = tuple(out_indices)
        self.model = _timm_create(name, pretrained=pretrained, **kwargs)
        self.name = name
        self.in_channels = in_channels
        self.channel_adapter = channel_adapter

        cfg = getattr(self.model, "pretrained_cfg", {}) or {}
        self._stem_name, stem = find_stem_conv(self.model, cfg.get("first_conv"))
        self.input_adapter: nn.Module = nn.Identity()
        if in_channels != stem.in_channels:
            if channel_adapter == "projection":
                self.input_adapter = ProjectionAdapter(in_channels, stem.in_channels)
            else:
                if stem.groups != 1:
                    raise ValueError("cannot adapt a grouped stem convolution")
                new_w = adapt_conv_weight(stem.weight, in_channels, channel_adapter)
                stem.weight = nn.Parameter(new_w.contiguous())
                stem.in_channels = in_channels

        info = self.model.feature_info
        self.feature_specs: tuple[FeatureSpec, ...] = tuple(
            FeatureSpec(c, r, n)
            for c, r, n in zip(info.channels(), info.reduction(), info.module_name())
        )
        self.mean = tuple(cfg.get("mean", (0.485, 0.456, 0.406)))
        self.std = tuple(cfg.get("std", (0.229, 0.224, 0.225)))

    @property
    def input_divisor(self) -> int:
        """Inputs whose H and W are multiples of this give exactly aligned feature maps."""
        return math.lcm(*(s.stride for s in self.feature_specs))

    def stem_conv(self) -> nn.Conv2d:
        return self.model.get_submodule(self._stem_name)

    def forward(self, x: Tensor) -> list[Tensor]:
        feats = self.model(self.input_adapter(x))
        if isinstance(feats, dict):
            feats = list(feats.values())
        return list(feats)


def create_backbone(name: str, pretrained: bool = True, **kwargs) -> TimmBackbone:
    return TimmBackbone(name, pretrained=pretrained, **kwargs)


__all__ = ["TimmBackbone", "create_backbone", "find_stem_conv"]
