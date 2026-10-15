"""timm models as multi-scale feature extractors with a uniform NCHW contract."""

from __future__ import annotations

import difflib
import inspect
import math
from collections.abc import Sequence

import timm
import torch
from torch import Tensor, nn

from ..adapters import CHANNEL_ADAPTERS, ProjectionAdapter, adapt_conv_weight
from ..features import FeatureSpec
from .necks import ViTPyramidNeck, default_pyramid_channels

PYRAMID_STRIDES = (4, 8, 16, 32)


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


def introspect(name: str) -> dict:
    """Cheap architecture introspection: build on the meta device (no memory, no weights) and
    read the constructor signature and the full feature_info."""
    try:
        with torch.device("meta"):
            m = timm.create_model(name.split(".", 1)[0], pretrained=False)
    except Exception:  # noqa: BLE001 - fall back to a real (random-init) build
        m = timm.create_model(name.split(".", 1)[0], pretrained=False)
    params = inspect.signature(type(m).__init__).parameters
    info = list(getattr(m, "feature_info", []) or [])
    reductions = [f["reduction"] for f in info]
    return {
        "init_params": set(params),
        "feature_info": info,
        "plain": len(info) > 1 and len(set(reductions)) == 1,
        "embed_dim": getattr(m, "embed_dim", None) or getattr(m, "num_features", None),
    }


def _size_kwargs(init_params: set[str]) -> dict:
    """Enable timm's own dynamic-size support where the architecture offers it."""
    kw = {}
    if "dynamic_img_size" in init_params:
        kw["dynamic_img_size"] = True
    if "dynamic_img_pad" in init_params:
        kw["dynamic_img_pad"] = True
    if "strict_img_size" in init_params:
        kw["strict_img_size"] = False
    return kw


def _norm(name: str) -> str:
    return name.replace(".", "_")


def find_stem_convs(model: nn.Module, first_conv=None) -> list[tuple[str, nn.Conv2d]]:
    """Locate the convolution(s) that consume the image.

    ``pretrained_cfg['first_conv']`` may be a name or a list of names (multi-branch stems such
    as MobileOne); feature wrappers may rename ``a.b`` to ``a_b``. Without it, the first
    ``Conv2d`` with 3 input channels in module order is used.
    """
    candidates = [
        (n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d) and m.in_channels == 3
    ]
    if not candidates:
        raise ValueError("could not find a 3-channel stem convolution to adapt")
    wanted = [first_conv] if isinstance(first_conv, str) else list(first_conv or [])
    found = []
    for want in wanted:
        for n, m in candidates:
            if _norm(n) == _norm(want) or _norm(n).endswith(_norm(want)):
                found.append((n, m))
                break
    return found or [candidates[0]]


def find_stem_conv(model: nn.Module, first_conv=None) -> tuple[str, nn.Conv2d]:
    return find_stem_convs(model, first_conv)[0]


class TimmBackbone(nn.Module):
    """A timm model wrapped to return ``list[Tensor]`` (NCHW) with static ``feature_specs``.

    Args:
        name: timm model name, optionally with a pretrained tag (``"resnet50.a1_in1k"``).
        pretrained: Load timm pretrained weights.
        in_channels: Number of input channels.
        channel_adapter: How to adapt the RGB stem when ``in_channels != 3``; see
            :func:`manyseg.adapters.adapt_conv_weight` and :class:`ProjectionAdapter`.
        out_indices: Feature levels to return (timm ``out_indices``). Default: all.
        rgb_indices: Input channels carrying R, G, B for ``rgb_plus_*`` / ``projection``.
        **timm_kwargs: Forwarded to ``timm.create_model`` (e.g. ``drop_path_rate``).
    """

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        in_channels: int = 3,
        channel_adapter: str = "repeat",
        out_indices: Sequence[int] | None = None,
        rgb_indices: Sequence[int] | None = None,
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
        arch = introspect(name)
        self.is_plain = arch["plain"]
        kwargs = dict(features_only=True, **_size_kwargs(arch["init_params"]))
        kwargs.update(timm_kwargs)
        self.fixed_size: tuple[int, int] | None = None
        if "img_size" in timm_kwargs and not ({"dynamic_img_size", "strict_img_size"}
                                              & arch["init_params"]):
            size = timm_kwargs["img_size"]
            self.fixed_size = (size, size) if isinstance(size, int) else tuple(size)
        if self.is_plain and out_indices is None:
            n = len(arch["feature_info"])
            out_indices = tuple(n * (i + 1) // 4 - 1 for i in range(4))
        if out_indices is not None:
            kwargs["out_indices"] = tuple(out_indices)
        self.block_indices = tuple(out_indices) if self.is_plain else None
        self.model = _timm_create(name, pretrained=pretrained, **kwargs)
        self.name = name
        self.pretrained = bool(pretrained)  # recorded for describe(); weights are already set
        self.in_channels = in_channels
        self.channel_adapter = channel_adapter

        cfg = getattr(self.model, "pretrained_cfg", {}) or {}
        stems = find_stem_convs(self.model, cfg.get("first_conv"))
        self.stem_conv_names = [n for n, _ in stems]
        self._stem_name = self.stem_conv_names[0]
        self.input_adapter: nn.Module = nn.Identity()
        if in_channels != stems[0][1].in_channels:
            if channel_adapter == "projection":
                self.input_adapter = ProjectionAdapter(in_channels, stems[0][1].in_channels,
                                                       rgb_indices=rgb_indices)
            else:
                for _, stem in stems:
                    if stem.groups != 1:
                        raise ValueError("cannot adapt a grouped stem convolution")
                    new_w = adapt_conv_weight(stem.weight, in_channels, channel_adapter,
                                              rgb_indices=rgb_indices)
                    stem.weight = nn.Parameter(new_w.contiguous())
                    stem.in_channels = in_channels

        info = self.model.feature_info
        self.nhwc = str(getattr(self.model, "output_fmt", "NCHW")).upper().endswith("NHWC")
        specs = tuple(
            FeatureSpec(c, r, n)
            for c, r, n in zip(info.channels(), info.reduction(), info.module_name())
        )
        self.neck: ViTPyramidNeck | None = None
        if self.is_plain:
            k = len(specs)
            strides = PYRAMID_STRIDES[len(PYRAMID_STRIDES) - k:] if k <= 4 else None
            if strides is None:
                raise ValueError(f"plain ViT pyramid supports up to 4 blocks, got {k}")
            channels = default_pyramid_channels(specs[0].channels, k)
            self.neck = ViTPyramidNeck(specs[0].channels, strides, channels, specs[0].stride)
            specs = self.neck.feature_specs
        self.feature_specs: tuple[FeatureSpec, ...] = specs
        self.mean = tuple(cfg.get("mean", (0.485, 0.456, 0.406)))
        self.std = tuple(cfg.get("std", (0.229, 0.224, 0.225)))

    @property
    def input_divisor(self) -> int:
        """Inputs whose H and W are multiples of this give exactly aligned feature maps."""
        return math.lcm(*(s.stride for s in self.feature_specs))

    def stem_conv(self) -> nn.Conv2d:
        return self.model.get_submodule(self._stem_name)

    def forward(self, x: Tensor) -> list[Tensor]:
        h, w = x.shape[-2], x.shape[-1]
        if self.fixed_size is not None and (h, w) != self.fixed_size:
            raise ValueError(
                f"{self.name} was built for a fixed input size {self.fixed_size}; got {(h, w)}"
            )
        feats = self.model(self.input_adapter(x))
        if isinstance(feats, dict):
            feats = list(feats.values())
        feats = list(feats)
        if self.nhwc:
            feats = [f.permute(0, 3, 1, 2) for f in feats]
        if self.neck is not None:
            feats = self.neck(feats, h, w)
        return feats


def create_backbone(name: str, pretrained: bool = True, **kwargs) -> TimmBackbone:
    return TimmBackbone(name, pretrained=pretrained, **kwargs)


__all__ = ["TimmBackbone", "create_backbone", "find_stem_conv"]
