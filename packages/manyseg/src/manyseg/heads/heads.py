"""Output heads. Heads return raw logits / values; activations belong to losses or users."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from ..features import FeatureSpec
from ..registry import Registry
from ..utils.resolution import scale_factor
from ..utils.spatial import pad_to_multiple

HEADS = Registry("head type")


class PixelHead(nn.Module):
    """Per-pixel prediction on a grid of ``scale`` x the input resolution.

    ``f = 1 / scale`` is the output stride (1, 2, 4 or 8). From a feature of stride ``d``:

    * ``d == f``: just the ``k x k`` conv; nothing is resampled (a decoder built with
      ``out_stride=f`` feeding a head with ``scale=1/f`` pays for no upsampling at all);
    * ``d > f``: conv, then bilinear upsampling by ``d / f`` (``align_corners=False``);
    * ``d < f``: area-average by ``f / d`` (block means), then the conv on the coarse grid.

    The output covers ``feature_size * d / f`` pixels (>= the scaled padded input); the model
    crops it to ``ceil(H / f) x ceil(W / f)``.
    """

    def __init__(self, spec: FeatureSpec, channels: int, kernel_size: int = 3,
                 bias_init: float = 0.0, scale: float = 1) -> None:
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
        self.conv = nn.Conv2d(spec.channels, channels, kernel_size, padding=kernel_size // 2)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.constant_(self.conv.bias, bias_init)  # e.g. logit(prior) for focal-loss heads

    def forward(self, x: Tensor) -> Tensor:
        if self.down > 1:
            x = F.avg_pool2d(pad_to_multiple(x, self.down), self.down)
        x = self.conv(x)
        if self.up > 1:
            h, w = x.shape[-2:]
            x = F.interpolate(x, size=(h * self.up, w * self.up), mode="bilinear",
                              align_corners=False)
        return x


class ImageHead(nn.Module):
    """Image-level prediction: global average pool, optional dropout, linear."""

    def __init__(self, spec: FeatureSpec, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(spec.channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(self.drop(x.mean(dim=(-2, -1))))


for _t in ("segmentation", "regression", "classification"):
    HEADS[_t] = _t


def create_head(spec_decoder: FeatureSpec, spec_encoder: FeatureSpec, cfg: dict) -> tuple[
    nn.Module, str
]:
    """Build a head from a config dict; returns (module, source) with source in
    {"decoder", "encoder"}. Image-scope heads default to the deepest encoder feature."""
    cfg = dict(cfg)
    kind = HEADS.get_or_raise(cfg.pop("type", "segmentation"))
    channels = cfg.pop("channels")
    uncertainty = bool(cfg.pop("uncertainty", False))
    if uncertainty:
        # Heteroscedastic regression: [mean (C), log-variance (C)]; the loss (e.g. Gaussian NLL)
        # lives outside the model.
        if kind != "regression":
            raise ValueError("uncertainty=True is only supported for regression heads")
        channels = 2 * channels
    scope = cfg.pop("scope", "pixel")
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
