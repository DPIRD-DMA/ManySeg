"""Output heads. Heads return raw logits / values; activations belong to losses or users."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from ..features import FeatureSpec
from ..registry import Registry

HEADS = Registry("head type")


class PixelHead(nn.Module):
    """Per-pixel prediction: ``k x k`` conv then upsample by the feature stride.

    Output covers ``stride * feature_size`` pixels (>= padded input); the model crops it.
    """

    def __init__(self, spec: FeatureSpec, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.stride = spec.stride
        self.conv = nn.Conv2d(spec.channels, channels, kernel_size, padding=kernel_size // 2)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.stride > 1:
            h, w = x.shape[-2:]
            x = F.interpolate(
                x, size=(h * self.stride, w * self.stride), mode="bilinear", align_corners=False
            )
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
    if cfg.pop("uncertainty", False):
        # Heteroscedastic regression: [mean (C), log-variance (C)]; the loss (e.g. Gaussian NLL)
        # lives outside the model.
        if kind != "regression":
            raise ValueError("uncertainty=True is only supported for regression heads")
        channels = 2 * channels
    scope = cfg.pop("scope", "pixel")
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
    return head, source
