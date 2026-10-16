"""Time-series inputs: one 2D encoder applied to every date, then a temporal aggregation at each
feature level, so the rest of the model (fusion, decoders, heads) sees an ordinary pyramid.

An input ``{"kind": "timeseries", "channels": C, "temporal": ...}`` takes ``(B, T, C, H, W)``
with any ``T``, plus two optional entries in the input dict:

* ``<name>_mask``: ``(B, T)`` bool, False for missing / cloudy / padded dates;
* ``<name>_dates``: ``(B, T)`` float acquisition dates in days (e.g. day of year, or days since a
  reference date); only ``"attention"`` uses them. Without dates, the position ``0..T-1`` is
  used.

Aggregations (``"temporal"``: a name or ``{"type": name, **options}``):

* ``"mean"`` (default): masked mean over the valid dates;
* ``"max"``: masked per-channel max;
* ``"attention"``: lightweight temporal attention per pixel, in the spirit of L-TAE (Garnot
  & Landrieu 2020) as used by U-TAE: each of ``heads`` channel groups gets softmax weights over
  the dates from a learned query against keys computed from the features and a sinusoidal
  encoding of the dates. The query starts at zero, so at initialisation the weights are uniform
  and the model is exactly the masked mean. Options: ``heads`` (default 4, reduced to a divisor
  of each level's channels), ``key_dim`` (16), ``date_dim`` (32), ``max_period`` (10000 days).

A sample whose dates are all masked yields zero features (no NaNs).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn

AGGREGATIONS = ("mean", "max", "attention")


def parse_temporal(name: str, cfg) -> tuple[str, dict]:
    cfg = {"type": cfg} if isinstance(cfg, str) else dict(cfg or {"type": "mean"})
    kind = cfg.pop("type", "mean")
    if kind not in AGGREGATIONS:
        raise ValueError(f"input {name!r}: temporal aggregation must be one of {AGGREGATIONS}, "
                         f"got {kind!r}")
    allowed = {"heads", "key_dim", "date_dim", "max_period"} if kind == "attention" else set()
    bad = set(cfg) - allowed
    if bad:
        raise ValueError(f"input {name!r}: unknown temporal options {sorted(bad)} for {kind!r}")
    return kind, cfg


def _valid(mask: Tensor | None, b: int, t: int, device) -> Tensor:
    if mask is None:
        return torch.ones(b, t, dtype=torch.bool, device=device)
    return mask.to(device=device, dtype=torch.bool)


class MeanAggregation(nn.Module):
    def forward(self, f: Tensor, mask: Tensor, dates: Tensor | None = None) -> Tensor:
        w = mask.to(f.dtype)[:, :, None, None, None]  # B T 1 1 1
        return (f * w).sum(1) / w.sum(1).clamp(min=1)


class MaxAggregation(nn.Module):
    def forward(self, f: Tensor, mask: Tensor, dates: Tensor | None = None) -> Tensor:
        m = mask[:, :, None, None, None]
        out = f.masked_fill(~m, float("-inf")).amax(1)
        return torch.where(mask.any(1)[:, None, None, None], out, torch.zeros_like(out))


def date_encoding(dates: Tensor, dim: int, max_period: float) -> Tensor:
    """Sinusoidal encoding of ``(B, T)`` dates in days -> ``(B, T, dim)``."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=dates.device,
                                                           dtype=torch.float32) / half)
    ang = dates.float()[..., None] * freqs
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class AttentionAggregation(nn.Module):
    def __init__(self, channels: int, heads: int = 4, key_dim: int = 16, date_dim: int = 32,
                 max_period: float = 10000.0) -> None:
        super().__init__()
        self.heads = math.gcd(channels, max(1, int(heads)))
        self.key_dim, self.date_dim, self.max_period = key_dim, date_dim, float(max_period)
        self.key = nn.Conv2d(channels, self.heads * key_dim, 1)
        self.date_key = nn.Linear(date_dim, self.heads * key_dim)
        self.query = nn.Parameter(torch.zeros(self.heads, key_dim))  # uniform weights at init

    def forward(self, f: Tensor, mask: Tensor, dates: Tensor | None = None) -> Tensor:
        b, t, c, h, w = f.shape
        if dates is None:
            dates = torch.arange(t, device=f.device, dtype=torch.float32).expand(b, t)
        k = self.key(f.flatten(0, 1)).view(b, t, self.heads, self.key_dim, h, w)
        pe = self.date_key(date_encoding(dates, self.date_dim, self.max_period))
        k = k + pe.view(b, t, self.heads, self.key_dim, 1, 1).to(k.dtype)
        scores = (k * self.query.to(k.dtype)[None, None, :, :, None, None]).sum(3)
        scores = scores / math.sqrt(self.key_dim)  # B T heads h w
        scores = scores.float().masked_fill(~mask[:, :, None, None, None], float("-inf"))
        any_valid = mask.any(1)[:, None, None, None, None]
        weights = torch.where(any_valid, scores, torch.zeros_like(scores)).softmax(1)
        weights = torch.where(any_valid, weights, torch.zeros_like(weights)).to(f.dtype)
        g = f.view(b, t, self.heads, c // self.heads, h, w)
        return (g * weights[:, :, :, None]).sum(1).reshape(b, c, h, w)


class TemporalEncoder(nn.Module):
    """A backbone applied to every date of a ``(B, T, C, H, W)`` series, aggregated over time
    per feature level. Behaves like a backbone (``feature_specs``, ``input_divisor``) for
    :class:`~manyseg.fusion.MultiEncoder`, which passes the input's mask and dates."""

    temporal = True

    def __init__(self, backbone: nn.Module, kind: str = "mean",
                 options: Mapping | None = None) -> None:
        super().__init__()
        self.backbone = backbone
        self.kind = kind
        self.feature_specs = backbone.feature_specs
        self.input_divisor = getattr(backbone, "input_divisor", 1)
        self.in_channels = getattr(backbone, "in_channels", None)
        self.name = getattr(backbone, "name", type(backbone).__name__)
        opts = dict(options or {})
        if kind == "mean":
            aggs = [MeanAggregation() for _ in self.feature_specs]
        elif kind == "max":
            aggs = [MaxAggregation() for _ in self.feature_specs]
        else:
            aggs = [AttentionAggregation(s.channels, **opts) for s in self.feature_specs]
        self.temporal_aggregation = nn.ModuleList(aggs)

    def forward(self, x: Tensor, mask: Tensor | None = None,
                dates: Tensor | None = None) -> list[Tensor]:
        if x.ndim != 5:
            raise ValueError(f"time-series input must be (B, T, C, H, W), got shape "
                             f"{tuple(x.shape)}")
        b, t = x.shape[:2]
        valid = _valid(mask, b, t, x.device)
        feats = self.backbone(x.flatten(0, 1))
        return [agg(f.view(b, t, *f.shape[1:]), valid, dates)
                for agg, f in zip(self.temporal_aggregation, feats)]
