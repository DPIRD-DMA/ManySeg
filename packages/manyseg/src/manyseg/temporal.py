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
* ``"concat"`` (option ``dates``, required): order-keeping, the dates' features concatenated in
  date order then 1x1 conv-BN-ReLU; needs exactly ``dates`` dates (export with that fixed T);
* ``"diff"``: ``|f_2 - f_1|`` of exactly two dates (bitemporal change); zero where either
  date is masked (a missing date is not change);
* ``"attention"``: lightweight temporal attention per pixel, in the spirit of L-TAE (Garnot
  & Landrieu 2020) as used by U-TAE: each of ``heads`` channel groups gets softmax weights over
  the dates from a learned query against keys computed from the features and a sinusoidal
  encoding of the dates. The query starts at zero, so at initialisation the weights are uniform
  and the model is exactly the masked mean. Options: ``heads`` (default 4, reduced to a divisor
  of each level's channels), ``key_dim`` (16), ``date_dim`` (32), ``max_period`` (10000 days).

A sample whose dates are all masked yields zero features. Masked dates never affect the
output, even if they contain NaNs: in training mode only valid dates are encoded (masked ones
stay out of BatchNorm batch statistics); in eval mode all dates are encoded and the masked ones
are dropped by the aggregation (static shapes for export).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn

AGGREGATIONS = ("mean", "max", "attention", "concat", "diff")


def parse_temporal(name: str, cfg) -> tuple[str, dict]:
    cfg = {"type": cfg} if isinstance(cfg, str) else dict(cfg or {"type": "mean"})
    kind = cfg.pop("type", "mean")
    if kind not in AGGREGATIONS:
        raise ValueError(f"input {name!r}: temporal aggregation must be one of {AGGREGATIONS}, "
                         f"got {kind!r}")
    allowed = {"attention": {"heads", "key_dim", "date_dim", "max_period"},
               "concat": {"dates"}}.get(kind, set())
    bad = set(cfg) - allowed
    if bad:
        raise ValueError(f"input {name!r}: unknown temporal options {sorted(bad)} for {kind!r}")
    if kind == "concat" and int(cfg.get("dates", 0)) < 1:
        raise ValueError(f"input {name!r}: 'concat' needs the fixed number of dates, e.g. "
                         "{'type': 'concat', 'dates': 2}")
    if "date_dim" in cfg and (int(cfg["date_dim"]) < 2 or int(cfg["date_dim"]) % 2):
        raise ValueError(f"input {name!r}: date_dim must be an even number >= 2, got "
                         f"{cfg['date_dim']}")
    return kind, cfg


def _per_date(v: Tensor | None, b: int, t: int, device, what: str, dtype) -> Tensor | None:
    if v is None:
        return None
    if v.ndim != 2 or v.shape[0] != b or v.shape[1] != t:
        raise ValueError(f"time-series {what} must be (B, T) = {(b, t)}, got {tuple(v.shape)}")
    return v.to(device=device, dtype=dtype)


class MeanAggregation(nn.Module):
    def forward(self, f: Tensor, mask: Tensor, dates: Tensor | None = None) -> Tensor:
        m = mask[:, :, None, None, None]  # B T 1 1 1; select (NaN * 0 is NaN), sum in fp32
        s = torch.where(m, f.float(), torch.zeros((), device=f.device)).sum(1)
        return (s / mask.sum(1).clamp(min=1)[:, None, None, None]).to(f.dtype)


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
        weights = torch.where(any_valid, weights, torch.zeros_like(weights))
        g = f.view(b, t, self.heads, c // self.heads, h, w).float()  # weighted sum in fp32
        g = torch.where(mask[:, :, None, None, None, None], g, torch.zeros_like(g))
        return (g * weights[:, :, :, None]).sum(1).reshape(b, c, h, w).to(f.dtype)


class ConcatAggregation(nn.Module):
    """Order-keeping: the ``dates`` feature maps concatenated in date order (masked dates
    zeroed), then 1x1 conv-BN-ReLU back to the level's channels. Needs exactly ``dates``."""

    def __init__(self, channels: int, dates: int) -> None:
        super().__init__()
        self.dates = int(dates)
        self.fuse = nn.Sequential(nn.Conv2d(self.dates * channels, channels, 1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

    def forward(self, f: Tensor, mask: Tensor, dates: Tensor | None = None) -> Tensor:
        if f.shape[1] != self.dates:
            raise ValueError(f"'concat' aggregation was built for {self.dates} dates, got "
                             f"{f.shape[1]}")
        f = torch.where(mask[:, :, None, None, None], f, torch.zeros_like(f))
        return self.fuse(f.flatten(1, 2))


class DiffAggregation(nn.Module):
    """``|f_2 - f_1|`` of a two-date series (bitemporal change); masked dates are zeroed."""

    def forward(self, f: Tensor, mask: Tensor, dates: Tensor | None = None) -> Tensor:
        if f.shape[1] != 2:
            raise ValueError(f"'diff' aggregation takes exactly 2 dates, got {f.shape[1]}")
        # A missing endpoint would look like change: zero unless both dates are valid.
        both = (mask[:, 0] & mask[:, 1])[:, None, None, None]
        d = (f[:, 1] - f[:, 0]).abs()
        return torch.where(both, d, torch.zeros_like(d))


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
        elif kind == "concat":
            aggs = [ConcatAggregation(s.channels, **opts) for s in self.feature_specs]
        elif kind == "diff":
            aggs = [DiffAggregation() for _ in self.feature_specs]
        else:
            aggs = [AttentionAggregation(s.channels, **opts) for s in self.feature_specs]
        self.temporal_aggregation = nn.ModuleList(aggs)
        # "concat" and "diff" need an exact number of dates (export keeps T static for them)
        self.fixed_dates: int | None = {"concat": getattr(aggs[0], "dates", None),
                                        "diff": 2}.get(kind) if aggs else None

    def forward(self, x: Tensor, mask: Tensor | None = None,
                dates: Tensor | None = None) -> list[Tensor]:
        if x.ndim != 5:
            raise ValueError(f"time-series input must be (B, T, C, H, W), got shape "
                             f"{tuple(x.shape)}")
        b, t = x.shape[:2]
        valid = _per_date(mask, b, t, x.device, "mask", torch.bool)
        dates = _per_date(dates, b, t, x.device, "dates", torch.float32)
        if valid is None:
            valid = torch.ones(b, t, dtype=torch.bool, device=x.device)
        frames = x.flatten(0, 1)
        if self.training and not bool(valid.any()):
            # Nothing valid: zero features of the right shapes, without an empty backbone batch.
            h, w = x.shape[-2:]
            feats = [x.new_zeros(b * t, s.channels, -(-h // s.stride), -(-w // s.stride))
                     for s in self.feature_specs]
        elif self.training and not bool(valid.all()):
            # Encode only the valid dates: masked ones must not enter BatchNorm batch
            # statistics. (In eval mode BN uses running statistics, so all dates are encoded
            # and masked ones are dropped by the aggregation; this keeps export static.)
            keep = valid.flatten().nonzero().squeeze(1)
            enc = self.backbone(frames[keep])
            feats = [None if e is None else e.new_zeros(b * t, *e.shape[1:])
                     .index_copy(0, keep, e) for e in enc]
        else:
            feats = self.backbone(frames)
        return [None if f is None else agg(f.view(b, t, *f.shape[1:]), valid, dates)
                for agg, f in zip(self.temporal_aggregation, feats)]

    def _restrict_levels(self, levels) -> None:
        """Only ``levels`` are read: the others get the parameter-free masked mean (a learned
        aggregation there would never get a gradient)."""
        for i, agg in enumerate(self.temporal_aggregation):
            if i not in set(levels) and any(True for _ in agg.parameters()):
                self.temporal_aggregation[i] = MeanAggregation()
        if hasattr(self.backbone, "_restrict_levels"):
            self.backbone._restrict_levels(levels)
