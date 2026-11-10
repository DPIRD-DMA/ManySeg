"""Encoders for single metadata fields (date, angles, coordinates, sensor id, ...).

A field reads ``columns`` consecutive columns of the packed ``(B, N)`` metadata tensor and
returns ``(encoding (B, width), valid (B,) bool)``. ``valid`` marks values the field can
encode (e.g. a category id in range); invalid and missing values are replaced by a learned
"unknown" vector in :class:`~manyseg.conditioning.MetadataEncoder`, never here. Inputs are
already finite (NaNs are replaced before fields see them).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from ..registry import Registry

FIELD_TYPES = Registry("metadata field type")
"""Metadata field encoders by field ``"type"``; ``FIELD_TYPES.register("name")`` adds one."""


@FIELD_TYPES.register("continuous")
class ContinuousField(nn.Module):
    """``(v - mean) / std``: one column, width 1."""

    columns, width = 1, 1

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        super().__init__()
        if std <= 0:
            raise ValueError(f"std must be > 0, got {std}")
        self.mean, self.std = float(mean), float(std)

    def forward(self, v: Tensor) -> tuple[Tensor, Tensor]:
        return (v - self.mean) / self.std, torch.ones_like(v[:, 0], dtype=torch.bool)


@FIELD_TYPES.register("cyclic")
class CyclicField(nn.Module):
    """``[sin, cos](2 pi v / period)`` (day of year, hour, azimuth): continuous across the
    period boundary. One column, width 2."""

    columns, width = 1, 2

    def __init__(self, period: float) -> None:
        super().__init__()
        if period <= 0:
            raise ValueError(f"period must be > 0, got {period}")
        self.period = float(period)

    def forward(self, v: Tensor) -> tuple[Tensor, Tensor]:
        a = v * (2 * math.pi / self.period)
        return torch.cat([torch.sin(a), torch.cos(a)], dim=1), \
            torch.ones_like(v[:, 0], dtype=torch.bool)


@FIELD_TYPES.register("coords")
class CoordsField(nn.Module):
    """Latitude/longitude in degrees (two columns, in that order).

    Encoded as the point ``u`` on the unit sphere ``(cos lat cos lon, cos lat sin lon,
    sin lat)``, which is continuous across the dateline and at the poles, plus
    ``frequencies - 1`` octaves of Fourier features ``[sin, cos](2^k pi u)`` (k = 1, ...) for
    finer spatial detail. The octaves are functions of ``u``, not of raw lat/lon, so they keep
    the dateline continuity and pole invariance. Width ``3 + 6 (frequencies - 1)``.
    """

    columns = 2

    def __init__(self, frequencies: int = 1) -> None:
        super().__init__()
        if frequencies < 1:
            raise ValueError(f"frequencies must be >= 1, got {frequencies}")
        self.frequencies = int(frequencies)
        self.width = 3 + 6 * (self.frequencies - 1)

    def forward(self, v: Tensor) -> tuple[Tensor, Tensor]:
        lat, lon = torch.deg2rad(v[:, 0:1]), torch.deg2rad(v[:, 1:2])
        u = torch.cat([torch.cos(lat) * torch.cos(lon), torch.cos(lat) * torch.sin(lon),
                       torch.sin(lat)], dim=1)
        parts = [u]
        for k in range(1, self.frequencies):
            a = u * (math.pi * 2 ** k)
            parts += [torch.sin(a), torch.cos(a)]
        return torch.cat(parts, dim=1), torch.ones_like(v[:, 0], dtype=torch.bool)


@FIELD_TYPES.register("categorical")
class CategoricalField(nn.Module):
    """Integer id in ``[0, num)`` (stored as a float column) -> learned embedding of size
    ``dim``. A value that is not integer-valued (``|v - round(v)| > 1e-6``), or whose rounded id
    lies outside ``[0, num)``, is invalid, i.e. missing."""

    columns = 1

    def __init__(self, num: int, dim: int = 8) -> None:
        super().__init__()
        if num < 1:
            raise ValueError(f"num must be >= 1, got {num}")
        self.num, self.width = int(num), int(dim)
        self.embedding = nn.Embedding(self.num, self.width)
        nn.init.normal_(self.embedding.weight, std=1.0)

    def forward(self, v: Tensor) -> tuple[Tensor, Tensor]:
        r = torch.round(v[:, 0])
        # range-check the rounded id that is looked up: 2.9999995 is integer-valued (within
        # 1e-6) and below num=3, but rounds to 3, which is out of range
        valid = ((v[:, 0] - r).abs() <= 1e-6) & (r >= 0) & (r < self.num)
        idx = r.long()
        return self.embedding(torch.where(valid, idx, torch.zeros_like(idx))), valid


def build_field(cfg: Mapping) -> nn.Module:
    """Build a field encoder from ``{"type": ..., **options}``."""
    cfg = dict(cfg)
    return FIELD_TYPES.get_or_raise(cfg.pop("type", None))(**cfg)
