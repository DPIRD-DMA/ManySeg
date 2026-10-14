"""Metadata encoder: a packed ``(B, N)`` float vector -> per-field encodings -> MLP -> ``(B, D)``.

Input convention (one convention, chosen for export): a vector input is ONE float tensor
``(B, N)`` whose columns are the declared fields in declaration order (``coords`` takes two
columns, lat then lon; ``categorical`` one column holding the id). A single tensor keeps the
model's input a flat ``dict[str, Tensor]`` (one ONNX input per vector, only a batch dimension
to make dynamic). :meth:`MetadataEncoder.pack` builds it from per-field values.

Missing values: a field is missing where the optional mask ``(B, F)`` (one column per field,
1 = present) is 0, or where any of its columns is NaN, or where the field rejects the value
(out-of-range category). Missing fields are replaced by a learned per-field "unknown" vector
(the "unknown embedding"), so a model runs with every field missing. During training,
``dropout`` drops each field of each sample independently with that probability, so the model
learns to cope with missing metadata.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from ..registry import check_module_names
from .fields import ContinuousField, build_field


class MetadataEncoder(nn.Module):
    """Encode metadata fields and mix them with a 2-layer MLP.

    Args:
        fields: ``{name: {"type": ..., **options}}`` (see ``FIELD_TYPES``), or
        channels: ``N`` pre-normalised floats (``N`` continuous fields named ``"0"``...).
        dim: Output embedding size ``D``; ``None`` skips the MLP and returns the field
            encodings (``D = width``), as used by the ``channels`` injection mode.
        dropout: Training-time probability of dropping each field (per sample) to unknown.
    """

    def __init__(self, fields: Mapping[str, Mapping] | None = None, *,
                 channels: int | None = None, dim: int | None = 64, dropout: float = 0.0) -> None:
        super().__init__()
        if (fields is None) == (channels is None):
            raise ValueError("a vector input takes exactly one of 'fields' or 'channels'")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")
        if fields is not None:
            if not fields:
                raise ValueError("'fields' must not be empty")
            check_module_names("field", fields)
            built = {n: build_field(cfg) for n, cfg in fields.items()}
        else:
            if channels < 1:
                raise ValueError(f"channels must be >= 1, got {channels}")
            built = {str(i): ContinuousField() for i in range(channels)}
        self.fields = nn.ModuleDict(built)
        self.field_names = tuple(built)
        self.unknown = nn.ParameterDict({n: nn.Parameter(torch.zeros(f.width))
                                         for n, f in built.items()})
        self._cols, c = [], 0
        for f in built.values():
            self._cols.append((c, c + f.columns))
            c += f.columns
        self.columns = c
        self.width = sum(f.width for f in built.values())
        self.dropout = float(dropout)
        if dim is None:
            self.dim, self.mlp = self.width, nn.Identity()
        else:
            self.dim = int(dim)
            self.mlp = nn.Sequential(nn.Linear(self.width, self.dim), nn.GELU(),
                                     nn.Linear(self.dim, self.dim))

    def pack(self, values: Mapping[str, Tensor]) -> Tensor:
        """Per-field values ``{name: (B,) or (B, columns)}`` -> the packed ``(B, N)`` tensor
        (use NaN for missing values). Not part of ``forward``."""
        cols = []
        for n in self.field_names:
            if n not in values:
                raise KeyError(f"missing field {n!r}; expected {list(self.field_names)}")
            v = torch.as_tensor(values[n], dtype=torch.float32)
            cols.append(v.reshape(v.shape[0], -1))
        return torch.cat(cols, dim=1)

    def encode_fields(self, values: Tensor, mask: Tensor | None = None) -> Tensor:
        """``(B, N)`` values (+ optional ``(B, F)`` presence mask) -> ``(B, width)``
        concatenated field encodings, with unknown vectors in place of missing fields."""
        if isinstance(values, Mapping):
            raise TypeError("metadata must be one packed (B, N) float tensor, not a dict of "
                            "fields; build it with MetadataEncoder.pack({field: values})")
        if values.ndim != 2 or values.shape[1] != self.columns:
            raise ValueError(f"metadata must be (B, {self.columns}) with {self.columns} columns "
                             f"for fields {list(self.field_names)}; got {tuple(values.shape)}")
        finite = torch.isfinite(values)
        values = torch.where(finite, values, torch.zeros_like(values))
        b = values.shape[0]
        drop = None
        if self.training and self.dropout > 0:
            drop = torch.rand(b, len(self.field_names), device=values.device) < self.dropout
        out = []
        for i, (name, (a, z)) in enumerate(zip(self.field_names, self._cols)):
            e, present = self.fields[name](values[:, a:z])
            present = present & finite[:, a:z].all(dim=1)
            if mask is not None:
                present = present & (mask[:, i] > 0)
            if drop is not None:
                present = present & ~drop[:, i]
            unk = self.unknown[name].to(e.dtype).expand(b, -1)
            out.append(torch.where(present[:, None], e, unk))
        return torch.cat(out, dim=1)

    def forward(self, values: Tensor, mask: Tensor | None = None) -> Tensor:
        return self.mlp(self.encode_fields(values, mask))
