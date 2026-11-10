"""Fusion operators: combine same-stride feature maps from several inputs into one.

Every op is constructed as ``Op(in_channels: list[int], out_channels: int)`` and called with a
list of tensors of equal spatial size; ``inputs[0]`` is the primary input.
"""

from __future__ import annotations

import numbers
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..registry import Registry

FUSION_OPS = Registry("fusion op")
"""Fusion ops by ``fusion={"op": ...}`` name; ``FUSION_OPS.register("name")`` adds one."""


def _proj(cin: int, cout: int) -> nn.Module:
    return nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1)


@FUSION_OPS.register("concat")
class ConcatFusion(nn.Module):
    """Concatenate channels, then 1x1 conv-BN-ReLU to ``out_channels``."""

    def __init__(self, in_channels: Sequence[int], out_channels: int) -> None:
        super().__init__()
        self.fuse = nn.Sequential(nn.Conv2d(sum(in_channels), out_channels, 1, bias=False),
                                  nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, xs: list[Tensor]) -> Tensor:
        return self.fuse(torch.cat(xs, dim=1))


def _two(op: str, in_channels: Sequence[int]) -> None:
    if len(in_channels) != 2:
        raise ValueError(f"'{op}' fusion compares exactly two inputs (e.g. before / after), got "
                         f"{len(in_channels)} at a fused level")


@FUSION_OPS.register("diff")
class DiffFusion(nn.Module):
    """``|a - b|`` of two inputs, each projected to ``out_channels`` (1x1, identity if equal):
    Siamese change-detection fusion (FC-Siam-diff). Symmetric in the two inputs when their
    projections are identities (a shared encoder)."""

    def __init__(self, in_channels: Sequence[int], out_channels: int) -> None:
        super().__init__()
        _two("diff", in_channels)
        self.proj = nn.ModuleList(_proj(c, out_channels) for c in in_channels)

    def forward(self, xs: list[Tensor]) -> Tensor:
        return (self.proj[0](xs[0]) - self.proj[1](xs[1])).abs()


@FUSION_OPS.register("concat_diff")
class ConcatDiffFusion(nn.Module):
    """``[a, b, |a - b|]`` (inputs projected to ``out_channels``), then 1x1 conv-BN-ReLU to
    ``out_channels``: keeps both dates and their difference (FC-Siam-conc with a difference)."""

    def __init__(self, in_channels: Sequence[int], out_channels: int) -> None:
        super().__init__()
        _two("concat_diff", in_channels)
        self.proj = nn.ModuleList(_proj(c, out_channels) for c in in_channels)
        self.fuse = nn.Sequential(nn.Conv2d(3 * out_channels, out_channels, 1, bias=False),
                                  nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, xs: list[Tensor]) -> Tensor:
        a, b = self.proj[0](xs[0]), self.proj[1](xs[1])
        return self.fuse(torch.cat([a, b, (a - b).abs()], dim=1))


@FUSION_OPS.register("add")
class AddFusion(nn.Module):
    """Project each input to ``out_channels`` (1x1, identity if equal) and sum."""

    def __init__(self, in_channels: Sequence[int], out_channels: int) -> None:
        super().__init__()
        self.proj = nn.ModuleList(_proj(c, out_channels) for c in in_channels)

    def forward(self, xs: list[Tensor]) -> Tensor:
        out = self.proj[0](xs[0])
        for p, x in zip(self.proj[1:], xs[1:]):
            out = out + p(x)
        return out


@FUSION_OPS.register("gated")
class GatedFusion(nn.Module):
    """Per-pixel softmax gates over inputs, predicted from all inputs jointly.

    Gate logits are initialised to zero, so fusion starts as an equal-weight average of the
    projected inputs.
    """

    def __init__(self, in_channels: Sequence[int], out_channels: int) -> None:
        super().__init__()
        self.proj = nn.ModuleList(_proj(c, out_channels) for c in in_channels)
        self.gate = nn.Conv2d(len(in_channels) * out_channels, len(in_channels), 3, padding=1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, xs: list[Tensor]) -> Tensor:
        ps = [p(x) for p, x in zip(self.proj, xs)]
        w = torch.softmax(self.gate(torch.cat(ps, dim=1)), dim=1)
        out = ps[0] * w[:, 0:1]
        for i in range(1, len(ps)):
            out = out + ps[i] * w[:, i:i + 1]
        return out


@FUSION_OPS.register("attention")
class ChannelAttentionFusion(nn.Module):
    """Squeeze-and-excitation over the concatenated channels, then 1x1 projection."""

    def __init__(self, in_channels: Sequence[int], out_channels: int, reduction: int = 4) -> None:
        super().__init__()
        c = sum(in_channels)
        hidden = max(8, c // reduction)
        self.se = nn.Sequential(nn.Conv2d(c, hidden, 1), nn.ReLU(inplace=True),
                                nn.Conv2d(hidden, c, 1), nn.Sigmoid())
        self.fuse = nn.Sequential(nn.Conv2d(c, out_channels, 1, bias=False),
                                  nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, xs: list[Tensor]) -> Tensor:
        x = torch.cat(xs, dim=1)
        return self.fuse(x * self.se(x.mean(dim=(-2, -1), keepdim=True)))


@FUSION_OPS.register("cross_attention")
class CrossAttentionFusion(nn.Module):
    """Primary tokens attend to the other inputs' tokens (multi-head, SDPA).

    ``out = primary + proj(attn(q=primary, kv=others))``; the output projection starts at zero,
    so the op is initially the identity on the primary input. Cost is quadratic in tokens: use
    it on coarse strides (e.g. 16, 32).
    """

    def __init__(self, in_channels: Sequence[int], out_channels: int, heads: int = 4,
                 dim: int | None = None) -> None:
        super().__init__()
        if len(in_channels) < 2:
            raise ValueError("cross_attention fuses a primary input with at least one other; "
                             f"got {len(in_channels)} input(s)")
        def positive_int(v):
            return isinstance(v, numbers.Integral) and not isinstance(v, bool) and v >= 1

        if not positive_int(heads):
            raise ValueError(f"cross_attention: heads must be an integer >= 1, got {heads!r}")
        if dim is not None and not positive_int(dim):
            raise ValueError(f"cross_attention: dim must be an integer >= 1, got {dim!r}")
        heads, dim = int(heads), min(256, out_channels) if dim is None else int(dim)
        if dim < heads:
            raise ValueError(f"cross_attention: dim {dim} is smaller than heads {heads}; each "
                             "head needs at least one channel")
        dim -= dim % heads
        self.heads = heads
        self.primary = _proj(in_channels[0], out_channels)
        self.q = nn.Conv2d(in_channels[0], dim, 1)
        self.kv = nn.ModuleList(nn.Conv2d(c, 2 * dim, 1) for c in in_channels[1:])
        self.out = nn.Conv2d(dim, out_channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _tokens(self, x: Tensor) -> Tensor:  # (B, C, H, W) -> (B, heads, HW, C/heads)
        b, c, h, w = x.shape
        # contiguous: SDPA's flash / memory-efficient kernels need a unit last-dim stride, and
        # otherwise fall back to the math kernel, which builds the full attention matrix
        # (measured: -50% step time, -77% memory)
        return x.reshape(b, self.heads, c // self.heads, h * w).transpose(-1, -2).contiguous()

    def forward(self, xs: list[Tensor]) -> Tensor:
        b, _, h, w = xs[0].shape
        q = self._tokens(self.q(xs[0]))
        ks, vs = [], []
        for kv, x in zip(self.kv, xs[1:]):
            k, v = kv(x).chunk(2, dim=1)
            ks.append(self._tokens(k))
            vs.append(self._tokens(v))
        a = F.scaled_dot_product_attention(q, torch.cat(ks, dim=2), torch.cat(vs, dim=2))
        a = a.transpose(-1, -2).reshape(b, -1, h, w)
        return self.primary(xs[0]) + self.out(a)
