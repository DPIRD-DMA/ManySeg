"""U-Net++ decoder (Zhou et al. 2018): nested, densely connected skip pathways.

Node ``X[i][j]`` (level ``i`` = stride of encoder feature ``i``, column ``j >= 1``) fuses all
previous nodes at its level with the upsampled node ``X[i+1][j-1]``. The final node
``X[0][L-1]`` is then upsampled without skips to ``out_stride``. Widths follow the U-Net
default at each stride, so ``unetplusplus`` and ``unet`` are directly comparable.

``out_stride > 1`` drops every encoder level finer than ``out_stride`` (with all of its nested
nodes) and the tail blocks below it, so the reduced decoder is the default one minus those
modules: e.g. with a stride-2..32 encoder, ``out_stride=2`` drops the tail block to stride 1
and ``out_stride=4`` also drops the stride-2 row ``X[0][*]`` (10 -> 6 nodes).

Not parameter-identical to SMP's ``UnetPlusPlus``, which sizes nested nodes from skip widths
(resnet34 @256²: ManySeg 25.8 M params / 26.8 GFLOPs vs SMP 26.1 M / 36.8 GFLOPs).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from ..features import FeatureSpec
from ..layers import init_decoder_
from .base import DECODERS, check_power_of_two_stride
from .unet import UNetBlock, upsample2x


class _Node(nn.Module):
    def __init__(self, cin: int, cout: int, norm, upsample: str) -> None:
        super().__init__()
        self.block = UNetBlock(cin, 0, cout, norm, upsample)
        self.upsample = upsample
        self.exact_2x = False  # set by the model when inputs are padded to the stride

    def forward(self, below: Tensor, same: list[Tensor]) -> Tensor:
        ref = same[0]
        up = upsample2x(below, ref.shape[-2:], self.upsample, self.exact_2x)
        x = torch.cat([up, *same], dim=1)
        return self.block.conv2(self.block.conv1(x))


@DECODERS.register("unetplusplus")
class UNetPlusPlusDecoder(nn.Module):
    def __init__(self, in_specs: Sequence[FeatureSpec], out_stride: int = 1,
                 norm: str | None = "batch", upsample: str = "nearest") -> None:
        super().__init__()
        for s in in_specs:
            if s.stride & (s.stride - 1):
                raise ValueError(f"UNetPlusPlusDecoder needs power-of-two strides, got {s.stride}")
        deepest = max(s.stride for s in in_specs)
        self._deepest_stride = deepest
        check_power_of_two_stride("UNetPlusPlusDecoder", out_stride, deepest)
        order = sorted((i for i, s in enumerate(in_specs) if s.stride >= out_stride),
                       key=lambda i: in_specs[i].stride)
        specs = [in_specs[i] for i in order]
        self._order = order
        # Nodes keep the level numbers of the full decoder, so a reduced decoder's state_dict is
        # a subset of the default one's.
        self._first = len(in_specs) - len(order)
        n_blocks = int(math.log2(deepest // out_stride))
        default = [max(16, 256 >> i) for i in range(n_blocks)]
        width = {deepest >> (k + 1): c for k, c in enumerate(default)}  # stride -> channels
        L = len(specs)
        self.L = L
        nodes = {}
        ch = {(i, 0): specs[i].channels for i in range(L)}
        for j in range(1, L):
            for i in range(0, L - j):
                cout = width[specs[i].stride]
                cin = ch[(i + 1, j - 1)] + sum(ch[(i, k)] for k in range(j))
                nodes[f"{i + self._first}_{j}"] = _Node(cin, cout, norm, upsample)
                # this node upsamples level i + 1 to level i: 2x unless out_indices skip one
                nodes[f"{i + self._first}_{j}"].step_2x = \
                    specs[i + 1].stride == 2 * specs[i].stride
                ch[(i, j)] = cout
        self.nodes = nn.ModuleDict(nodes)
        tail, c, s = [], ch[(0, L - 1)], specs[0].stride
        while s > out_stride:
            s //= 2
            tail.append(UNetBlock(c, 0, width[s], norm, upsample))
            c = width[s]
        self.tail = nn.ModuleList(tail)
        self.out_spec = FeatureSpec(c, out_stride, "unetplusplus")
        init_decoder_(self)

    def _set_input_divisor(self, divisor: int) -> None:
        """Inputs padded to a multiple of the deepest stride: every level is exactly twice the
        next, so the upsampling needs no gather (E025)."""
        for node in self.nodes.values():
            node.exact_2x = divisor % self._deepest_stride == 0 and node.step_2x

    def forward(self, features: list[Tensor]) -> Tensor:
        e = [features[i] for i in self._order]
        X = {(i, 0): e[i] for i in range(self.L)}
        for j in range(1, self.L):
            for i in range(0, self.L - j):
                X[(i, j)] = self.nodes[f"{i + self._first}_{j}"](
                    X[(i + 1, j - 1)], [X[(i, k)] for k in range(j)])
        x = X[(0, self.L - 1)]
        for blk in self.tail:
            x = blk(x, None)
        return x
