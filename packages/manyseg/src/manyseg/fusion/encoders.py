"""Multi-input encoders. They behave like a backbone (``feature_specs``, ``list[Tensor]``)
but take a ``dict[str, Tensor]``; padding to each encoder's divisor happens inside."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..features import FeatureSpec
from ..utils.spatial import pad_to_multiple
from .ops import FUSION_OPS

ALIGNMENTS = ("same_grid", "same_extent")


def _check_inputs(inputs: Mapping[str, Tensor], names: Sequence[str]) -> None:
    if not isinstance(inputs, Mapping):
        raise TypeError(f"this model takes a dict of inputs with keys {list(names)}")
    for n in names:
        if n not in inputs:
            raise KeyError(f"missing input {n!r}; expected keys {list(names)}")


def _resize_valid(f: Tensor, valid: tuple, target_valid: tuple, target_full: tuple) -> Tensor:
    """Resample the valid (unpadded) region of ``f`` onto the target's valid region, then pad
    to the target's full (padded) size. Used for ``same_extent`` alignment."""
    f = f[..., : valid[0], : valid[1]]
    f = F.interpolate(f, size=target_valid, mode="bilinear", align_corners=False)
    return F.pad(f, (0, target_full[1] - target_valid[1], 0, target_full[0] - target_valid[0]))


def _ceil_div(a, b):
    return (a + b - 1) // b


class EarlyFusionEncoder(nn.Module):
    """Concatenate inputs along channels (fixed key order) in front of one backbone."""

    fused_strides: tuple[int, ...] = ()

    def __init__(self, names: Sequence[str], backbone: nn.Module) -> None:
        super().__init__()
        self.names = tuple(names)
        self.backbone = backbone
        self.feature_specs = backbone.feature_specs
        self.input_divisor = backbone.input_divisor
        self.primary = self.names[0]

    def forward(self, inputs: Mapping[str, Tensor]) -> list[Tensor]:
        _check_inputs(inputs, self.names)
        ref = inputs[self.primary].shape[-2:]
        for n in self.names[1:]:
            if inputs[n].shape[-2:] != ref:
                raise ValueError(
                    f"early fusion needs same_grid inputs; {n!r} is {tuple(inputs[n].shape[-2:])}"
                    f" but {self.primary!r} is {tuple(ref)}"
                )
        x = torch.cat([inputs[n] for n in self.names], dim=1)
        return self.backbone(pad_to_multiple(x, self.input_divisor))


class MultiEncoder(nn.Module):
    """Per-input encoders whose features are fused per stride (intermediate fusion), or
    returned side by side for a :class:`LateFusionDecoder` (``mode="late"``).

    Args:
        encoders: ``{name: backbone}``; the first is the primary input. Sharing a backbone
            between names is allowed (same module object).
        adapters: optional per-input modules applied before the encoder (e.g. projection to a
            shared encoder's channel count).
        alignment: ``{name: "same_grid" | "same_extent"}`` (default same_grid).
        op: fusion op name (intermediate mode).
        strides: strides to fuse (default: every primary stride present in another input).
    """

    def __init__(self, encoders: Mapping[str, nn.Module], *, mode: str = "intermediate",
                 op: str = "concat", strides: Sequence[int] | None = None,
                 alignment: Mapping[str, str] | None = None,
                 adapters: Mapping[str, nn.Module] | None = None,
                 op_kwargs: Mapping | None = None) -> None:
        super().__init__()
        if mode not in ("intermediate", "late"):
            raise ValueError(f"mode must be 'intermediate' or 'late', got {mode!r}")
        self.names = tuple(encoders)
        self.primary = self.names[0]
        self.mode = mode
        self.encoders = nn.ModuleDict(encoders)
        self.adapters = nn.ModuleDict(adapters or {})
        self.alignment = {n: (alignment or {}).get(n, "same_grid") for n in self.names}
        for n, a in self.alignment.items():
            if a not in ALIGNMENTS:
                raise ValueError(f"alignment for {n!r} must be one of {ALIGNMENTS}, got {a!r}")
        self.divisors = {n: int(encoders[n].input_divisor) for n in self.names}
        self.input_divisor = 1  # padding is handled per input here
        pspecs = encoders[self.primary].feature_specs

        if mode == "late":
            self.groups: list[tuple[int, int]] = []
            specs: list[FeatureSpec] = []
            for n in self.names:
                s = list(encoders[n].feature_specs)
                self.groups.append((len(specs), len(specs) + len(s)))
                specs += s
            self.feature_specs = tuple(specs)
            self.fused_strides = ()
            return

        by_stride = {n: {s.stride: i for i, s in enumerate(encoders[n].feature_specs)}
                     for n in self.names}
        candidates = [s.stride for s in pspecs
                      if any(s.stride in by_stride[n] for n in self.names[1:])]
        self.fused_strides = tuple(candidates if strides is None
                                   else [s for s in candidates if s in set(strides)])
        if strides is not None and set(strides) - set(self.fused_strides):
            missing = sorted(set(strides) - set(self.fused_strides))
            raise ValueError(f"cannot fuse strides {missing}: not present in the primary and "
                             f"at least one other input")
        # plan: per primary level -> list of (name, feature index)
        self._plan: list[list[tuple[str, int]]] = []
        ops = {}
        for li, spec in enumerate(pspecs):
            members = [(self.primary, li)]
            if spec.stride in self.fused_strides:
                members += [(n, by_stride[n][spec.stride]) for n in self.names[1:]
                            if spec.stride in by_stride[n]]
                chans = [encoders[n].feature_specs[i].channels for n, i in members]
                ops[str(li)] = FUSION_OPS.get_or_raise(op)(chans, spec.channels,
                                                           **dict(op_kwargs or {}))
            self._plan.append(members)
        self.fusers = nn.ModuleDict(ops)
        self.feature_specs = tuple(pspecs)

    def _encode(self, inputs: Mapping[str, Tensor]):
        _check_inputs(inputs, self.names)
        ph, pw = inputs[self.primary].shape[-2:]
        feats, valid = {}, {}
        for n in self.names:
            x = inputs[n]
            h, w = x.shape[-2:]
            if self.alignment[n] == "same_grid" and n != self.primary and (h, w) != (ph, pw):
                raise ValueError(
                    f"input {n!r} is {(h, w)} but primary {self.primary!r} is {(ph, pw)}; "
                    "inputs must share a grid (same_grid), or declare alignment='same_extent'"
                )
            if n in self.adapters:
                x = self.adapters[n](x)
            feats[n] = self.encoders[n](pad_to_multiple(x, self.divisors[n]))
            valid[n] = (h, w)
        return feats, valid

    def forward(self, inputs: Mapping[str, Tensor]) -> list[Tensor]:
        feats, valid = self._encode(inputs)
        if self.mode == "late":
            out = []
            for n in self.names:
                out += feats[n]
            return out
        ph, pw = valid[self.primary]
        out = []
        for li, members in enumerate(self._plan):
            ref = feats[self.primary][li]
            if len(members) == 1:
                out.append(ref)
                continue
            s = self.feature_specs[li].stride
            xs = [ref]
            for n, i in members[1:]:
                f = feats[n][i]
                if self.alignment[n] == "same_extent":
                    vh, vw = valid[n]
                    f = _resize_valid(f, (_ceil_div(vh, s), _ceil_div(vw, s)),
                                      (_ceil_div(ph, s), _ceil_div(pw, s)), ref.shape[-2:])
                xs.append(f)
            out.append(self.fusers[str(li)](xs))
        return out
