"""Multi-input encoders. They behave like a backbone (``feature_specs``, ``list[Tensor]``,
``None`` at levels nothing reads) but take a ``dict[str, Tensor]``; padding happens inside
(same-grid inputs to a common multiple, same-extent inputs to their own, fixed-size
backbones to their size)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..features import FeatureSpec
from ..utils.spatial import crop_to, pad_to_multiple, resize_bilinear
from .ops import FUSION_OPS

ALIGNMENTS = ("same_grid", "same_extent")
"""Input ``"alignment"`` values: same pixel grid, or same extent at another resolution."""


def _check_inputs(inputs: Mapping[str, Tensor], names: Sequence[str],
                  optional: Sequence[str] = ()) -> None:
    if not isinstance(inputs, Mapping):
        raise TypeError(f"this model takes a dict of inputs with keys {list(names)}")
    for n in names:
        if n not in inputs:
            raise KeyError(f"missing input {n!r}; expected keys {list(names)}")
    unknown = [k for k in inputs if k not in names and k not in optional]
    if unknown:  # a misspelt "<name>_mask" would otherwise be ignored silently
        raise ValueError(f"unknown input keys {unknown}; this model takes {list(names)}"
                         + (f" and optionally {list(optional)}" if optional else ""))


def _resize_valid(f: Tensor, valid: tuple, target_valid: tuple, target_full: tuple) -> Tensor:
    """Resample the valid (unpadded) region of ``f`` onto the target's valid region, then pad
    to the target's full (padded) size. Used for ``same_extent`` alignment."""
    f = crop_to(f, *valid)  # not slicing: export would specialise on stride-aligned sizes
    f = resize_bilinear(f, target_valid)  # keeps the dtype: a concat fusion follows (E027)
    return F.pad(f, (0, target_full[1] - target_valid[1], 0, target_full[0] - target_valid[0]))


def _ceil_div(a, b):
    return (a + b - 1) // b


class EarlyFusionEncoder(nn.Module):
    """Concatenate inputs along channels (fixed key order) in front of one backbone.

    ``channels`` (optional, ``{name: C}``) records each input's share of the concatenation for
    :func:`manyseg.describe.describe`; the backbone's stem only sees the total.
    """

    fused_strides: tuple[int, ...] = ()

    def __init__(self, names: Sequence[str], backbone: nn.Module,
                 channels: Mapping[str, int] | None = None) -> None:
        super().__init__()
        self.names = tuple(names)
        self.channels = dict(channels or {})
        self.backbone = backbone
        self.feature_specs = backbone.feature_specs
        self.input_divisor = backbone.input_divisor
        self.primary = self.names[0]

    def _restrict_levels(self, levels) -> None:
        if hasattr(self.backbone, "_restrict_levels"):
            self.backbone._restrict_levels(levels)

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
        self.fixed_sizes = {n: fs for n in self.names if (fs := getattr(
            getattr(encoders[n], "backbone", encoders[n]), "fixed_size", None)) is not None}
        # same-grid inputs are all padded to a common multiple of their divisors, so features
        # line up level by level (as if encoded on one padded grid)
        self.grid_divisor = math.lcm(*(d for n, d in self.divisors.items()
                                       if self.alignment[n] == "same_grid"))
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

    def _restrict_levels(self, levels, per_input=None, check_inputs: bool = True) -> None:
        """Only the fused ``levels`` (and ``per_input``: ``{input: its own levels}``, for
        per-input heads) are read. Unread levels are not fused (the primary's feature passes
        through, as with ``fusion["strides"]``), and each encoder learns which of its levels
        are needed, so no module is left without a gradient."""
        levels = set(levels)
        need = {n: set((per_input or {}).get(n, ())) for n in self.names}
        if self.mode == "late":
            for (a, b), n in zip(self.groups, self.names):
                need[n] |= {i - a for i in levels if a <= i < b}
        else:
            for li, members in enumerate(self._plan):
                if li not in levels:
                    if len(members) > 1:
                        self._plan[li] = members[:1]
                        del self.fusers[str(li)]
                    continue
                for n, i in members:
                    need[n].add(i)
            self.fused_strides = tuple(self.feature_specs[li].stride
                                       for li, m in enumerate(self._plan) if len(m) > 1)
        # an input nothing reads is an error when it has parameters of its own (they would
        # never get a gradient); one sharing a read input's encoder only costs compute
        def params(n):  # the input's encoder and its channel adapter, if any
            mods = [self.encoders[n]] + ([self.adapters[n]] if n in self.adapters else [])
            return [p for m in mods for p in m.parameters()]

        shared = {id(p) for n in self.names if need[n] for p in params(n)}
        unread = [n for n in self.names if not need[n] and
                  any(id(p) not in shared for p in params(n))]
        if unread and check_inputs:
            how = ("late fusion combines inputs in the decoder, and no head reads one" if
                   self.mode == "late" else "no fused level that a decoder or head reads "
                   "includes it")
            raise ValueError(f"input(s) {unread} would never be read: {how}. Add a head that "
                             "reads a decoder, change the fusion, or drop the input")
        for n in self.names:  # trailing unread levels: their blocks would get no gradient
            specs = self.encoders[n].feature_specs
            if check_inputs and need[n] and max(need[n]) < len(specs) - 1:
                last = max(need[n])
                raise ValueError(
                    f"input {n!r}: its levels at strides "
                    f"{[s.stride for s in specs[last + 1:]]} are never read (nothing fuses or "
                    f"decodes them, e.g. the primary {self.primary!r} has no level there), so "
                    "the blocks computing them would get no gradient. Limit the input with "
                    f"'out_indices' so its deepest level is stride {specs[last].stride}")
        for n in self.names:
            if hasattr(self.encoders[n], "_restrict_levels"):
                self.encoders[n]._restrict_levels(sorted(need[n]))

    def _pad(self, n: str, x: Tensor) -> Tensor:
        """Pad to the encoder's stride multiple, or up to a fixed-size backbone's size."""
        fixed = self.fixed_sizes.get(n)
        if fixed is None:
            return pad_to_multiple(x, self.grid_divisor if self.alignment[n] == "same_grid"
                                   else self.divisors[n])
        h, w = x.shape[-2], x.shape[-1]
        if h > fixed[0] or w > fixed[1]:
            raise ValueError(f"input {n!r}: its backbone has a fixed input size {fixed}; got "
                             f"{(h, w)}. Use tiled inference (ManyPred) or "
                             "backbone_kwargs={'img_size': ...}")
        return F.pad(x, (0, fixed[1] - w, 0, fixed[0] - h))

    def _encode(self, inputs: Mapping[str, Tensor]):
        _check_inputs(inputs, self.names, [f"{n}_{s}" for n in self.names
                                           if getattr(self.encoders[n], "temporal", False)
                                           for s in ("mask", "dates")])
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
            enc = self.encoders[n]
            x = self._pad(n, x)
            if getattr(enc, "temporal", False):  # (B, T, C, H, W) + optional mask / dates
                feats[n] = enc(x, inputs.get(f"{n}_mask"), inputs.get(f"{n}_dates"))
            else:
                feats[n] = enc(x)
            valid[n] = (h, w)
        return feats, valid

    def forward(self, inputs: Mapping[str, Tensor], return_inputs: bool = False):
        """Fused features; with ``return_inputs``, also ``{input: its own features}`` (padded
        grid), for per-input heads."""
        feats, valid = self._encode(inputs)
        if return_inputs:
            return self._fuse(feats, valid), feats
        return self._fuse(feats, valid)

    def _fuse(self, feats, valid) -> list[Tensor]:
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
                elif n in self.fixed_sizes or self.primary in self.fixed_sizes:
                    # same grid, one padded to a fixed size: the valid regions coincide
                    f = crop_to(f, ref.shape[-2], ref.shape[-1])
                xs.append(f)
            out.append(self.fusers[str(li)](xs))
        return out
