"""Single-tensor outputs for pipelines that expect one (fastai metrics, old training loops)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class ConcatHeads(nn.Module):
    """Wraps a dict-output model; ``forward`` returns ``torch.cat([out[k] for k in order], 1)``.

    ``channels`` maps each head to its channel slice of the output (``out[:, sl]``; empty
    if a custom head class has no ``out_channels`` attribute). Save,
    describe and ``select_heads`` work on ``.model``; the wrapper adds no parameters, so its
    ``state_dict`` is the model's under a ``model.`` prefix."""

    def __init__(self, model: nn.Module, order: Sequence[str]) -> None:
        super().__init__()
        if isinstance(order, str):
            raise TypeError(f"order is a list of head names, got the string {order!r}")
        heads = getattr(model, "heads", None)
        if heads is None:
            raise TypeError(f"concat_heads takes a manyseg model, got {type(model).__name__}")
        if getattr(model, "single_output", None) is not None:
            raise ValueError("this model returns one head's tensor (single_output); it needs "
                             "no concatenation")
        order = list(order)
        unknown = [k for k in order if k not in heads]
        if unknown or not order or len(set(order)) != len(order):
            raise ValueError(f"order must name distinct heads of {list(heads)}, got {order}")
        scopes = {getattr(heads[k], "scope", "pixel") for k in order}
        if len(scopes) > 1:
            raise ValueError("pixel and image heads have different shapes; concatenate one "
                             "scope at a time")
        if getattr(heads[order[0]], "scope", "pixel") == "pixel" and len(
                {getattr(heads[k], "factor", 1) for k in order}) > 1:
            raise ValueError("heads with different 'scale' have different grids; concatenate "
                             "heads of one scale")
        self.model = model
        self.order = order
        self.channels: dict[str, slice] = {}
        start = 0
        for k in order:
            c = _out_channels(heads[k])
            if c is None:  # a custom head that doesn't say
                self.channels = {}
                break
            self.channels[k] = slice(start, start + c)
            start += c

    def forward(self, *args, **kwargs) -> Tensor:
        out = self.model(*args, **kwargs)
        missing = [k for k in self.order if k not in out]
        if missing:
            raise KeyError(f"model returned no {missing}: were heads removed by select_heads?")
        return torch.cat([out[k] for k in self.order], 1)

    def extra_repr(self) -> str:
        return f"order={self.order}"


def _out_channels(head: nn.Module) -> int | None:
    from ..heads import ImageHead, PixelHead

    if isinstance(head, PixelHead):
        return head.conv.out_channels
    if isinstance(head, ImageHead):
        return head.fc.out_features
    return getattr(head, "out_channels", None)


def concat_heads(model: nn.Module, order: Sequence[str]) -> ConcatHeads:
    """A wrapper returning one tensor: the heads in ``order`` concatenated along channels.

    For pipelines built around a single output tensor (fastai metrics that slice
    ``pred[:, :2]``, older losses). Heads must share a scope and, for pixel heads, a
    ``scale``. Prefer the dict output where the pipeline allows: losses that read
    ``out["name"]`` don't break when a head changes width."""
    return ConcatHeads(model, order)
