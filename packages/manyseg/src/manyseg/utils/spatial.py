from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def padding_for(h, w, divisor: int):
    """(pad_h, pad_w) taking (h, w) up to multiples of ``divisor``.

    Uses floor division rather than ``%``: Inductor cannot lower ``%`` on symbolic sizes.
    """
    return (h + divisor - 1) // divisor * divisor - h, (w + divisor - 1) // divisor * divisor - w


def crop_to(y: Tensor, h, w) -> Tensor:
    """Crop the bottom/right of ``y`` to ``(h, w)``.

    Implemented as negative padding so the output size is ``size - (size - h)`` symbolically.
    Plain slicing ``y[..., :h, :w]`` makes torch.export try to prove ``h <= size``, which it
    cannot do for ceil-rounded sizes.
    """
    return F.pad(y, (0, w - y.shape[-1], 0, h - y.shape[-2]))


def pad_to_multiple(x: Tensor, divisor: int, value: float = 0.0) -> Tensor:
    """Zero-pad the bottom/right of ``x`` so H and W are multiples of ``divisor``.

    Pixel (i, j) keeps its position, so outputs can be cropped back with ``[..., :H, :W]``.
    """
    if divisor <= 1:
        return x
    pad_h, pad_w = padding_for(x.shape[-2], x.shape[-1], divisor)
    return F.pad(x, (0, pad_w, 0, pad_h), value=value)


def _nearest_index(n_in, n_out, device) -> Tensor:
    # the exact nearest rule floor(dst * n_in / n_out), in integers
    return torch.clamp(torch.arange(n_out, device=device) * n_in // n_out, max=n_in - 1)


def resize_nearest(x: Tensor, size) -> Tensor:
    """Nearest-neighbour resize to ``size``: source row ``floor(i * H / H_out)``, as two index
    gathers in integer arithmetic.

    Equal to ``F.interpolate(x, size, mode="nearest")`` whenever the output is a multiple of
    the input (always with stride padding); at other sizes PyTorch's float32 rounding can pick
    the row before the exact one. Used instead of ``F.interpolate`` because torch.compile
    (torch 2.14, CUDA, ``dynamic=True``) miscompiles both ``mode="nearest"`` and any float
    version of this index: wrong outputs at sizes such as 49x66 -> 98x132 after compiling at
    another size (docs/upstream/torch-inductor-dynamic-nearest.md). The integer form compiles
    correctly.
    """
    h, w = x.shape[-2], x.shape[-1]
    x = x.index_select(-2, _nearest_index(h, size[0], x.device))
    return x.index_select(-1, _nearest_index(w, size[1], x.device))
