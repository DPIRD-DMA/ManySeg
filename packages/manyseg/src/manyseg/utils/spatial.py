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
    cannot do for ceil-rounded sizes. In eager mode, a ``y`` already of that size is returned
    as is (``F.pad`` would copy it).
    """
    if not torch.compiler.is_compiling() and (y.shape[-2], y.shape[-1]) == (h, w):
        return y
    return F.pad(y, (0, w - y.shape[-1], 0, h - y.shape[-2]))


def pad_to_multiple(x: Tensor, divisor: int, value: float = 0.0) -> Tensor:
    """Zero-pad the bottom/right of ``x`` so H and W are multiples of ``divisor``.

    Pixel (i, j) keeps its position, so outputs can be cropped back with ``[..., :H, :W]``.
    """
    if divisor <= 1:
        return x
    pad_h, pad_w = padding_for(x.shape[-2], x.shape[-1], divisor)
    return F.pad(x, (0, pad_w, 0, pad_h), value=value)


def resize_bilinear(x: Tensor, size) -> Tensor:
    """Bilinear resize (``align_corners=False``) that keeps ``x``'s dtype: CUDA autocast runs
    ``upsample_bilinear2d`` in float32, and concatenating that output would make the whole
    buffer float32 (27% more UPerNet training memory). The consumer is a conv that casts
    to bf16 anyway."""
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False).to(x.dtype)


def upsample2x_nearest(x: Tensor, size) -> Tensor:
    """Nearest-neighbour 2x upsampling, cropped to ``size``: ``F.interpolate(scale_factor=2)``
    (the kernel SMP's U-Net uses; its backward is a cheap reduction).

    Equal to :func:`resize_nearest` when each output dimension is ``2n`` or ``2n - 1`` for an
    input of ``n`` (``floor(i * n / (2n - 1)) == i // 2`` for ``i < 2n - 1``): a ceil-rounded
    feature pyramid, or one padded to the stride. Unlike ``mode="nearest"`` with ``size=``,
    ``scale_factor=2`` compiles correctly with ``dynamic=True``.
    """
    y = F.interpolate(x, scale_factor=2, mode="nearest")
    # autocast runs upsampling in float32; nearest only copies values, so cast back losslessly
    # (else bf16 decoders carry float32 maps, ~10% more training memory)
    return crop_to(y if y.dtype == x.dtype else y.to(x.dtype), size[0], size[1])


def is_2x(n_in, n_out) -> bool:
    """``n_out`` is ``2 n_in`` or ``2 n_in - 1`` (plain ints only: no guards on symbolic
    sizes)."""
    return isinstance(n_in, int) and isinstance(n_out, int) and 2 * n_in - 1 <= n_out <= 2 * n_in


def fit_nearest(x: Tensor, size, exact: bool = False) -> Tensor:
    """:func:`resize_nearest` to ``size`` for an ``x`` that is at most one row / column larger
    (an upsampled map against a ceil-rounded skip, or a padded output against the input): then
    nearest resizing is a crop (``floor(i * (n + 1) / n) == i`` for ``i < n``), and no gather is
    needed. ``exact``: the caller knows the sizes qualify (no check is traced under
    torch.compile); otherwise eager mode checks them and falls back to the gather."""
    if exact or (not torch.compiler.is_compiling() and _crops(x.shape[-2], size[0])
                 and _crops(x.shape[-1], size[1])):
        return crop_to(x, size[0], size[1])
    return resize_nearest(x, size)


def _crops(n_in, n_out) -> bool:
    return isinstance(n_in, int) and isinstance(n_out, int) and n_out <= n_in <= n_out + 1


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
    another size. The integer form compiles correctly, but its backward is an atomic
    scatter-add, slow in bf16: for 2x
    upsampling prefer :func:`upsample2x_nearest`.
    """
    h, w = x.shape[-2], x.shape[-1]
    x = x.index_select(-2, _nearest_index(h, size[0], x.device))
    return x.index_select(-1, _nearest_index(w, size[1], x.device))
