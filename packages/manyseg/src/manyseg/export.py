"""Export helpers (deployment is kept separate from model construction)."""

from __future__ import annotations

import contextlib
from collections.abc import Mapping

import torch
from torch import Tensor, nn


def _min_size(model: nn.Module) -> int:
    """A safe lower bound on H/W that keeps every feature map larger than 1x1 (torch.export
    specialises size-1 dimensions). Padded to the network's deepest stride, ``stride + 1``
    gives two cells at the deepest level; padded less (``pad_to_stride=False``, or a
    multi-input model whose grid multiple stops at the returned levels), floor-rounding layers
    such as a 4x4 / stride 4 patchify or PVT's spatial-reduction attention need
    ``2 * stride``. Conservative: some backbones work below it."""
    bb = getattr(model, "backbone", None)
    # every encoder of a multi-input model, at the deepest stride its network computes
    mods = list(bb.modules()) if isinstance(bb, nn.Module) else []
    stride = max([s.stride for s in getattr(bb, "feature_specs", ())]
                 + [s.stride for m in mods for s in getattr(m, "feature_specs", ())]
                 + [int(getattr(m, "network_stride", 1)) for m in mods], default=1)
    core = getattr(model, "model", model)  # Embedder wraps the model
    if hasattr(bb, "divisors"):  # multi-input: each input padded to its own multiple
        pad = min(getattr(bb, "grid_divisor", 1), *bb.divisors.values())
    else:
        pad = int(getattr(core, "divisor", 1)) if getattr(core, "pad_to_stride", True) else 1
    return stride + 1 if pad >= stride else 2 * stride


@contextlib.contextmanager
def eval_mode(model: nn.Module):
    """Run the block with ``model`` in eval mode, then restore every submodule's own
    train/eval flag (so e.g. BatchNorm frozen with ``.eval()`` during training stays frozen)."""
    modes = [(m, m.training) for m in model.modules()]
    model.eval()
    try:
        yield model
    finally:
        for m, training in modes:
            m.training = training


# timm's window-attention classes, by (module, class): their dynamic-size export fails inside
# timm (older torch versions hung instead), so they are refused up front with a clear message.
# Matched exactly so that an unrelated module with a ``window_size`` attribute is not rejected.
_WINDOW_ATTENTION = {
    ("timm.models.swin_transformer", "WindowAttention"),
    ("timm.models.swin_transformer_v2", "WindowAttention"),
    ("timm.models.swin_transformer_v2_cr", "WindowMultiHeadAttention"),
    ("timm.models.davit", "WindowAttention"),
}


def _windowed(model: nn.Module) -> str | None:
    """Class name of the first timm window-attention module (Swin, Swin V2, DaViT), if any."""
    return next((type(m).__name__ for m in model.modules()
                 if (type(m).__module__, type(m).__name__) in _WINDOW_ATTENTION), None)


def _checked_shapes(model: nn.Module, example, dynamic: bool, max_size: int,
                    dynamic_batch: bool):
    """:func:`dynamic_shapes`, after refusing dynamic H/W for window-attention backbones,
    whose dynamic-size export fails inside timm."""
    if dynamic and (block := _windowed(model)) is not None:
        raise ValueError(f"window-attention backbones ({block}) cannot be exported with dynamic "
                         "H/W; pass dynamic=False to fix H/W at the example's size")
    return dynamic_shapes(model, example, dynamic=dynamic, max_size=max_size,
                          dynamic_batch=dynamic_batch)


def _export(module: nn.Module, example, shapes, dynamic: bool = True
            ) -> torch.export.ExportedProgram:
    """``torch.export`` with deferred runtime asserts, naming the argument that avoids a failed
    dynamic-shape export."""
    try:
        return torch.export.export(module, (example,), dynamic_shapes=shapes,
                                   prefer_deferred_runtime_asserts_over_guards=True)
    except Exception as e:  # torch.export raises several types (UserError, RuntimeError, ...)
        if shapes is None:
            raise
        what, fix = (("dynamic H/W", "pass dynamic=False to fix H/W at the example's size")
                     if dynamic else
                     ("a dynamic batch", "pass dynamic_batch=False to fix the batch at the "
                      "example's size"))
        raise ValueError(f"could not export this model with {what}; {fix}. Export error: "
                         f"{type(e).__name__}: {e}") from e


def dynamic_shapes(model: nn.Module, example, *, dynamic: bool = True, max_size: int = 4096,
                   dynamic_batch: bool = False):
    """``dynamic_shapes`` for ``torch.export``: H and W dynamic in [min, max_size] (all inputs
    of a multi-input model share the H/W symbols only if they are on the same grid).

    Vector (metadata) inputs and their masks, ``(B, N)``, have no spatial dimensions: only the
    batch dimension is dynamic (with ``dynamic_batch``); ``N`` is fixed by the model.

    ``dynamic=False`` keeps H/W (and a time series' number of dates) at the example's, for
    models that cannot export dynamic sizes; with ``dynamic_batch`` only the batch is then
    dynamic, else the result is None (everything fixed). The arguments mean what they mean
    for :func:`export_model` and :func:`manyseg.quantize.quantize_pt2e`."""
    if not dynamic and not dynamic_batch:
        return None
    lo = _min_size(model)
    batch = torch.export.Dim("batch", min=1, max=1024) if dynamic_batch else None
    if batch is not None:
        b = next(iter(example.values())) if isinstance(example, Mapping) else example
        if b.shape[0] < 2:
            raise ValueError("dynamic_batch=True needs an example with batch size >= 2 "
                             "(torch.export specialises size-1 dimensions); got batch 1")

    def hw(t: Tensor, prefix: str):
        if not dynamic:
            return None, None
        if min(t.shape[-2:]) < lo:
            raise ValueError(f"example input must be at least {lo}x{lo} so no feature map is "
                             f"1x1; got {tuple(t.shape[-2:])}")
        return (torch.export.Dim(f"{prefix}h", min=lo, max=max_size),
                torch.export.Dim(f"{prefix}w", min=lo, max=max_size))

    def dims(t: Tensor, h, w, time=None):
        d = {t.ndim - 2: h, t.ndim - 1: w} if h is not None else {}
        if time is not None:
            d[1] = time
        if batch is not None:
            d[0] = batch
        return d

    if not isinstance(example, Mapping):
        return {"x": dims(example, *hw(example, ""))}
    bb = getattr(model, "backbone", None)
    encoders = getattr(bb, "encoders", {})
    alignment = getattr(bb, "alignment", {})
    # Time series (B, T, C, H, W): T is dynamic too, shared with "<name>_mask" / "<name>_dates",
    # unless the aggregation needs an exact number of dates ("concat", "diff").
    times = {}
    for k, t in example.items():
        if t.ndim == 5 and dynamic:
            fixed = getattr(encoders[k], "fixed_dates", None) if k in encoders else None
            if fixed is not None:
                continue
            if t.shape[1] < 2:
                raise ValueError(f"example time series {k!r} needs at least 2 dates "
                                 "(torch.export specialises size-1 dimensions)")
            times[k] = torch.export.Dim(f"{k}_t", min=1, max=1024)
    shapes, grids = {}, {}
    for k, t in example.items():
        if t.ndim == 2:  # vector input or its mask (B, N); a time series' mask / dates (B, T)
            base = k.rsplit("_", 1)[0] if k.endswith(("_mask", "_dates")) else None
            d = {1: times[base]} if base in times else {}
            if batch is not None:
                d[0] = batch
            shapes[k] = d or None
            continue
        # same-grid inputs share their H/W symbols; a same_extent input always has its own,
        # even when the example happens to match another input's size
        grid = (k,) if alignment.get(k) == "same_extent" else tuple(t.shape[-2:])
        if grid not in grids:
            grids[grid] = hw(t, f"{k}_")
        shapes[k] = dims(t, *grids[grid], times.get(k))
    return {"x": shapes}


def export_model(model: nn.Module, example, *, dynamic: bool = True, max_size: int = 4096,
                 dynamic_batch: bool = False) -> torch.export.ExportedProgram:
    """``torch.export`` with dynamic H/W in ``[min, max_size]`` (and optionally batch;
    ``dynamic_batch`` needs an example with batch size >= 2). ``model`` is exported in eval mode
    and then returned to its previous mode.

    Window-attention backbones (Swin, DaViT) cannot export dynamic H/W and need
    ``dynamic=False``: the program then only accepts the example's H/W, and its batch size
    too unless ``dynamic_batch=True`` (dynamic batch, fixed H/W). The arguments mean the same
    as for :func:`manyseg.quantize.quantize_pt2e`.

    Uses deferred runtime asserts: some timm blocks (ConvNeXt/MetaFormer layer norms, Conv2dSame)
    produce guards that are always true but that export's solver cannot prove, e.g.
    ``1 % (512 * ceil(w / 32)) != 0``.
    """
    shapes = _checked_shapes(model, example, dynamic, max_size, dynamic_batch)
    with eval_mode(model):
        return _export(model, example, shapes, dynamic)
