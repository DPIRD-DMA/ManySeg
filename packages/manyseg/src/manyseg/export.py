"""Export helpers (deployment is kept separate from model construction)."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn


def _min_size(model: nn.Module) -> int:
    """Smallest H/W that keeps every feature map larger than 1x1 (torch.export specialises
    size-1 dimensions)."""
    bb = getattr(model, "backbone", None)
    specs = getattr(bb, "feature_specs", ())
    return max((s.stride for s in specs), default=1) + 1


def dynamic_shapes(model: nn.Module, example, max_size: int = 4096,
                   dynamic_batch: bool = False):
    """``dynamic_shapes`` for ``torch.export``: H and W dynamic in [min, max_size] (all inputs
    of a multi-input model share the H/W symbols only if they are on the same grid).

    Vector (metadata) inputs and their masks, ``(B, N)``, have no spatial dimensions: only the
    batch dimension is dynamic (with ``dynamic_batch``); ``N`` is fixed by the model."""
    lo = _min_size(model)
    batch = torch.export.Dim("batch", min=1, max=1024) if dynamic_batch else None

    def hw(t: Tensor, prefix: str):
        if min(t.shape[-2:]) < lo:
            raise ValueError(f"example input must be at least {lo}x{lo} so no feature map is "
                             f"1x1; got {tuple(t.shape[-2:])}")
        return (torch.export.Dim(f"{prefix}h", min=lo, max=max_size),
                torch.export.Dim(f"{prefix}w", min=lo, max=max_size))

    def dims(t: Tensor, h, w, time=None):
        d = {t.ndim - 2: h, t.ndim - 1: w}
        if time is not None:
            d[1] = time
        if batch is not None:
            d[0] = batch
        return d

    if not isinstance(example, Mapping):
        return {"x": dims(example, *hw(example, ""))}
    # Time series (B, T, C, H, W): T is dynamic too, shared with "<name>_mask" / "<name>_dates".
    times = {}
    for k, t in example.items():
        if t.ndim == 5:
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
        grid = tuple(t.shape[-2:])  # same-grid inputs share their H/W symbols
        if grid not in grids:
            grids[grid] = hw(t, f"{k}_")
        shapes[k] = dims(t, *grids[grid], times.get(k))
    return {"x": shapes}


def export_model(model: nn.Module, example, *, max_size: int = 4096,
                 dynamic_batch: bool = False) -> torch.export.ExportedProgram:
    """``torch.export`` with dynamic H/W (and optionally batch).

    Uses deferred runtime asserts: some timm blocks (ConvNeXt/MetaFormer layer norms, Conv2dSame)
    produce guards that are always true but that export's solver cannot prove, e.g.
    ``1 % (512 * ceil(w / 32)) != 0``.
    """
    model = model.eval()
    return torch.export.export(
        model, (example,), dynamic_shapes=dynamic_shapes(model, example, max_size, dynamic_batch),
        prefer_deferred_runtime_asserts_over_guards=True,
    )
