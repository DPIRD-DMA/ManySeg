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

    def spec(t: Tensor, prefix: str):
        if min(t.shape[-2:]) < lo:
            raise ValueError(f"example input must be at least {lo}x{lo} so no feature map is "
                             f"1x1; got {tuple(t.shape[-2:])}")
        d = {2: torch.export.Dim(f"{prefix}h", min=lo, max=max_size),
             3: torch.export.Dim(f"{prefix}w", min=lo, max=max_size)}
        if batch is not None:
            d[0] = batch
        return d

    if isinstance(example, Mapping):
        shapes, seen = {}, {}
        for k, t in example.items():
            if t.ndim == 2:  # vector input or mask: (B, N)
                shapes[k] = {0: batch} if batch is not None else None
                continue
            key = tuple(t.shape[-2:])
            if key not in seen:
                seen[key] = spec(t, f"{k}_")
            shapes[k] = seen[key]
        return {"x": shapes}
    return {"x": spec(example, "")}


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
