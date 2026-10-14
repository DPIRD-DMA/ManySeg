"""Instances from boundary + distance outputs (runs outside the exported model graph).

Pipeline: foreground = extent > t_e; seeds = foreground & boundary < t_b & distance > t_s;
markers = connected components of the seeds; then markers grow back over the foreground:
``backend="skimage"`` uses seeded watershed on ``-distance`` (``pip install manyseg[instances]``);
``backend="torch"`` (no extra dependencies, GPU-capable) grows all markers at equal speed through
4-connected foreground (ties -> larger label), which splits a symmetric boundary band exactly.

Scaled heads (head ``scale`` < 1): the label map comes back on the head grid
(``ceil(H * scale) x ceil(W * scale)``); map it to the input grid with
``manyseg.utils.upsample_output(labels, (H, W), scale, mode="nearest")``. ``min_size`` is in
head pixels.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _neighbour_max(lab: Tensor) -> Tensor:
    p = torch.nn.functional.pad(lab, (1, 1, 1, 1))
    return torch.stack([p[:, :-2, 1:-1], p[:, 2:, 1:-1], p[:, 1:-1, :-2], p[:, 1:-1, 2:]]).amax(0)


@torch.no_grad()
def connected_components(mask: Tensor, check_every: int = 8) -> Tensor:
    """4-connected component labels for a bool ``(B, H, W)`` mask (0 = background). Torch only;
    labels are not sequential (see :func:`relabel_sequential`)."""
    b, h, w = mask.shape
    lab = torch.arange(1, h * w + 1, device=mask.device).view(1, h, w).expand(b, h, w)
    lab = torch.where(mask, lab, 0)
    it = 0
    while True:
        prev = lab
        for _ in range(check_every):
            lab = torch.where(mask, torch.maximum(lab, _neighbour_max(lab)), 0)
        it += check_every
        if torch.equal(lab, prev) or it > h * w:
            return lab


@torch.no_grad()
def grow_labels(markers: Tensor, region: Tensor) -> Tensor:
    """Grow labelled markers into unlabelled ``region`` pixels at equal speed (4-connectivity)."""
    lab = markers.clone()
    while True:
        cand = _neighbour_max(lab)
        new = region & (lab == 0) & (cand > 0)
        if not bool(new.any()):
            return lab
        lab = torch.where(new, cand, lab)


def relabel_sequential(labels: Tensor) -> Tensor:
    """Map ids to 1..K per image, keeping 0 as background."""
    out = torch.zeros_like(labels)
    for i in range(labels.shape[0] if labels.dim() == 3 else 1):
        li = labels[i] if labels.dim() == 3 else labels
        u, inv = torch.unique(li, return_inverse=True)
        seq = inv + (0 if bool((u == 0).any()) else 1)
        if labels.dim() == 3:
            out[i] = seq
        else:
            out = seq
    return out


def _ensure_seed_per_component(fg: Tensor, seeds: Tensor, dist: Tensor) -> Tensor:
    """Give every foreground component without a seed one seed at its distance maximum.

    Objects that lie entirely inside the boundary band (tiny or thin instances) would otherwise
    vanish. Touching objects that are *both* entirely boundary still merge (inherent to the
    representation).
    """
    comp = connected_components(fg)
    b = fg.shape[0]
    n = int(comp.max()) + 1
    flat = comp + torch.arange(b, device=fg.device)[:, None, None] * n
    flat = flat.reshape(-1)
    has_seed = torch.zeros(b * n, dtype=torch.bool, device=fg.device)
    has_seed[flat[seeds.reshape(-1)]] = True
    d = torch.where(fg, dist, torch.full_like(dist, -1.0)).reshape(-1)
    dmax = torch.full((b * n,), -1.0, device=fg.device).scatter_reduce(0, flat, d, "amax")
    at_max = (d == dmax[flat]) & fg.reshape(-1) & ~has_seed[flat]
    pix = torch.arange(flat.numel(), device=fg.device)
    first = torch.full((b * n,), flat.numel(), device=fg.device).scatter_reduce(
        0, flat[at_max], pix[at_max], "amin")
    extra = torch.zeros_like(at_max)
    extra[first[first < flat.numel()]] = True
    return seeds | extra.view_as(seeds)


def _foreground(extent: Tensor, logits: bool, threshold: float) -> Tensor:
    if extent.dtype in (torch.long, torch.int32, torch.int64, torch.uint8, torch.bool):
        return (extent if extent.dim() == 3 else extent[:, 0]) > 0
    if extent.shape[1] == 1:
        p = extent[:, 0].sigmoid() if logits else extent[:, 0]
        return p > threshold
    return extent.argmax(1) > 0  # multi-class logits/probs with background at index 0


@torch.no_grad()
def instances_from_outputs(
    extent: Tensor,
    boundary: Tensor,
    distance: Tensor,
    *,
    logits: bool = False,
    extent_threshold: float = 0.5,
    boundary_threshold: float = 0.5,
    seed_threshold: float = 0.25,
    min_size: int = 0,
    backend: str = "auto",
) -> Tensor:
    """Integer instance label map ``(B, H, W)`` (0 = background, ids 1..K per image).

    Args:
        extent: ``(B, 1, H, W)`` probabilities/logits, ``(B, C, H, W)`` multi-class scores with
            background at index 0, or integer labels.
        boundary: ``(B, 1, H, W)`` probabilities (or logits with ``logits=True``).
        distance: ``(B, 1, H, W)`` predicted (normalised) distance.
        backend: ``"skimage"`` (watershed), ``"torch"`` or ``"auto"`` (skimage if installed).
    """
    fg = _foreground(extent, logits, extent_threshold)
    bnd = boundary[:, 0].sigmoid() if logits else boundary[:, 0]
    dist = distance[:, 0].float()
    seeds = fg & (bnd < boundary_threshold) & (dist > seed_threshold)
    seeds = _ensure_seed_per_component(fg, seeds, dist)
    markers = connected_components(seeds)
    if backend == "auto":
        try:
            import skimage  # noqa: F401

            backend = "skimage"
        except ImportError:
            backend = "torch"
    if backend == "skimage":
        from skimage.segmentation import watershed

        out = torch.stack([
            torch.from_numpy(watershed(-dist[i].cpu().numpy(), markers[i].cpu().numpy(),
                                       mask=fg[i].cpu().numpy())).to(markers)
            for i in range(markers.shape[0])
        ])
    elif backend == "torch":
        out = grow_labels(markers, fg)
    else:
        raise ValueError(f"backend must be auto|torch|skimage, got {backend!r}")
    if min_size > 0:
        for i in range(out.shape[0]):
            ids, counts = torch.unique(out[i], return_counts=True)
            small = ids[(counts < min_size) & (ids > 0)]
            out[i][torch.isin(out[i], small)] = 0
    return relabel_sequential(out)
