"""Instances from boundary + distance outputs (runs outside the exported model graph).

Pipeline: foreground = extent > t_e; seeds = foreground & boundary < t_b & distance > t_s;
markers = connected components of the seeds; then markers grow back over the foreground:
``backend="skimage"`` uses seeded watershed on ``-distance``, markers of equal distance leaving
the queue in raster order (``pip install manyseg[instances]``);
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
def connected_components(mask: Tensor, check_every: int = 8, *,
                         origin: tuple[int, int] | None = None,
                         scene_width: int | None = None) -> Tensor:
    """4-connected component labels for a bool ``(B, H, W)`` mask (0 = background). Torch only;
    labels are not sequential (see :func:`relabel_sequential`).

    ``origin`` = (row, col) of the mask in a larger scene of width ``scene_width``: labels are
    then scene pixel indices (``(row + r) * scene_width + col + c + 1``) instead of array ones,
    so a window labels and orders its components as a decode of the whole scene would.

    Each pixel starts with its own index and takes the largest label among its neighbours
    until nothing changes: every component ends with its largest pixel index. Pointer
    jumping (a pixel also takes the label of the pixel its label names, which is in the same
    component) lets long components converge in far fewer sweeps (18-44x fewer on long or
    field-like components) with the same result. On CPU with scipy installed, scipy's labelling
    is mapped to the same ids (much faster on large components: 102 sweeps, ~9 s, for one 1024 px
    tile of farm fields)."""
    lab = None
    if mask.device.type == "cpu":
        try:
            from scipy import ndimage
        except ImportError:
            ndimage = None
        if ndimage is not None:
            lab = _connected_components_scipy(mask, ndimage)
    if lab is None:
        lab = _connected_components_torch(mask, check_every)
    return lab if origin is None else _scene_index(lab, origin, scene_width)


def _scene_index(lab: Tensor, origin: tuple[int, int], scene_width: int | None) -> Tensor:
    """Array pixel index + 1 -> scene pixel index + 1 (monotone: raster order is kept)."""
    w = lab.shape[-1]
    if scene_width is None or scene_width < origin[1] + w:
        raise ValueError(f"scene_width {scene_width} must cover origin {origin} + width {w}")
    p = lab.long() - 1
    out = (origin[0] + p // w) * scene_width + origin[1] + p % w + 1
    return torch.where(lab > 0, out, 0)


def anchor_labels(labels: Tensor, origin: tuple[int, int] = (0, 0),
                  scene_width: int | None = None) -> Tensor:
    """Relabel each instance of ``(B, H, W)`` (or ``(H, W)``) ``labels`` with its *anchor* + 1:
    the scene pixel index of its first pixel in raster order (``origin`` / ``scene_width`` as in
    :func:`connected_components`; default: the array is the scene). The ids do not depend on
    the window a scene was decoded in. int64, 0 stays background."""
    import numpy as np

    one = labels.dim() == 2
    lab = (labels[None] if one else labels).cpu().numpy()
    b, h, w = lab.shape
    if scene_width is None and tuple(origin) != (0, 0):
        raise ValueError("anchor_labels needs scene_width with a non-zero origin")
    sw = w if scene_width is None else scene_width
    out = np.zeros(lab.shape, np.int64)
    for i in range(b):
        u, first = np.unique(lab[i].reshape(-1), return_index=True)  # first = raster order
        r, c = first // w, first % w
        ids = (origin[0] + r) * sw + origin[1] + c + 1
        ids[u == 0] = 0
        out[i] = ids[np.searchsorted(u, lab[i])]
    t = torch.from_numpy(out)
    return t[0] if one else t


def _connected_components_scipy(mask: Tensor, ndimage) -> Tensor:
    import numpy as np

    b, h, w = mask.shape
    out = np.zeros((b, h, w), np.int64)
    idx = np.arange(1, h * w + 1, dtype=np.int64).reshape(h, w)
    for i in range(b):
        lab, n = ndimage.label(mask[i].numpy())  # 4-connectivity
        if n:
            last = np.zeros(n + 1, np.int64)
            np.maximum.at(last, lab.reshape(-1), idx.reshape(-1))
            out[i] = np.where(lab > 0, last[lab], 0)
    return torch.from_numpy(out)


def _connected_components_torch(mask: Tensor, check_every: int = 8) -> Tensor:
    b, h, w = mask.shape
    lab = torch.arange(1, h * w + 1, device=mask.device).view(1, h, w).expand(b, h, w)
    lab = torch.where(mask, lab, 0)
    it = 0
    while True:
        prev = lab
        for _ in range(check_every):
            lab = torch.where(mask, torch.maximum(lab, _neighbour_max(lab)), 0)
            flat = lab.reshape(b, -1)
            jumped = flat.gather(1, (flat - 1).clamp(min=0)).view(b, h, w)
            lab = torch.where(mask, torch.maximum(lab, jumped), 0)
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


def _marker_order(image, markers):
    """``image`` as float64 with marker pixels lowered by a few float64 ulps, more for pixels
    earlier in raster order. skimage's watershed starts all markers at age 0, so it orders
    equal-valued markers by heap internals that depend on the whole array; with this
    nudge they leave the queue in raster order, the same in a window as in the whole scene.
    The nudge stays below a quarter of a float32 ulp, so no other order changes (``image`` is
    float32 here) and plateaus still fill first in, first out."""
    import numpy as np

    out = image.astype(np.float64)
    idx = np.flatnonzero(markers)
    if len(idx) >= 1 << 26:
        raise ValueError("too many marker pixels for the raster-order nudge")
    v = out.reshape(-1)[idx]
    out.reshape(-1)[idx] = v - np.spacing(np.abs(v)) * (len(idx) - np.arange(len(idx)))
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


def _drop_small_seeds(seeds: Tensor, min_seed: int) -> Tensor:
    """Seeds without their 4-connected components of fewer than ``min_seed`` pixels (counted
    per image, so the result does not depend on the rest of the batch).

    Such fragments sit where boundary bands meet (the tip of an acute corner, a pixel on the
    seed threshold's edge) and would each grow into an extra instance; the instance around them
    floods their pixels instead. A foreground component left without any seed gets one back
    from :func:`_ensure_seed_per_component`, so small objects are kept.
    """
    comp = connected_components(seeds)  # labels restart in every image: offset them per image
    n = int(comp.max()) + 1
    flat = comp + torch.arange(comp.shape[0], device=comp.device)[:, None, None] * n
    _, inv, counts = torch.unique(flat, return_inverse=True, return_counts=True)
    return seeds & (counts[inv] >= min_seed)


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
    logits: bool,
    extent_threshold: float = 0.5,
    boundary_threshold: float = 0.5,
    seed_threshold: float = 0.25,
    min_seed: int = 0,
    min_size: int = 0,
    backend: str = "auto",
    origin: tuple[int, int] | None = None,
    scene_width: int | None = None,
    ids: str = "sequential",
) -> Tensor:
    """Integer instance label map ``(B, H, W)`` (0 = background, ids 1..K per image).

    Args:
        extent: ``(B, 1, H, W)`` probabilities/logits, ``(B, C, H, W)`` multi-class scores with
            background at index 0, or integer labels.
        boundary: ``(B, 1, H, W)`` probabilities (or logits with ``logits=True``).
        distance: ``(B, 1, H, W)`` predicted (normalised) distance. Non-finite values (NaN,
            +-inf) count as 0 for both seeding and growth (as ``DensityBlockDecoder`` counts
            NaN density as 0): such a pixel is not a seed (with ``seed_threshold >= 0``) but
            is still part of its instance. A NaN pixel of a 1-channel ``extent`` is
            background, and of ``boundary`` not a seed (NaN compares False).
        logits: ``True`` if ``extent`` / ``boundary`` are raw model outputs (logits, as heads
            return them), ``False`` if they are already probabilities.
        extent_threshold: Foreground threshold on the extent probability (1-channel extent
            only; a multi-class extent is foreground where its argmax is not 0).
        boundary_threshold: Pixels with a boundary probability at or above this are not seeds.
        seed_threshold: Seeds need a predicted distance above this.
        min_seed: Seed fragments (4-connected) of fewer pixels are dropped, so they do not
            become extra instances (0 keeps all). A foreground component whose seeds are
            all dropped still gets one, at its distance maximum.
        min_size: Instances smaller than this many pixels are removed (0 keeps all).
        backend: ``"skimage"`` (watershed), ``"torch"`` or ``"auto"`` (skimage if installed).
        origin: ``(row, col)`` of these arrays in a larger scene of width ``scene_width``
            (block-wise decoding). Tie-breaks then follow scene pixel indices, so an instance
            that lies well inside the window gets the same pixels as in a whole-scene decode.
        scene_width: Width of that scene (required with ``origin``).
        ids: ``"sequential"`` (1..K per image) or ``"anchor"``: anchor + 1, the scene pixel
            index of the instance's first pixel in raster order (:func:`anchor_labels`), the
            same id whatever window the scene was decoded in.
    """
    if ids not in ("sequential", "anchor"):
        raise ValueError(f"ids must be sequential|anchor, got {ids!r}")
    fg = _foreground(extent, logits, extent_threshold)
    bnd = boundary[:, 0].sigmoid() if logits else boundary[:, 0]
    # Non-finite distances (NaN nodata, a diverged head) count as 0, before both the seed
    # choice and the growth: a NaN would otherwise poison its component's distance maximum
    # (deleting a seedless object) and the watershed's priority queue.
    dist = torch.nan_to_num(distance[:, 0].float(), nan=0.0, posinf=0.0, neginf=0.0)
    if fg.numel() == 0:  # empty batch (or tile): nothing to label
        return torch.zeros(fg.shape, dtype=torch.long, device=fg.device)
    seeds = fg & (bnd < boundary_threshold) & (dist > seed_threshold)
    if min_seed > 1:
        seeds = _drop_small_seeds(seeds, min_seed)
    seeds = _ensure_seed_per_component(fg, seeds, dist)
    markers = connected_components(seeds, origin=origin, scene_width=scene_width)
    if backend == "auto":
        try:
            import skimage  # noqa: F401

            backend = "skimage"
        except ImportError:
            backend = "torch"
    if backend == "skimage":
        try:
            from skimage.segmentation import watershed
        except ImportError as e:
            raise ImportError("backend='skimage' needs the 'instances' extra: "
                              "pip install 'manyseg[instances]'") from e

        out = torch.stack([
            torch.from_numpy(watershed(_marker_order(-dist[i].cpu().numpy(), mk), mk,
                                       mask=fg[i].cpu().numpy())).to(markers)
            for i in range(markers.shape[0])
            for mk in [markers[i].cpu().numpy()]
        ])
    elif backend == "torch":
        out = grow_labels(markers, fg)
    else:
        raise ValueError(f"backend must be auto|torch|skimage, got {backend!r}")
    if min_size > 0:
        for i in range(out.shape[0]):
            u, counts = torch.unique(out[i], return_counts=True)
            small = u[(counts < min_size) & (u > 0)]
            out[i][torch.isin(out[i], small)] = 0
    if ids == "anchor":
        return anchor_labels(out, origin or (0, 0), scene_width)
    return relabel_sequential(out)
