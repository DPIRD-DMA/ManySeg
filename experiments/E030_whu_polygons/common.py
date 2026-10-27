"""E030 shared code: cached WHU tiles, dihedral augmentation, B0 / B2 targets, tasks, loaders.

Cache (``prepare.py``), per split: ``{split}_img.npy`` (N, 512, 512, 3) uint8, ``{split}_inst.npy``
(N, 512, 512) int32 instance ids, ``{split}_poly.npz`` with ``wkb`` (one bytes blob),
``wkb_offsets`` (P + 1) and ``tile_offsets`` (N + 1): tile i owns polygons
``tile_offsets[i]:tile_offsets[i+1]``, in pixel coordinates (GIS convention).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import shapely
import torch
import torch.nn.functional as F
from shapely import affinity

from manyseg.vectors import encode_graph_tensor, graph_tensor_heads, vertex_graph

TILE = 512
SPACING = 20.0  # graph-tensor vertex spacing (px); the vector head predicts vector / SPACING
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


def load_polygons(path: Path) -> list[list]:
    z = np.load(path)
    blob, wo, to = z["wkb"].tobytes(), z["wkb_offsets"], z["tile_offsets"]
    polys = [shapely.from_wkb(blob[wo[k]:wo[k + 1]]) for k in range(len(wo) - 1)]
    return [polys[to[i]:to[i + 1]] for i in range(len(to) - 1)]


def save_polygons(path: Path, tiles: list[list]) -> None:
    parts, wo, to = [], [0], [0]
    for polys in tiles:
        for p in polys:
            b = shapely.to_wkb(p)
            parts.append(b)
            wo.append(wo[-1] + len(b))
        to.append(to[-1] + len(polys))
    np.savez_compressed(path, wkb=np.frombuffer(b"".join(parts), np.uint8),
                        wkb_offsets=np.array(wo, np.int64), tile_offsets=np.array(to, np.int64))


# --------------------------------------------------------------------------------------
# Dihedral transforms (8): k quarter turns counter-clockwise in the image, then optional hflip.
# Arrays are (H, W, ...) with row 0 at the top; polygons are (x, y) pixel coordinates.
# --------------------------------------------------------------------------------------


def dihedral_array(a: np.ndarray, d: int) -> np.ndarray:
    k, flip = d % 4, d >= 4
    a = np.rot90(a, k, axes=(0, 1))
    return np.ascontiguousarray(a[:, ::-1] if flip else a)


def dihedral_polygon(p, d: int, size: int = TILE):
    """Same transform as :func:`dihedral_array` for (x, y) geometry in a ``size`` square."""
    k, flip = d % 4, d >= 4
    # np.rot90 (counter-clockwise on the displayed image): new[r, c] = old[c, W-1-r]
    # -> a point (x, y) maps to (y, W - x)
    for _ in range(k):
        p = affinity.affine_transform(p, [0, 1, -1, 0, 0, size])  # x' = y, y' = size - x
    if flip:
        p = affinity.affine_transform(p, [-1, 0, 0, 1, size, 0])  # x' = size - x
    return p


class WHUTiles(torch.utils.data.Dataset):
    """Training / evaluation samples. ``arm`` "B0": (image, instance ids); "B2": (image, dict of
    graph-tensor targets + extent). ``augment``: a random dihedral transform per sample."""

    def __init__(self, cache: Path, split: str, arm: str, augment: bool):
        self.img = np.load(cache / f"{split}_img.npy", mmap_mode="r")
        self.inst = np.load(cache / f"{split}_inst.npy", mmap_mode="r")
        self.polys = load_polygons(cache / f"{split}_poly.npz") if arm == "B2" else None
        self.arm, self.augment = arm, augment

    def __len__(self) -> int:
        return len(self.img)

    def __getitem__(self, i):
        d = int(torch.randint(8, (1,))) if self.augment else 0
        img = dihedral_array(np.asarray(self.img[i]), d)
        inst = dihedral_array(np.asarray(self.inst[i]), d)
        x = torch.from_numpy(img).permute(2, 0, 1)  # uint8 (3, H, W)
        if self.arm == "B0":
            return x, torch.from_numpy(inst.astype(np.int64))
        polys = [dihedral_polygon(p, d) for p in self.polys[i]]
        t = encode_graph_tensor(vertex_graph(polys, SPACING, border=(TILE, TILE)), (TILE, TILE))
        y = {"vertex": torch.from_numpy(t["vertex"]),
             "edge": torch.from_numpy(t["edge"].astype(np.uint8)),
             "vector": torch.from_numpy((t["vector"] / SPACING).reshape(-1, TILE, TILE).astype(np.float16)),
             "support": torch.from_numpy(t["support"]),
             "extent": torch.from_numpy((inst > 0).astype(np.uint8))}
        return x, y


class DeviceLoader:
    """Moves batches to the device and normalises images (ImageNet statistics)."""

    def __init__(self, loader, device):
        self.loader, self.device = loader, device

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        mean, std = MEAN.to(self.device), STD.to(self.device)
        for x, y in self.loader:
            x = (x.to(self.device, non_blocking=True).float() - mean) / std
            if isinstance(y, dict):
                y = {k: v.to(self.device, non_blocking=True) for k, v in y.items()}
            else:
                y = y.to(self.device, non_blocking=True)
            yield x, y


def b2_heads() -> dict:
    return graph_tensor_heads()


class GraphTensorTask:
    """B2 training task for the bench trainer: BCE vertex (soft Gaussian target, logits) + BCE
    edge on support pixels + smooth-L1 vector (/ SPACING) where the edge target is 1 + dice-BCE
    extent. Validation reports the loss and the extent IoU.

    E032 options (off by default, so the E030 loss is unchanged):
    * ``weighted``: vertex BCE weighted 1 + 10 x target; edge BCE with pos_weight 5 (missing a
      vertex or edge costs a whole building at decode time; a false positive rarely does).
    * ``endpoint``: weight of a destination loss. At pixels within ~1.5 px of a target vertex
      centre (target vertex > 0.6) that have a true edge in sector k, the predicted vector
      (x SPACING) from the pixel centre gives an endpoint; the *target* vertex map is sampled
      there (bilinear, ``align_corners=False``, endpoints within half a pixel of the tile edge or outside it masked) and
      -log(sample + 1e-4) is averaged. Sampling the fixed target cannot be gamed by raising the
      predicted vertex map (Codex review).
    """

    name = "graph_tensor"
    primary_metric = "extent_iou"
    higher_is_better = True

    def __init__(self, weighted: bool = False, endpoint: float = 0.0):
        self.weighted, self.endpoint = weighted, endpoint

    def heads(self):
        return b2_heads()

    def targets(self, y):
        return y

    def loss(self, out, t):
        from manyseg_bench import losses

        tvx = t["vertex"].float()
        lv = F.binary_cross_entropy_with_logits(out["vertex"][:, 0].float(), tvx,
                                                weight=(1 + 10 * tvx) if self.weighted else None)
        sup = t["support"].bool()
        if sup.any():
            e = out["edge"].float().permute(0, 2, 3, 1)[sup]  # (P, S)
            pw = torch.full((e.shape[1],), 5.0, device=e.device) if self.weighted else None
            le = F.binary_cross_entropy_with_logits(e, t["edge"].permute(0, 2, 3, 1)[sup].float(),
                                                    pos_weight=pw)
            S = t["edge"].shape[1]
            on = t["edge"].bool()  # (B, S, H, W)
            pv = out["vector"].float().view(out["vector"].shape[0], S, 2, *out["vector"].shape[2:])
            tv = t["vector"].float().view_as(pv)
            m = on[:, :, None].expand_as(pv)
            lvec = F.smooth_l1_loss(pv[m], tv[m]) if m.any() else pv.sum() * 0
        else:
            le = lvec = out["edge"].sum() * 0
        lx = losses.dice_bce(out["extent"], t["extent"][:, None].float())
        total = lv + le + lvec + lx
        if self.endpoint > 0:
            total = total + self.endpoint * self.endpoint_loss(out, t)
        return total

    @staticmethod
    def endpoint_loss(out, t):
        tv = t["vertex"].float()  # (B, H, W) target vertex map, fixed
        B, H, W = tv.shape
        S = t["edge"].shape[1]
        m = (tv > 0.6)[:, None] & t["edge"].bool()  # (B, S, H, W): near a vertex centre, true edge
        if not m.any():
            return out["vector"].sum() * 0
        v = out["vector"].float().view(B, S, 2, H, W) * SPACING  # pixels
        ys, xs = torch.meshgrid(torch.arange(H, device=tv.device) + 0.5,
                                torch.arange(W, device=tv.device) + 0.5, indexing="ij")
        ex, ey = xs + v[:, :, 0], ys + v[:, :, 1]  # (B, S, H, W) endpoints, geometric (x, y)
        # fully inside: with align_corners=False a point within half a pixel of the edge samples
        # the zero padding (Codex review), so those endpoints are masked
        inside = (ex > 0.5) & (ex < W - 0.5) & (ey > 0.5) & (ey < H - 0.5)
        grid = torch.stack([2 * ex / W - 1, 2 * ey / H - 1], -1).view(B, S * H, W, 2)
        samp = F.grid_sample(tv[:, None], grid, mode="bilinear", padding_mode="zeros",
                             align_corners=False).view(B, S, H, W)
        sel = m & inside
        if not sel.any():
            return out["vector"].sum() * 0
        return -torch.log(samp[sel] + 1e-4).mean()

    def evaluator(self):
        return _GraphTensorEval(self)


class _GraphTensorEval:
    def __init__(self, task):
        self.t, self.loss, self.n, self.inter, self.union = task, 0.0, 0, 0, 0

    def update(self, out, y):
        n = y["extent"].shape[0]
        self.loss += float(self.t.loss(out, y)) * n
        self.n += n
        p, g = out["extent"][:, 0] > 0, y["extent"].bool()
        self.inter += int((p & g).sum())
        self.union += int((p | g).sum())

    def compute(self):
        return {"loss": self.loss / max(1, self.n), "extent_iou": self.inter / max(1, self.union)}


def b2_probabilities(out: dict, i: int) -> dict[str, np.ndarray]:
    """One sample's decoder inputs as float32 numpy: probabilities and vectors in pixels."""
    S = out["edge"].shape[1]
    return {"vertex": torch.sigmoid(out["vertex"][i, 0].float()).cpu().numpy(),
            "edge": torch.sigmoid(out["edge"][i].float()).cpu().numpy(),
            "vector": (out["vector"][i].float() * SPACING).view(S, 2, TILE, TILE).cpu().numpy(),
            "extent": torch.sigmoid(out["extent"][i, 0].float()).cpu().numpy()}


def b0_probabilities(out: dict, i: int) -> dict[str, np.ndarray]:
    return {"extent": torch.sigmoid(out["extent"][i, 0].float()).cpu().numpy(),
            "boundary": torch.sigmoid(out["boundary"][i, 0].float()).cpu().numpy(),
            "distance": out["distance"][i, 0].float().cpu().numpy()}


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


__all__ = ["DeviceLoader", "GraphTensorTask", "WHUTiles", "b0_probabilities", "b2_heads",
           "b2_probabilities", "dihedral_array", "dihedral_polygon", "load_polygons",
           "save_polygons", "seed_everything"]
