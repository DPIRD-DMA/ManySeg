"""Block-wise vector decoding of large scenes (experimental; design note ManySeg-e028/notes/
tile_edges_manypred.md, option (c)).

Dense fields (already stitched, e.g. by ManyPred) are decoded in windows = core + halo; the cores
partition the scene. An object inside one core that does not touch its window edge is vectorised
by that block alone. Larger objects are matched across blocks on the area their windows share
and either taken from a block that sees them whole or rebuilt from each block's core pixels, then
vectorised once. Objects are keyed by their anchor (first scene pixel in raster order), so the
output does not depend on the block grid, provided the window decoder labels an object the same
way as a global decode (true when its tie-breaks follow raster order and its reach is below the
halo; tested for connected components and ``instances_from_outputs``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class Block:
    core: tuple[slice, slice]
    window: tuple[slice, slice]


def block_grid(
    shape: tuple[int, int], block: int, halo: int, offset: tuple[int, int] = (0, 0)
) -> list[Block]:
    """Cores tile the scene (the first row / column of cores may be shortened by ``offset``);
    windows extend each core by ``halo`` px, clipped to the scene."""
    H, W = shape

    def cuts(n, o):
        o = o % block
        edges = [0] + list(range(o if o else block, n, block)) + [n]
        return sorted(set(edges))

    out = []
    rs, cs = cuts(H, offset[0]), cuts(W, offset[1])
    for r0, r1 in zip(rs[:-1], rs[1:]):
        for c0, c1 in zip(cs[:-1], cs[1:]):
            out.append(
                Block(
                    (slice(r0, r1), slice(c0, c1)),
                    (
                        slice(max(r0 - halo, 0), min(r1 + halo, H)),
                        slice(max(c0 - halo, 0), min(c1 + halo, W)),
                    ),
                )
            )
    return out


@dataclass
class _Piece:
    """One block's view of an object that crosses its core boundary."""

    block: int
    bbox: tuple[slice, slice]  # scene coords of ``mask``
    mask: np.ndarray  # bool, bbox-sized
    core: tuple[slice, slice]
    window: tuple[slice, slice]
    complete: bool  # does not touch the window edge (scene edges excepted)


def _crop(fields, sl):
    return {
        k: (v[..., sl[0], sl[1]] if v.ndim == 3 else v[sl[0], sl[1]]) for k, v in fields.items()
    }


def _inter(a: tuple[slice, slice], b: tuple[slice, slice]):
    rr = slice(max(a[0].start, b[0].start), min(a[0].stop, b[0].stop))
    cc = slice(max(a[1].start, b[1].start), min(a[1].stop, b[1].stop))
    return (rr, cc) if rr.start < rr.stop and cc.start < cc.stop else None


def _sub(m: np.ndarray, bbox, region):
    return m[
        region[0].start - bbox[0].start : region[0].stop - bbox[0].start,
        region[1].start - bbox[1].start : region[1].stop - bbox[1].start,
    ]


def _anchor(mask: np.ndarray, r0: int, c0: int, W: int) -> int:
    fr = int(np.argmax(mask.reshape(-1)))  # first pixel in raster order
    return (r0 + fr // mask.shape[1]) * W + (c0 + fr % mask.shape[1])


def decode_blocks(
    fields: dict,
    decode_labels,
    vectorise,
    block: int = 2048,
    halo: int = 256,
    offset: tuple[int, int] = (0, 0),
    min_overlap: float = 0.5,
) -> dict:
    """Decode a large scene block by block.

    ``fields``: name -> (H, W) or (C, H, W) array (numpy or memmap), already stitched.
    ``decode_labels(window_fields) -> (h, w) int labels`` (0 = background) for one window.
    ``vectorise(mask, fields_crop, origin) -> geometry`` for one object's boolean ``mask``,
    ``origin`` = (row, col) of the mask in the scene; geometry in scene pixel coords.

    An object whose bbox lies inside one block's core and that does not touch the window edge is
    vectorised by that block. Every other object becomes a *piece* in each block whose core it
    reaches. Pieces from different blocks are the same object if their masks agree on the shared
    area (overlap coefficient >= ``min_overlap``). A group that has a complete piece uses it;
    otherwise its mask is the union of each piece's core pixels (cores partition the scene).
    Returns {anchor (scene pixel index of the object's first pixel): geometry} plus ``"_stats"``.
    """
    if halo < 1:
        raise ValueError("halo must be >= 1: pieces are matched on the area their windows share")
    any_field = next(iter(fields.values()))
    H, W = any_field.shape[-2:]
    out: dict = {}
    pieces: list[_Piece] = []
    blocks = block_grid((H, W), block, halo, offset)
    for bi, b in enumerate(blocks):
        wf = _crop(fields, b.window)
        lab = decode_labels(wf)
        r0, c0 = b.window[0].start, b.window[1].start
        wh, ww = lab.shape
        cr = (
            slice(b.core[0].start - r0, b.core[0].stop - r0),
            slice(b.core[1].start - c0, b.core[1].stop - c0),
        )
        for k, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl is None:
                continue
            m = lab[sl] == k
            touches = (
                (r0 > 0 and sl[0].start == 0)
                or (c0 > 0 and sl[1].start == 0)
                or (r0 + wh < H and sl[0].stop == wh)
                or (c0 + ww < W and sl[1].stop == ww)
            )
            inside = (
                sl[0].start >= cr[0].start
                and sl[0].stop <= cr[0].stop
                and sl[1].start >= cr[1].start
                and sl[1].stop <= cr[1].stop
            )
            if inside and not touches:
                out[_anchor(m, r0 + sl[0].start, c0 + sl[1].start, W)] = vectorise(
                    m, _crop(wf, sl), (r0 + sl[0].start, c0 + sl[1].start)
                )
                continue
            bbox = (
                slice(r0 + sl[0].start, r0 + sl[0].stop),
                slice(c0 + sl[1].start, c0 + sl[1].stop),
            )
            reg = _inter(bbox, b.core)
            if reg is None or not _sub(m, bbox, reg).any():
                continue  # another block's core holds all of it
            pieces.append(_Piece(bi, bbox, m, b.core, b.window, not touches))
    # group pieces from different blocks that agree on their shared area
    parent = list(range(len(pieces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    order = sorted(range(len(pieces)), key=lambda i: pieces[i].bbox[0].start)
    for x, i in enumerate(order):
        a = pieces[i]
        for j in order[x + 1 :]:
            b_ = pieces[j]
            if b_.bbox[0].start >= a.bbox[0].stop:
                break
            if a.block == b_.block:
                continue
            reg = _inter(a.bbox, b_.bbox)
            reg = reg and _inter(reg, a.window)
            reg = reg and _inter(reg, b_.window)
            if reg is None:
                continue
            ma, mb = _sub(a.mask, a.bbox, reg), _sub(b_.mask, b_.bbox, reg)
            inter = int((ma & mb).sum())
            if inter and inter / min(int(ma.sum()), int(mb.sum())) >= min_overlap:
                parent[find(i)] = find(j)
    groups: dict = {}
    for i, p in enumerate(pieces):
        groups.setdefault(find(i), []).append(p)
    for g in groups.values():
        full = [p for p in g if p.complete]
        if full:
            p = min(full, key=lambda p: p.block)
            bbox, m = p.bbox, p.mask
        else:
            r0, r1 = min(p.bbox[0].start for p in g), max(p.bbox[0].stop for p in g)
            c0, c1 = min(p.bbox[1].start for p in g), max(p.bbox[1].stop for p in g)
            canvas = np.zeros((r1 - r0, c1 - c0), bool)
            for p in g:
                reg = _inter(p.bbox, p.core)
                if reg is not None:
                    canvas[
                        reg[0].start - r0 : reg[0].stop - r0, reg[1].start - c0 : reg[1].stop - c0
                    ] |= _sub(p.mask, p.bbox, reg)
            rows, cols = np.nonzero(canvas.any(1))[0], np.nonzero(canvas.any(0))[0]
            bbox = (slice(r0 + rows[0], r0 + rows[-1] + 1), slice(c0 + cols[0], c0 + cols[-1] + 1))
            m = canvas[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        out[_anchor(m, bbox[0].start, bbox[1].start, W)] = vectorise(
            m, _crop(fields, bbox), (bbox[0].start, bbox[1].start)
        )
    out["_stats"] = {"blocks": len(blocks), "pieces": len(pieces), "merged_objects": len(groups)}
    return out
