"""Block-wise decoding must equal a global decode, whatever the block grid (design (c))."""

import numpy as np
import shapely
from manyseg.vectors import polygons_from_labels
from manyseg.vectors.blocks import block_grid, decode_blocks
from scipy import ndimage


def _scene(seed=0, H=700, W=900):
    rng = np.random.default_rng(seed)
    m = np.zeros((H, W), bool)
    for _ in range(60):  # small and medium buildings
        r, c = rng.integers(0, H - 20), rng.integers(0, W - 20)
        m[r : r + rng.integers(4, 40), c : c + rng.integers(4, 40)] = True
    m[100:140, 50:800] = True  # long buildings larger than any halo
    m[300:650, 600:640] = True
    return m


def _labels(wf):
    return ndimage.label(wf["mask"] > 0.5)[0]


def _vec(mask, wf, origin):  # mask is bbox-cropped
    (p,) = polygons_from_labels(mask.astype(np.int32), simplify=0.0, min_area=1) or [None]
    return shapely.affinity.translate(p, origin[1], origin[0]) if p is not None else None


def _canon(polys):
    return sorted(
        (round(p.area, 3), round(p.bounds[0], 3), round(p.bounds[1], 3))
        for p in polys
        if p is not None
    )


def test_block_grid_tiles_scene():
    bs = block_grid((700, 900), 256, 32, offset=(100, 37))
    cov = np.zeros((700, 900), int)
    for b in bs:
        cov[b.core] += 1
    assert (cov == 1).all()


def test_blocks_equal_global_decode_for_any_grid():
    m = _scene().astype(np.float32)
    lab = ndimage.label(m > 0.5)[0]
    glob = _canon(_vec(lab == k, {}, (0, 0)) for k in range(1, lab.max() + 1))
    for block, halo, off in ((256, 32, (0, 0)), (300, 64, (77, 151)), (200, 16, (13, 0))):
        out = decode_blocks({"mask": m}, _labels, _vec, block=block, halo=halo, offset=off)
        st = out.pop("_stats")
        assert _canon(out.values()) == glob, (block, halo, off, st)
        assert st["pieces"] > 0  # the long buildings exercise the merge


def test_blocks_equal_global_with_instance_decoder():
    """Real B0 decoder (seeds + equal-speed growth) on noisy fields of touching terraces."""
    import torch
    from manyseg.instances.postprocess import instances_from_outputs

    rng = np.random.default_rng(3)
    H, W = 400, 500
    inst = np.zeros((H, W), np.int32)
    k = 0
    for _ in range(12):
        r, c, h = rng.integers(0, H - 40), rng.integers(0, W - 150), rng.integers(12, 35)
        for _ in range(rng.integers(2, 7)):
            w = rng.integers(8, 30)
            k += 1
            inst[r : r + h, c : c + w] = k
            c += w
    k += 1
    inst[200:225, 20:480] = k
    edge = np.zeros_like(inst, bool)
    for ax in (0, 1):
        for s in (1, -1):
            edge |= (inst != np.roll(inst, s, ax)) & (inst > 0)
    d = ndimage.distance_transform_edt(~edge) + 1
    d[inst == 0] = 0
    mx = ndimage.maximum(d, inst, index=np.arange(k + 1))

    def noise(s):
        return ndimage.gaussian_filter(rng.normal(0, 1, (H, W)), 2) * s

    f = {
        "extent": np.clip((inst > 0) + noise(0.6), 0, 1).astype(np.float32),
        "boundary": np.clip(((d <= 2) & (inst > 0)) + noise(0.6), 0, 1).astype(np.float32),
        "distance": np.clip(d / np.maximum(mx[inst], 1) + noise(0.15), 0, 1).astype(np.float32),
    }

    def labels(wf):
        t = [
            torch.from_numpy(np.ascontiguousarray(wf[n]))[None, None]
            for n in ("extent", "boundary", "distance")
        ]
        return instances_from_outputs(*t, logits=False, min_size=10, backend="torch")[0].numpy()

    def key(mask, wf, origin):
        rr, cc = np.nonzero(mask)
        return (int(mask.sum()), int(rr.min() + origin[0]), int(cc.min() + origin[1]))

    lab = labels(f)
    glob = sorted(key(lab == i, None, (0, 0)) for i in range(1, lab.max() + 1))
    for block, halo, off in ((128, 32, (50, 90)), (200, 64, (0, 0))):
        out = decode_blocks(f, labels, key, block=block, halo=halo, offset=off)
        st = out.pop("_stats")
        assert sorted(out.values()) == glob, (block, halo, off, st)


def _scene_masks(out, shape):
    """vectorise = (origin, mask) -> per-object scene masks."""
    res = []
    for o, m in out.values():
        full = np.zeros(shape, bool)
        full[o[0] : o[0] + m.shape[0], o[1] : o[1] + m.shape[1]] = m
        res.append(full)
    return res


def _raw(mask, wf, origin):
    return origin, mask.copy()


def test_min_size_applies_to_merged_object():
    """Codex blocks #1: a long thin object must not be dropped by per-window size filtering."""
    m = np.zeros((60, 1000), np.float32)
    m[30, 10:990] = 1  # 980 px, one pixel wide
    m[5:8, 5:8] = 1  # 9 px, removed
    out = decode_blocks({"mask": m}, _labels, _raw, block=100, halo=10, min_size=150)
    out.pop("_stats")
    (obj,) = _scene_masks(out, m.shape)
    assert obj.sum() == 980


def test_window_dependent_decoder_never_drops_or_duplicates():
    """Codex blocks #2/#3: one corridor, seeds far apart; the decoder assigns each pixel to the
    nearest *visible* seed, so windows disagree. Output must still be disjoint and cover all."""
    H, W = 40, 900
    m = np.zeros((H, W), np.float32)
    m[10:30, 20:880] = 1
    seeds = np.zeros((H, W), np.float32)
    for c in (60, 300, 520, 840):
        seeds[20, c] = 1

    def dec(wf):
        fg = wf["mask"] > 0.5
        lab = np.zeros(fg.shape, np.int32)
        if not wf["seed"].any():
            return ndimage.label(fg)[0]
        _, (ri, ci) = ndimage.distance_transform_edt(wf["seed"] == 0, return_indices=True)
        sid = ndimage.label(wf["seed"] > 0)[0]
        lab[fg] = sid[ri[fg], ci[fg]]
        return lab

    for block, halo, off in ((150, 20, (0, 0)), (128, 40, (0, 77)), (200, 5, (0, 13))):
        out = decode_blocks(
            {"mask": m, "seed": seeds}, dec, _raw, block=block, halo=halo, offset=off
        )
        out.pop("_stats")
        objs = _scene_masks(out, m.shape)
        cover = np.sum(objs, axis=0)
        assert cover.max() == 1 and ((cover == 1) == (m > 0)).all(), (block, halo, off)


def test_u_shape_that_leaves_a_window_is_one_object():
    m = np.zeros((300, 300), np.float32)
    m[20:30, 20:280] = 1
    m[20:280, 20:30] = 1
    m[20:280, 270:280] = 1  # a U open at the bottom: windows near the bottom see two arms
    for block, halo, off in ((64, 8, (0, 0)), (100, 16, (37, 11))):
        out = decode_blocks({"mask": m}, _labels, _raw, block=block, halo=halo, offset=off)
        out.pop("_stats")
        assert len(out) == 1, (block, halo, off)


def _road_prob(seed=0, H=600, W=700):
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W), np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(25):  # thick segments: pixels within half-width of the segment
        p, q = rng.integers(0, [W, H]).astype(float), rng.integers(0, [W, H]).astype(float)
        d = q - p
        t = np.clip(((xx - p[0]) * d[0] + (yy - p[1]) * d[1]) / max(d @ d, 1e-9), 0, 1)
        img[np.hypot(xx - p[0] - t * d[0], yy - p[1] - t * d[1]) <= rng.integers(3, 12) / 2] = 255
    for _ in range(40):  # specks (min_area) and a wide plaza (thinning reach)
        r, c = rng.integers(0, H - 5), rng.integers(0, W - 5)
        img[r : r + 3, c : c + 3] = 255
    img[200:260, 300:400] = 255
    p = img.astype(np.float32) / 255
    return np.clip(p + ndimage.gaussian_filter(rng.normal(0, 0.3, p.shape), 2), 0, 1)


def _edges(vg):
    ends = (sorted([vg.xy[a], vg.xy[b]], key=tuple) for a, b in vg.edges)
    return sorted(tuple(np.round(np.concatenate(e), 3)) for e in ends)


def test_graph_blocks_equal_global():
    from manyseg.vectors.graph import graph_from_mask, graph_from_mask_blocks

    for seed in (0, 1):
        p = _road_prob(seed)
        g = graph_from_mask(p, 0.5, spur=10, gap=25, min_area=30)
        for block, halo in ((128, 48), (200, 64), (256, 40)):
            b = graph_from_mask_blocks(p, 0.5, block=block, halo=halo, spur=10, gap=25, min_area=30)
            assert _edges(b) == _edges(g), (seed, block, halo)
