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
