"""Block decoders (manyseg.vectors.blocks): a block decode equals the whole-scene decode,
whatever the block grid (contract: ManySeg-e028/notes/large_scene_vectors_contract.md §3)."""

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

shapely = pytest.importorskip("shapely")
pytest.importorskip("skimage")
from manyseg.instances import (  # noqa: E402
    anchor_labels,
    connected_components,
    instances_from_outputs,
)
from manyseg.vectors import planar_faces, polygons_from_corners, polygons_from_labels  # noqa: E402
from manyseg.vectors.blocks import (  # noqa: E402
    InstanceBlockDecoder,
    Scene,
    SceneRecords,
    _BlockResult,
    block_decoder,
    block_grid,
    decode_scene,
    records_to_geojson,
    write_records,
)
from manyseg.vectors.roofs import roof_faces, roof_mesh, roof_planes  # noqa: E402
from scipy import ndimage  # noqa: E402

GRIDS = ((128, 32, (0, 0)), (150, 48, (61, 97)), (200, 40, (13, 0)))


def _same(a, b) -> bool:
    return shapely.equals_exact(a, b, 0)


# ---------------------------------------------------------------- grid


def test_block_grid_partitions_scene_and_flags_edges():
    bs = block_grid(700, 900, 256, 32, offset=(100, 37))
    cov = np.zeros((700, 900), int)
    for i, b in enumerate(bs):
        assert b.index == i
        t, bt, lf, rt = b.core
        cov[t:bt, lf:rt] += 1
        w = b.window
        assert w == (max(t - 32, 0), min(bt + 32, 700), max(lf - 32, 0), min(rt + 32, 900))
        assert b.scene_edges == (w[0] == 0, w[1] == 700, w[2] == 0, w[3] == 900)
    assert (cov == 1).all()
    assert [b.core[0] for b in bs] == sorted(b.core[0] for b in bs)  # row-major
    assert len(block_grid(10, 10, 4096, 256)) == 1


# ---------------------------------------------------------------- label-level merge


class MaskDecoder(InstanceBlockDecoder):
    """Outline decoder over a plain ``mask`` field, labels from a pluggable function."""

    def __init__(self, fn=None, **kw):
        super().__init__(method="outline", simplify=0.0, min_area=0, **kw)
        self.fn = fn or (lambda f: ndimage.label(f["mask"][0] > 0.5)[0])

    def labels(self, fields, origin, scene_width):
        return self.fn(fields)


def _scene(seed=0, H=700, W=900):
    rng = np.random.default_rng(seed)
    m = np.zeros((H, W), bool)
    for _ in range(60):  # small and medium buildings
        r, c = rng.integers(0, H - 20), rng.integers(0, W - 20)
        m[r : r + rng.integers(4, 40), c : c + rng.integers(4, 40)] = True
    m[100:140, 50:800] = True  # long buildings larger than any halo
    m[300:650, 600:640] = True
    return m


def _masks(recs, shape):
    """Records -> per-object scene masks (outline polygons of axis-aligned pixel sets)."""
    from manyseg.vectors.roofs import _face_region

    return [_face_region(r["geometry"], shape, (0, 0), 0) for r in recs]


def test_blocks_equal_global_decode_for_any_grid():
    m = _scene().astype(np.float32)
    lab = ndimage.label(m > 0.5)[0]
    glob = polygons_from_labels(lab, simplify=0.0, min_area=0)
    for block, halo, off in ((256, 32, (0, 0)), (300, 64, (77, 151)), (200, 16, (13, 0))):
        out = decode_scene({"mask": m}, MaskDecoder(min_size=0), block, halo, offset=off)
        assert len(out) == len(glob), (block, halo, off, out.stats)
        assert all(any(_same(g, r["geometry"]) for g in glob) for r in out)
        assert out.stats["pieces"] > 0  # the long buildings exercise the merge


def test_ids_are_scene_anchors():
    m = _scene(1).astype(np.float32)
    lab = ndimage.label(m > 0.5)[0]
    anchors = sorted(np.unique(anchor_labels(torch.from_numpy(lab)).numpy())[1:] - 1)
    out = decode_scene({"mask": m}, MaskDecoder(min_size=0), 160, 24, offset=(40, 70))
    assert [r["id"] for r in out] == [int(a) for a in anchors]


def test_min_size_applies_to_merged_object():
    m = np.zeros((60, 1000), np.float32)
    m[30, 10:990] = 1  # 980 px, one pixel wide
    m[5:8, 5:8] = 1  # 9 px, removed
    out = decode_scene({"mask": m}, MaskDecoder(min_size=150), 100, 10)
    assert len(out) == 1 and out[0]["pixels"] == 980


def test_window_dependent_decoder_never_drops_or_duplicates():
    """One corridor, seeds far apart; the decoder assigns each pixel to the nearest *visible*
    seed, so windows disagree. Output must still be disjoint and cover all foreground."""
    H, W = 40, 900
    m = np.zeros((H, W), np.float32)
    m[10:30, 20:880] = 1
    seeds = np.zeros((H, W), np.float32)
    for c in (60, 300, 520, 840):
        seeds[20, c] = 1

    def dec(f):
        fg, sd = f["mask"][0] > 0.5, f["seed"][0]
        lab = np.zeros(fg.shape, np.int32)
        if not sd.any():
            return ndimage.label(fg)[0]
        _, (ri, ci) = ndimage.distance_transform_edt(sd == 0, return_indices=True)
        sid = ndimage.label(sd > 0)[0]
        lab[fg] = sid[ri[fg], ci[fg]]
        return lab

    for block, halo, off in ((150, 20, (0, 0)), (128, 40, (0, 77)), (200, 5, (0, 13))):
        out = decode_scene({"mask": m, "seed": seeds}, MaskDecoder(dec, min_size=0), block, halo,
                           offset=off)
        cover = np.sum(_masks(out, m.shape), axis=0)
        assert cover.max() == 1 and ((cover == 1) == (m > 0)).all(), (block, halo, off)


def test_whole_view_separates_instances_a_window_joins():
    """Regression: a window that misses a large instance's seed joins it with its touching
    neighbour. Matching on the whole area both windows see links that piece to one object
    only, and the neighbouring block's whole views keep the two apart (as the global decode)."""
    H, W = 30, 400
    m = np.zeros((H, W), np.float32)
    m[10:20, 60:330] = 1  # instance A = cols 60-299 (seed col 70), B = cols 300-329

    class SeedDecoder(MaskDecoder):
        def labels(self, fields, origin, scene_width):
            fg = fields["mask"][0] > 0.5
            cols = np.arange(fg.shape[1])[None] + origin[1]
            if origin[1] <= 70 < origin[1] + fg.shape[1]:  # sees A's seed: the true split
                return np.where(fg, np.where(cols < 300, 1, 2), 0)
            return fg.astype(np.int32)  # B's seed floods everything it sees

    glob = np.where(m > 0, np.where(np.arange(W)[None] < 300, 1, 2), 0)
    ref = polygons_from_labels(glob, simplify=0.0, min_area=0)
    for block, halo in ((150, 120), (100, 210)):  # the block right of A misses A's seed
        out = decode_scene({"mask": m}, SeedDecoder(min_size=0), block, halo)
        assert _canon(r["geometry"] for r in out) == _canon(ref), (block, halo, out.stats)


def test_thin_diagonal_object_merges_sparsely():
    """Codex review #2: a long staircase line spanning many blocks is one object; the merge
    works on its pixels, not on dense canvases over its bounding box."""
    n = 900
    m = np.zeros((n, n), np.float32)
    r = np.arange(n)
    m[r, r] = 1
    m[r[:-1], r[1:]] = 1  # 4-connected staircase
    out = decode_scene({"mask": m}, MaskDecoder(min_size=0), 100, 12, offset=(37, 0))
    assert len(out) == 1 and out[0]["pixels"] == int(m.sum()) and out[0]["id"] == 0


def test_u_shape_that_leaves_a_window_is_one_object():
    m = np.zeros((300, 300), np.float32)
    m[20:30, 20:280] = 1
    m[20:280, 20:30] = 1
    m[20:280, 270:280] = 1  # a U open at the bottom: windows near the bottom see two arms
    for block, halo, off in ((64, 8, (0, 0)), (100, 16, (37, 11))):
        out = decode_scene({"mask": m}, MaskDecoder(min_size=0), block, halo, offset=off)
        assert len(out) == 1, (block, halo, off)


def test_empty_scene_and_empty_blocks():
    m = np.zeros((300, 400), np.float32)
    assert list(decode_scene({"mask": m}, MaskDecoder(), 100, 16)) == []
    m[150:160, 10:20] = 1  # one object, all other blocks empty
    out = decode_scene({"mask": m}, MaskDecoder(), 100, 16)
    assert [r["id"] for r in out] == [150 * 400 + 10] and out[0]["pixels"] == 100


# ---------------------------------------------------------------- determinism of the window decode


def test_connected_components_origin_matches_scene_labels():
    rng = np.random.default_rng(0)
    m = torch.from_numpy(ndimage.gaussian_filter(rng.normal(size=(120, 150)), 2) > 0.1)[None]
    g = connected_components(m)
    t, lf, h, w = 30, 41, 60, 70
    for impl in ("scipy", "torch"):
        if impl == "torch":
            from manyseg.instances.postprocess import _connected_components_torch, _scene_index

            loc = _scene_index(_connected_components_torch(m[:, t:t + h, lf:lf + w]), (t, lf), 150)
        else:
            loc = connected_components(m[:, t:t + h, lf:lf + w], origin=(t, lf), scene_width=150)
        inner = ndimage.label(m[0, t:t + h, lf:lf + w].numpy())[0]
        for k, sl in enumerate(ndimage.find_objects(inner), 1):  # inside: same scene id
            if sl[0].start > 0 and sl[1].start > 0 and sl[0].stop < h and sl[1].stop < w:
                assert torch.equal(loc[0][inner == k], g[0, t:t + h, lf:lf + w][inner == k])
    with pytest.raises(ValueError):
        connected_components(m, origin=(0, 5), scene_width=100)


# _terraces' distance is normalised: the fields decoder gets E046's thresholds, not E052's px
NORM = {"extent_threshold": 0.6, "seed_threshold": 0.3}


def _terraces(seed=3, H=400, W=500, long_row=True):
    """Noisy instance fields of rows of touching houses (+ a terrace longer than any halo),
    corner blobs at house corners and random wall directions; height = a plane per house."""
    rng = np.random.default_rng(seed)
    inst = np.zeros((H, W), np.int32)
    k = 0
    for _ in range(14):
        r, c, h = rng.integers(0, H - 40), rng.integers(0, W - 150), rng.integers(12, 35)
        for _ in range(rng.integers(2, 7)):
            w = rng.integers(8, 30)
            k += 1
            inst[r : r + h, c : c + w] = k
            c += w
    if long_row:
        for c in range(20, 480, 46):  # 10 touching houses, 460 px in all
            k += 1
            inst[200:226, c : c + 46] = k
    edge = np.zeros_like(inst, bool)
    for ax in (0, 1):
        for s in (1, -1):
            edge |= (inst != np.roll(inst, s, ax)) & (inst > 0)
    d = ndimage.distance_transform_edt(~edge) + 1
    d[inst == 0] = 0
    mx = ndimage.maximum(d, inst, index=np.arange(k + 1))

    def noise(s, sig=2):
        return ndimage.gaussian_filter(rng.normal(0, 1, (H, W)), sig) * s

    blobs = np.zeros((H, W))
    for sl in ndimage.find_objects(inst):
        if sl is not None:
            for y in (sl[0].start, sl[0].stop - 1):
                for x in (sl[1].start, sl[1].stop - 1):
                    blobs[y, x] = 1
    blobs = ndimage.gaussian_filter(blobs, 1.5)
    planes = rng.normal(0, [0.2, 0.2, 3], (k + 1, 3)) + [0, 0, 8]
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    height = np.where(inst > 0, planes[inst, 0] * (xx % 50) + planes[inst, 1] * (yy % 50)
                      + planes[inst, 2], 0) + noise(0.1, 1)
    def prob(x):
        return np.clip(x, 0, 1).astype(np.float32)[None]

    return {
        "extent": prob((inst > 0) + noise(0.6)),
        "boundary": prob(((d <= 2) & (inst > 0)) + noise(0.6)),
        "distance": prob(d / np.maximum(mx[inst], 1) + noise(0.15)),
        "corner": prob(blobs / blobs.max() + noise(0.15, 1)),
        "direction": np.stack([noise(3), noise(3)]).astype(np.float32),
        "height": height.astype(np.float32),
    }


THR = dict(extent_threshold=0.6, boundary_threshold=0.5, seed_threshold=0.15)


def _labels(f, backend="auto", **thr):
    t = [torch.from_numpy(f[n]) for n in ("extent", "boundary", "distance")]
    return instances_from_outputs(*[x[None] for x in t], logits=False, min_size=16,
                                  backend=backend, **(thr or THR))[0]


@pytest.mark.parametrize("backend", ["skimage", "torch"])
def test_instance_window_decode_matches_scene(backend):
    """Interior instances of a window get the same pixels as in the whole-scene decode (scene
    index tie-breaks; skimage's equal-distance ties follow raster order)."""
    f = _terraces()
    # quantised fields make many equal distances (ties) as float16 stitching does
    f = {k: v.astype(np.float16).astype(np.float32) for k, v in f.items()}
    g = anchor_labels(_labels(f, backend)).numpy()
    for t, lf, h, w in ((37, 51, 200, 260), (150, 0, 250, 300)):
        sub = {k: v[..., t:t + h, lf:lf + w] for k, v in f.items()}
        tt = [torch.from_numpy(sub[n])[None] for n in ("extent", "boundary", "distance")]
        loc = instances_from_outputs(*tt, logits=False, min_size=16, backend=backend,
                                     origin=(t, lf), scene_width=500, ids="anchor", **THR)[0]
        loc = loc.numpy()
        n = 0
        for i in np.unique(loc)[1:]:
            rr, cc = np.nonzero(loc == i)
            if (rr.min() > 0 and rr.max() < h - 1 and cc.max() < w - 1
                    and (cc.min() > 0 or lf == 0)):  # not cut by the window
                assert (g[t:t + h, lf:lf + w][loc == i] == i).all()
                assert (g == i).sum() == (loc == i).sum()
                n += 1
        assert n > 10


# ---------------------------------------------------------------- footprints / fields / roofs


def _canon(polys):
    return sorted(shapely.normalize(p).wkb for p in polys)


@pytest.mark.parametrize("method", ["corners", "outline"])
def test_footprints_blocks_equal_whole_scene(method):
    f = _terraces()
    lab = _labels(f)
    if method == "corners":
        ref = polygons_from_corners(lab, f["corner"][0], f["direction"], corner_threshold=0.5,
                                    big_dev=4.0, agree=0.4, merge_deg=8.0)
    else:
        ref = polygons_from_labels(lab, simplify=1.0, min_area=16)
    dec = block_decoder("footprints", method=method)
    for block, halo, off in GRIDS:
        out = decode_scene(f, dec, block, halo, offset=off)
        assert _canon(r["geometry"] for r in out) == _canon(ref), (block, halo, off, out.stats)
        assert out.stats["merged_objects"] > 0
        assert [r["id"] for r in out] == sorted(r["id"] for r in out)
    one = decode_scene(f, dec, 4096, 256)  # one block = the whole scene
    assert _canon(r["geometry"] for r in one) == _canon(ref)


def test_footprints_logits_apply_to_extent_boundary_and_corner():
    import torch

    f = _terraces()
    lg = {k: (torch.logit(torch.from_numpy(np.clip(v, 1e-3, 1 - 1e-3).astype(np.float32))).numpy()
              if k in ("extent", "boundary", "corner") else v) for k, v in f.items()}
    back = {k: (torch.sigmoid(torch.from_numpy(v)).numpy() if k in ("extent", "boundary", "corner")
                else v) for k, v in lg.items()}
    ref = decode_scene(back, block_decoder("footprints"), 200, 40)
    got = decode_scene(lg, block_decoder("footprints", logits=True), 200, 40)
    assert len(ref) > 10 and _canon(r["geometry"] for r in got) == _canon(
        r["geometry"] for r in ref)


def test_fields_blocks_equal_whole_scene():
    f = _terraces(5)
    thr = dict(extent_threshold=0.6, boundary_threshold=0.3, seed_threshold=0.3)
    ref = polygons_from_labels(_labels(f, **thr), simplify=0.5, min_area=16)
    for block, halo, off in GRIDS:
        out = decode_scene(f, block_decoder("fields", **NORM), block, halo, offset=off)
        assert _canon(r["geometry"] for r in out) == _canon(ref), (block, halo, off)


def _roof_ref(f, mesh="J2"):
    lab = _labels(f, extent_threshold=0.4, boundary_threshold=0.3, seed_threshold=0.15)
    return roof_faces(lab, f["corner"][0], f["direction"], f["height"], corner_threshold=0.5,
                      big_dev=4.0, agree=0.6, merge_deg=8.0, mesh=mesh)


def _assert_same_roofs(out, ref, ctx):
    assert [(r["id"], r["part"]) for r in out] == [(r["id"], r["part"]) for r in ref], ctx
    for a, b in zip(out, ref):
        assert _same(a["geometry"], b["geometry"]), (ctx, a["id"])
        assert a["plane"] == b["plane"], (ctx, a["id"])
        if "mesh" in b:
            assert np.array_equal(a["mesh"], b["mesh"]), (ctx, a["id"])


def test_roofs_blocks_equal_whole_scene():
    """Planar join, planes and J2 mesh across seams: faces of the long terrace (a cluster
    larger than any halo) are joined in combine, the rest in their blocks."""
    f = _terraces(7)
    ref = _roof_ref(f)
    assert sum(r["mesh"] is not None for r in ref) == len(ref)
    dec = block_decoder("roofs")
    for block, halo, off in GRIDS:
        out = decode_scene(f, dec, block, halo, offset=off)
        _assert_same_roofs(out, ref, (block, halo, off, out.stats))
        assert out.stats["resolved_in_blocks"] > 0 and out.stats["joined_in_combine"] > 0
        assert out.stats["nan_planes"] == 0
    out = decode_scene(f, block_decoder("roofs", mesh=None), 150, 40)
    _assert_same_roofs(out, _roof_ref(f, None), "no mesh")
    assert "mesh" not in out[0]


def test_roof_faces_match_polygons_from_corners_and_planar_faces():
    f = _terraces(7)
    lab = _labels(f, extent_threshold=0.4, boundary_threshold=0.3, seed_threshold=0.15)
    kw = dict(corner_threshold=0.5, big_dev=4.0, agree=0.6, merge_deg=8.0)
    faces = polygons_from_corners(lab, f["corner"][0], f["direction"], **kw)
    recs = roof_faces(lab, f["corner"][0], f["direction"], f["height"], mesh=None, **kw)
    a = shapely.union_all(planar_faces(faces))
    b = shapely.union_all([r["geometry"] for r in recs])
    assert len(recs) == len(planar_faces(faces))
    assert a.symmetric_difference(b).area < 1e-6 * a.area


def test_planar_faces_ignore_faces_out_of_reach():
    """A face's join depends only on faces within reach (cluster by cluster)."""
    a = shapely.box(0, 0, 10, 10)
    b = shapely.Polygon([(10.8, 0), (20, 0), (20, 10), (10.6, 10)])
    far = [shapely.box(100 + 13 * i, 0, 110 + 13 * i, 10) for i in range(5)]
    alone = planar_faces([a, b])
    crowd = planar_faces([a, *far, b])
    assert _same(alone[0], crowd[0]) and _same(alone[1], crowd[-1])


def test_roof_planes_and_mesh():
    yy, xx = np.mgrid[0:60, 0:80] + 0.5
    h = np.where(xx < 40, 5 + 0.1 * xx, 13 - 0.1 * xx).astype(np.float32)
    faces = [shapely.box(0, 0, 40, 60), shapely.box(40, 0, 80, 60)]
    P = roof_planes(faces, h)
    np.testing.assert_allclose(P, [[0.1, 0, 5], [-0.1, 0, 13]], atol=1e-4)
    # a crop at an integer origin gives the same planes
    np.testing.assert_array_equal(roof_planes(faces, h[0:60, 0:80], (0, 0)), P)
    rings = roof_mesh(faces, P)
    shared = [v for r in rings for v in r if v[0] == 40]
    assert all(abs(v[2] - 9) < 1e-3 for v in shared)  # the ridge sits on both planes
    with pytest.raises(ValueError):
        roof_mesh(faces, P, join="J3")


# ---------------------------------------------------------------- API, pickling, output


def test_results_are_picklable_and_small():
    f = _terraces(7)
    for task in ("footprints", "roofs"):
        dec = block_decoder(task)
        b = block_grid(400, 500, 150, 40)[4]
        t, bt, lf, rt = b.window
        res = dec({k: v[..., t:bt, lf:rt] if v.ndim == 3 else v[None, t:bt, lf:rt]
                   for k, v in f.items()}, b, Scene(400, 500))
        assert isinstance(res, _BlockResult)
        back = pickle.loads(pickle.dumps(res))
        assert len(back.records) == len(res.records)
        rec = pickle.dumps(res.records)
        assert len(rec) < 2000 * max(1, len(res.records))  # small objects carry no rasters
        pickle.loads(pickle.dumps(dec))
    out = decode_scene(f, block_decoder("footprints"), 150, 40)
    back = pickle.loads(pickle.dumps(out))
    assert isinstance(back, SceneRecords) and back.stats == out.stats


def test_registry_and_halo_check():
    assert block_decoder("footprints").method == "corners"
    assert block_decoder("footprints", method="outline").simplify == 1.0
    assert block_decoder("fields").method == "outline"
    assert block_decoder("roofs", mesh=None).mesh is None
    with pytest.raises(ValueError):
        block_decoder("rivers")
    with pytest.raises(ValueError):
        block_decoder("footprints", method="graph")
    f = _terraces()
    with pytest.raises(ValueError):
        decode_scene(f, block_decoder("roofs"), 128, 8)
    b = block_grid(400, 500, 128, 8)[5]  # as ManyPred would call it, without decode_scene
    t, bt, lf, rt = b.window
    with pytest.raises(ValueError, match="halo"):
        block_decoder("footprints")({k: v[..., t:bt, lf:rt] for k, v in f.items()}, b,
                                    Scene(400, 500))


def test_decode_scene_checks_offset_grids_and_scene_size():
    """Codex review #3 / #4: an offset grid splits a scene smaller than one block; a scene that
    does not match the fields would give wrong anchors."""
    f = _terraces()
    with pytest.raises(ValueError, match="halo"):
        decode_scene(f, block_decoder("footprints"), 4096, 4, offset=(200, 0))
    with pytest.raises(ValueError, match="scene"):
        decode_scene(f, block_decoder("footprints"), 200, 40, scene=Scene(400, 1000))


def test_decode_scene_empty_fields_and_default_one_block():
    with pytest.raises(ValueError, match="must not be empty"):
        decode_scene({}, block_decoder("footprints"))
    f = _terraces()
    one = decode_scene(f, block_decoder("fields", **NORM))  # default 4096 px > scene: one block
    assert one.stats["blocks"] == 1
    ref = decode_scene(f, block_decoder("fields", **NORM), 200, 40)
    assert _canon(r["geometry"] for r in one) == _canon(r["geometry"] for r in ref)


def test_public_surface():
    import manyseg.vectors as V

    for name in ("graph_from_mask_blocks", "dihedral_shape", "SceneRecords"):
        assert name in V.__all__ and hasattr(V, name)
    for name in ("roof_faces", "roof_records", "face_plane", "roof_planes", "roof_mesh",
                 "Records", "BlockResult"):
        assert name not in V.__all__ and not hasattr(V, name), name
    from manyseg.vectors import roofs

    assert all(hasattr(roofs, n) for n in ("roof_faces", "roof_records", "face_plane",
                                           "roof_planes", "roof_mesh"))


def test_bridge_ends_mutates_and_returns_the_graph():
    import networkx as nx
    from manyseg.vectors import bridge_ends

    g = nx.Graph()
    for n, xy in enumerate([(10, 50), (40, 50), (50, 50), (80, 50)]):
        g.add_node(n, xy=np.array(xy, float))
    g.add_edges_from([(0, 1), (2, 3)])
    assert bridge_ends(g, 20, (100, 100)) is g and g.has_edge(1, 2)
    assert bridge_ends(g, 0, (100, 100)) is g


def test_roads_registry_entry():
    try:
        from manyseg.vectors.roads import RoadBlockDecoder  # noqa: F401
    except ImportError:
        pytest.skip("RoadBlockDecoder not merged yet")
    assert block_decoder("roads") is not None


def test_names_mapping_and_2d_fields():
    f = _terraces()
    ren = {"ext": f["extent"][0], "bnd": f["boundary"], "dst": f["distance"]}
    dec = block_decoder("fields", names={"extent": "ext", "boundary": "bnd", "distance": "dst"},
                        **NORM)
    out = decode_scene(ren, dec, 200, 40)
    ref = decode_scene(f, block_decoder("fields", **NORM), 200, 40)
    assert _canon(r["geometry"] for r in out) == _canon(r["geometry"] for r in ref)


def test_write_records(tmp_path):
    f = _terraces(7)
    out = decode_scene(f, block_decoder("roofs"), 200, 40)
    sc = Scene(400, 500, (0.25, 0, 1000, 0, -0.25, 2000), "EPSG:28992")
    fc = records_to_geojson(out, sc.transform, sc.crs)
    assert len(fc["features"]) == len(out) and fc["crs"]["properties"]["name"] == "EPSG:28992"
    p = fc["features"][0]
    assert {"id", "part", "plane_a", "plane_b", "plane_c"} <= set(p["properties"])
    x, y = p["geometry"]["coordinates"][0][0]
    gx, gy = out[0]["geometry"].exterior.coords[0]
    assert (x, y) == pytest.approx((1000 + 0.25 * gx, 2000 - 0.25 * gy))
    path = write_records(out, tmp_path / "roofs.geojson", sc, mesh=True)
    js = json.loads(path.read_text())
    assert len(js["features"][0]["geometry"]["coordinates"][0][0]) == 3  # x, y, z
    pytest.importorskip("geopandas")
    import geopandas as gpd

    g = gpd.read_file(write_records(out, tmp_path / "roofs.gpkg", sc))
    assert len(g) == len(out) and g.crs.to_epsg() == 28992


def test_scene_records_carry_transform_and_crs(tmp_path):
    f = _terraces(7)
    sc = Scene(400, 500, (0.25, 0, 1000, 0, -0.25, 2000), "EPSG:28992")
    out = decode_scene(f, block_decoder("footprints"), 200, 40, scene=sc)
    assert isinstance(out, SceneRecords) and isinstance(out, list)
    assert out.transform == sc.transform and out.crs == sc.crs
    ref = records_to_geojson(out, sc.transform, sc.crs)
    assert out.to_geojson() == ref == records_to_geojson(out)
    px = records_to_geojson(out, None)  # explicit None: pixel coordinates, so no CRS named
    assert px["features"][0]["geometry"] != ref["features"][0]["geometry"] and "crs" not in px
    assert records_to_geojson(out, None, "EPSG:28992")["crs"] == ref["crs"]  # unless asked
    a = json.loads(out.write(tmp_path / "a.geojson").read_text())
    b = json.loads(write_records(out, tmp_path / "b.geojson").read_text())
    c = json.loads(write_records(out, tmp_path / "c.geojson", sc).read_text())
    assert a == b == c == ref
    back = pickle.loads(pickle.dumps(out))
    assert back.transform == sc.transform and back.crs == sc.crs and list(back) == list(out)
    plain = decode_scene(f, block_decoder("footprints"), 200, 40)  # no georeferencing
    assert plain.transform is None and "crs" not in plain.to_geojson()


# ---------------------------------------------------------------- roads: graph_from_mask_blocks


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


def test_graph_blocks_torch_and_empty():
    """Codex 10-28 #2/#3: torch input (with grad) and zero-sized scenes."""
    import torch
    from manyseg.vectors.graph import graph_from_mask, graph_from_mask_blocks

    p = _road_prob(0, 200, 240)
    t = torch.from_numpy(p).requires_grad_(True)
    a = graph_from_mask_blocks(t, 0.5, block=96, halo=48, spur=10, gap=25, min_area=30)
    b = graph_from_mask(p, 0.5, spur=10, gap=25, min_area=30)
    assert _edges(a) == _edges(b)
    for shape in ((0, 10), (10, 0)):
        assert len(graph_from_mask_blocks(np.zeros(shape, np.float32), 0.5).edges) == 0


# ---------------------------------------------------------------- real stitched fields (slow)

E051 = Path(os.environ.get("MANYSEG_E051_FIELDS",
                           "$MANYSEG_BENCH_DATA/cache/e051_fields"))


@pytest.mark.slow
@pytest.mark.skipif(not (E051 / "meta.json").exists(), reason="e051 fixture fields not found")
@pytest.mark.parametrize("scene", ["leidscherijn_0", "staphorst_0"])
@pytest.mark.parametrize("task", ["footprints", "fields", "roofs"])
def test_real_fields_blocks_equal_whole_scene(task, scene):
    """2048 x 2048 seamless model outputs (E051): block decodes are identical to the
    whole-scene decode of the same fields."""
    meta = json.loads((E051 / "meta.json").read_text())
    key = f"roofs_{scene}"
    f = {k: np.load(E051 / f"{key}_{k}.npy", mmap_mode="r") for k in meta[key]["heads"]}
    f["height"] = np.load(E051 / f"{key}_ndsm.npy", mmap_mode="r")
    a = {k: np.asarray(v, np.float32) for k, v in f.items()}
    if task == "roofs":
        F3 = meta["_decoder_params"]["roofs_F3"]
        lab = _labels(a, **F3["thr"])
        ref = roof_faces(lab, a["corner"][0], a["direction"], a["height"], **F3["params"])
    elif task == "footprints":
        ref = polygons_from_corners(_labels(a), a["corner"][0], a["direction"],
                                    corner_threshold=0.5, big_dev=4.0, agree=0.4, merge_deg=8.0)
    else:
        thr = dict(extent_threshold=0.6, boundary_threshold=0.3, seed_threshold=0.3)
        ref = polygons_from_labels(_labels(a, **thr), simplify=0.5, min_area=16)
    for block, halo, off in ((768, 128, (0, 0)), (1024, 256, (333, 333))):
        # E051's fields are a roof model's (normalised distance): E046's fields thresholds
        out = decode_scene(f, block_decoder(task, **(NORM if task == "fields" else {})),
                           block, halo, offset=off)
        if task == "roofs":
            _assert_same_roofs(out, ref, (block, halo))
        else:
            assert _canon(r["geometry"] for r in out) == _canon(ref), (block, halo, out.stats)


def test_graph_blocks_small_blob_across_a_seam():
    """As the road block decoder (Codex 10-29 #1): a 96-px, 1-px-wide blob across the x = 128
    seam is cut by both windows at halo 40 (windows end at x = 168 and start at x = 88); with
    min_area 100 neither window may keep it as 'maybe large', so the halo is raised to
    min_area - 1."""
    from manyseg.vectors.graph import graph_from_mask, graph_from_mask_blocks

    p = np.zeros((400, 400), np.float32)
    p[100:300, 60] = 0.9  # a road, kept
    p[200, 80:176] = 0.9  # 96 px across the x = 128 seam, past both windows' edges: removed
    g = graph_from_mask(p, 0.5, gap=0, min_area=100)
    assert len(g.edges) == 1
    for halo in (40, 99, 120):
        b = graph_from_mask_blocks(p, 0.5, block=128, halo=halo, gap=0, min_area=100)
        assert _edges(b) == _edges(g), halo
