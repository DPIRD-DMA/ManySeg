"""Vector-output training presets: CornerLineTask / RoadGraphTask losses and evaluators equal the
experiment implementations (E030 ``CornerLineTask``, E044 ``RoadTask``), the loader's dihedral
augmentation transforms direction targets as the geometry would encode them, and both tasks
train end to end through ``run_training``."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("shapely")
pytest.importorskip("networkx")
pytest.importorskip("rasterio")

import networkx as nx  # noqa: E402
import shapely  # noqa: E402
from manyseg.vectors import corner_targets, dihedral_geometry, dihedral_xy, road_targets  # noqa: E402
from manyseg.vectors.augment import dihedral  # noqa: E402
from manyseg_bench import datasets as bench_datasets  # noqa: E402
from manyseg_bench.datasets import DatasetInfo, GPULoader  # noqa: E402
from manyseg_bench.datasets.base import load_split, save_split  # noqa: E402
from manyseg_bench.tasks import CornerLineTask, RoadGraphTask, build_task  # noqa: E402

EXPERIMENTS = Path(__file__).resolve().parents[3] / "experiments"


@pytest.fixture(scope="module")
def exp():
    """E030 common.py and E044 train44.py, imported as the experiments run them."""
    dirs = [EXPERIMENTS / "E030_whu_polygons", EXPERIMENTS / "E044_cityscale_unified"]
    if not all(d.is_dir() for d in dirs):
        pytest.skip("not a repository checkout (no experiments/)")
    sys.path[:0] = [str(d) for d in dirs]
    try:
        import common
        import train44
    except ImportError as e:  # pragma: no cover - experiment dependencies missing
        pytest.skip(f"experiment modules not importable: {e}")
    finally:
        for d in dirs:
            sys.path.remove(str(d))
    return SimpleNamespace(CornerLineTask=common.CornerLineTask, RoadTask=train44.RoadTask)


def _gauss(points, size, sigma=2.0):
    yy, xx = np.mgrid[:size, :size] + 0.5
    out = np.zeros((size, size), np.float32)
    for x, y in points:
        out = np.maximum(out, np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2)))
    return out


def _corner_batch(seed, B=2, size=48, corners=True, directions=True):
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    inst = torch.zeros(B, size, size, dtype=torch.long)
    for b in range(B):
        for k in range(4):
            y, x = rng.integers(0, size - 12, 2)
            inst[b, y:y + int(rng.integers(5, 12)), x:x + int(rng.integers(5, 12))] = k + 1
    corner = torch.from_numpy(np.stack([
        _gauss(rng.uniform(0, size, (6 if corners else 0, 2)), size) for _ in range(B)]))
    t = {"instances": inst, "corner": corner,
         "direction": (torch.rand(B, 2, size, size, generator=g) * 2 - 1).half(),
         "direction_mask": torch.rand(B, size, size, generator=g) < (0.2 if directions else 0)}
    out = {k: torch.randn(B, c, size, size, generator=g) * 2 for k, c in
           (("extent", 1), ("boundary", 1), ("distance", 1), ("corner", 1), ("direction", 2))}
    return out, t


def _as_e030(t):
    return {("inst" if k == "instances" else k): v for k, v in t.items()}


CASES = [dict(seed=0), dict(seed=1), dict(seed=2, corners=False),
         dict(seed=3, directions=False)]


@pytest.mark.parametrize("case", CASES, ids=[str(c) for c in CASES])
def test_corner_line_loss_equals_e030(exp, case):
    out, t = _corner_batch(**case)
    task = CornerLineTask()
    ours = task.loss(out, task.targets(t))
    ref = exp.CornerLineTask().loss(out, _as_e030(t))
    print(f"corner_lines {case}: ours {float(ours):.8f} E030 {float(ref):.8f}")
    torch.testing.assert_close(ours, ref, rtol=1e-6, atol=1e-6)
    parts = task.losses(out, t)
    assert float(parts["corner"]) > 0 and float(ours) > 0


def test_corner_line_evaluator_matches_e030(exp):
    ev, ref = CornerLineTask().evaluator(), exp.CornerLineTask().evaluator()
    for seed in range(3):
        out, t = _corner_batch(seed)
        out["extent"] = (t["instances"][:, None] > 0).float() * 8 - 4  # near-perfect outputs
        ev.update(out, t)
        ref.update(out, _as_e030(t))
    got, want = ev.compute(), ref.compute()
    assert got["pq"] == want["pq"] and got["pq"] > 0
    assert got["instance_loss"] == pytest.approx(want["loss"], rel=1e-6)
    assert got["loss"] > got["instance_loss"]  # + corner + direction terms


def _road_batch(seed, B=2, size=40, crossings=True, roads=True):
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    kp = np.stack([np.stack([_gauss(rng.uniform(0, size, (n, 2)), size)
                             for n in (3, 2, 1 if crossings else 0)]) for _ in range(B)])
    mask = (torch.rand(B, size, size, generator=g) < (0.3 if roads else 0)).float()
    d = torch.rand(B, 4, size, size, generator=g) * 2 - 1
    second = torch.rand(B, 1, size, size, generator=g) < (0.05 if crossings else 0)
    d[:, 2:] *= second
    t = {"mask": mask, "line": mask * (torch.rand(B, size, size, generator=g) < 0.5),
         "keypoints": torch.from_numpy(kp), "direction": d}
    out = {k: torch.randn(B, c, size, size, generator=g) * 2 for k, c in
           (("mask", 1), ("line", 1), ("keypoints", 3), ("direction", 4))}
    return out, t


def _as_e044(t):
    return {("kp" if k == "keypoints" else k): v for k, v in t.items()}


def _kp_out(out):
    return {("kp" if k == "keypoints" else k): v for k, v in out.items()}


ROAD_CASES = [dict(seed=0), dict(seed=1, w=(1.0, 0.5, 2.0, 0.3)),
              dict(seed=2, crossings=False), dict(seed=3, roads=False)]


@pytest.mark.parametrize("case", ROAD_CASES, ids=[str(c) for c in ROAD_CASES])
def test_road_loss_equals_e044(exp, case):
    case = dict(case)
    w = case.pop("w", (1.0, 1.0, 1.0, 1.0))
    out, t = _road_batch(**case)
    ours = RoadGraphTask(weights=w).loss(out, t)
    ref = exp.RoadTask(w).loss(_kp_out(out), _as_e044(t))
    print(f"roads {case} w={w}: ours {float(ours):.8f} E044 {float(ref):.8f}")
    torch.testing.assert_close(ours, ref, rtol=1e-6, atol=1e-6)


def test_road_evaluator_matches_e044(exp):
    ev, ref = RoadGraphTask().evaluator(), exp.RoadTask().evaluator()
    for seed in range(3):
        out, t = _road_batch(seed)
        ev.update(out, t)
        ref.update(_kp_out(out), _as_e044(t))
    got, want = ev.compute(), ref.compute()
    assert got["mask_iou"] == want["mask_iou"] and 0 < got["mask_iou"] < 1
    assert got["loss"] == pytest.approx(want["loss"], rel=1e-6)


def test_tasks_registered_and_full_resolution_only():
    assert isinstance(build_task("corner_lines", 1), CornerLineTask)
    assert isinstance(build_task("roads", 1, weights=(1, 1, 2, 1)), RoadGraphTask)
    assert CornerLineTask().direction_targets == RoadGraphTask().direction_targets == (
        "direction",)
    assert set(RoadGraphTask().heads()) == {"mask", "line", "keypoints", "direction"}
    out, t = _road_batch(0, size=40)
    half = {k: v[..., ::2, ::2] for k, v in out.items()}
    with pytest.raises(ValueError, match="full resolution"):
        RoadGraphTask().loss(half, t)


# --- loader: dihedral augmentation of direction targets ------------------------------------------

def _corner_sample(size=48):
    rng = np.random.default_rng(5)
    polys = []
    for _ in range(4):
        cx, cy = rng.uniform(10, size - 10, 2)
        p = shapely.box(cx - 5.3, cy - 3.7, cx + 5.1, cy + 4.2)
        polys.append(shapely.affinity.rotate(p, rng.uniform(0, 180), origin="centroid"))
    return polys, rng.integers(0, 256, (size, size, 3)).astype(np.uint8)


def _transforms_seen(loader, x0, y0, encode, size):
    """Every training sample is some dihedral transform of the sample, and its labels equal
    ``encode(d)``, the targets encoded from the geometry transformed by ``d``."""
    seen = set()
    for x, y in loader:
        for i in range(x.shape[0]):
            ds = [d for d in range(8) if torch.equal(x[i], dihedral(x0, d))]
            assert len(ds) == 1
            want = encode(ds[0])
            for k, v in y.items():
                w = torch.as_tensor(want[k])
                assert v.dtype == (torch.bool if w.dtype == torch.bool else
                                   torch.long if not w.is_floating_point() else torch.float32), k
                torch.testing.assert_close(v[i].to(w.dtype), w, atol=1e-6, rtol=0,
                                           msg=f"{k}, d={ds[0]}")
            seen.add(ds[0])
    return seen


def test_dihedral_loader_corner_targets():
    size = 48
    polys, img = _corner_sample(size)

    def encode(d):
        t = corner_targets([dihedral_geometry(p, d, (size, size)) for p in polys], (size, size))
        return {"corner": t["corner"], "direction": t["direction"],
                "direction_mask": t["direction_mask"]}

    labels = {k: np.repeat(v[None], 16, 0) for k, v in encode(0).items()}
    common = dict(mean=[0.0] * 3, std=[1.0] * 3, device=torch.device("cpu"))
    x0 = next(iter(GPULoader(img[None], {k: v[:1] for k, v in labels.items()}, 1,
                             train=False, **common)))[0][0]
    loader = GPULoader(np.repeat(img[None], 16, 0), labels, 8, train=True, seed=0,
                       dihedral=True, directions=("direction",), **common)
    seen = set()
    for _ in range(3):
        seen |= _transforms_seen(loader, x0, None, encode, size)
    assert seen == set(range(8))


def test_dihedral_loader_road_targets():
    size = 40
    g = nx.Graph()
    for n, p in {0: (19.3, 21.1), 1: (0.0, 23.7), 2: (35.2, 8.4), 3: (17.9, 40.0),
                 4: (31.6, 33.2), 5: (8.1, 0.0), 6: (27.7, 40.0)}.items():
        g.add_node(n, xy=np.array(p))
    g.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (5, 6)], width=5.0)
    cross = [[16.3, 15.2]]

    def encode(d):
        h = nx.Graph()
        h.add_nodes_from((n, {"xy": dihedral_xy(a["xy"], d, (size, size))})
                         for n, a in g.nodes(data=True))
        h.add_edges_from(g.edges(data=True))
        return road_targets(h, (size, size), crossings=dihedral_xy(cross, d, (size, size)))

    t0 = encode(0)
    assert (np.abs(t0["direction"][2:]).sum(0) > 0).sum() > 5  # a two-mode crossing
    img = np.random.default_rng(0).integers(0, 256, (size, size, 3)).astype(np.uint8)
    labels = {k: np.repeat(v[None], 16, 0).astype(np.float16) for k, v in t0.items()}
    common = dict(mean=[0.0] * 3, std=[1.0] * 3, device=torch.device("cpu"))
    x0 = next(iter(GPULoader(img[None], {k: v[:1] for k, v in labels.items()}, 1,
                             train=False, **common)))[0][0]
    loader = GPULoader(np.repeat(img[None], 16, 0), labels, 8, train=True, seed=1,
                       dihedral=True, directions=("direction",), device_resident=False,
                       **common)

    def encode16(d):  # stored as float16, as a cache would
        return {k: v.astype(np.float16).astype(np.float32) for k, v in encode(d).items()}

    seen = set()
    for _ in range(3):
        seen |= _transforms_seen(loader, x0, None, encode16, size)
    assert seen == set(range(8))


def test_loader_dict_labels_affine_and_errors():
    n, size = 4, 16
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (n, size, size, 3)).astype(np.uint8)
    labels = {"ids": rng.integers(0, 5, (n, size, size)).astype(np.int32),
              "soft": rng.uniform(0, 1, (n, 2, size, size)).astype(np.float32),
              "flag": rng.uniform(0, 1, (n, size, size)) < 0.5}
    common = dict(mean=[0.0] * 3, std=[1.0] * 3, device=torch.device("cpu"))
    x, y = next(iter(GPULoader(img, labels, 4, train=True, scale=(0.5, 1.0), **common)))
    assert y["ids"].dtype == torch.long and y["flag"].dtype == torch.bool
    assert y["soft"].dtype == torch.float32
    # nearest resampling: values come from the label, not rounded
    assert set(np.unique(y["soft"].numpy())) <= set(np.unique(labels["soft"]))
    assert not torch.equal(y["soft"], y["soft"].round())
    with pytest.raises(ValueError, match="dihedral"):
        GPULoader(img, labels, 4, train=True, directions=("soft",), **common)
    with pytest.raises(ValueError, match="not label entries"):
        GPULoader(img, labels, 4, train=True, dihedral=True, directions=("dir",), **common)
    with pytest.raises(ValueError, match="square"):
        GPULoader(img[:, :, :12], {k: v[..., :12] for k, v in labels.items()}, 4, train=True,
                  dihedral=True, **common)
    GPULoader(img, labels, 4, train=False, directions=("soft",), **common)  # eval: no augment


def test_split_with_label_dict_round_trips(tmp_path):
    imgs = np.zeros((3, 8, 8, 3), np.uint8)
    labels = {"mask": np.ones((3, 8, 8), np.float16),
              "direction": np.zeros((3, 4, 8, 8), np.float16)}
    save_split(tmp_path, "train", imgs, labels, ["a", "b", "c"])
    s = load_split(tmp_path, "train")
    assert set(s.masks) == {"mask", "direction"}
    assert s.masks["direction"].shape == (3, 4, 8, 8)


# --- end to end: run_training on synthetic caches ------------------------------------------------

def _register(monkeypatch, name, task):
    info = DatasetInfo(name=name, num_classes=1, in_channels=3, class_names=("object",),
                       mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25))
    ds = SimpleNamespace(INFO=info, TASK=task, EVAL_BATCH_SIZE=2)
    monkeypatch.setitem(bench_datasets.DATASETS, name, ds)


def _corner_cache(root, n=4, size=32):
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 60, (n, size, size, 3)).astype(np.uint8)
    inst = np.zeros((n, size, size), np.int32)
    polys = [shapely.box(4, 4, 14, 12), shapely.box(18, 6, 28, 20), shapely.box(6, 18, 15, 28)]
    for k, p in enumerate(polys):
        x0, y0, x1, y1 = (int(v) for v in p.bounds)
        inst[:, y0:y1, x0:x1] = k + 1
        imgs[:, y0:y1, x0:x1] += 150
    t = corner_targets(polys, (size, size))
    labels = {"instances": inst, "corner": np.repeat(t["corner"][None], n, 0),
              "direction": np.repeat(t["direction"][None], n, 0).astype(np.float16),
              "direction_mask": np.repeat(t["direction_mask"][None], n, 0)}
    for split in ("train", "val"):
        save_split(root / "cache" / "synthetic_corners", split, imgs, labels,
                   [str(i) for i in range(n)])


def _road_cache(root, n=4, size=32):
    g = nx.Graph()
    for k, p in enumerate([(0.0, 16.2), (32.0, 15.1), (15.3, 0.0), (16.6, 32.0)]):
        g.add_node(k, xy=np.array(p))
    g.add_edges_from([(0, 1), (2, 3)], width=4.0)
    t = road_targets(g, (size, size), crossings=[[16.0, 15.6]])
    imgs = np.repeat((t["mask"][..., None] * 180 + 20).astype(np.uint8), 3, -1)
    labels = {k: np.repeat(v[None], n, 0).astype(np.float16) for k, v in t.items()}
    for split in ("train", "val"):
        save_split(root / "cache" / "synthetic_roads", split, np.repeat(imgs[None], n, 0),
                   labels, [str(i) for i in range(n)])


@pytest.mark.parametrize("name,task", [("synthetic_corners", "corner_lines"),
                                       ("synthetic_roads", "roads")])
def test_run_training_end_to_end(tmp_path, monkeypatch, name, task):
    from manyseg_bench.experiments import run_training
    from manyseg_bench.providers import ModelSpec

    _register(monkeypatch, name, task)
    (_corner_cache if task == "corner_lines" else _road_cache)(tmp_path)
    rec = run_training("manyseg", ModelSpec("unet", "resnet18", weights=None), name, "smoke",
                       device="cpu", out_dir=tmp_path / "runs", data=tmp_path,
                       recipe_overrides={"precision": "fp32", "batch_size": 2})
    heads = rec["spec"]["options"]["heads"]
    prior = "corner" if task == "corner_lines" else "keypoints"
    assert heads[prior]["bias_init"] == pytest.approx(np.log(0.1 / 0.9))
    assert rec["primary_metric"] in rec["final_val"]
    assert np.isfinite(rec["history"][0]["train_loss"])
