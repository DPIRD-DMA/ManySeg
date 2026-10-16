"""Time-series inputs in the bench: (N, T, H, W, C) arrays through GPULoader (dict and stack
mode), per-series mask/dates entries, the provider, and an end-to-end run."""

import numpy as np
import pytest
import torch
from manyseg_bench.datasets import GPULoader
from manyseg_bench.providers import ModelSpec, get_provider


def arrays(n=4, t=3, h=16, w=16, c=2):
    rng = np.random.default_rng(0)
    s2 = rng.normal(size=(n, t, h, w, c)).astype(np.float16)
    dem = rng.normal(size=(n, h, w, 1)).astype(np.float16)
    masks = rng.integers(0, 2, (n, h, w)).astype(np.uint8)
    return {"s2": s2, "dem": dem}, masks


def loader(train, stack=False, **kw):
    imgs, masks = arrays()
    return GPULoader(imgs, masks, 2, {"s2": [0.0, 0.0], "dem": [0.0]},
                     {"s2": [1.0, 1.0], "dem": [1.0]}, torch.device("cpu"), train=train,
                     stack=stack, drop_last=False, **kw)


def test_dict_mode_yields_series_and_rasters():
    (x, y), *_ = loader(False)
    assert x["s2"].shape == (2, 3, 2, 16, 16) and x["dem"].shape == (2, 1, 16, 16)
    imgs, _ = arrays()
    torch.testing.assert_close(x["s2"], torch.from_numpy(imgs["s2"][:2]).float()
                               .permute(0, 1, 4, 2, 3))


def test_stack_mode_flattens_dates_into_channels():
    (x, _), *_ = loader(False, stack=True)
    assert x.shape == (2, 3 * 2 + 1, 16, 16)  # dates x bands, then the dem


def test_augmentation_moves_every_date_and_the_mask_together():
    """Same geometric transform for all dates, other inputs and the labels."""
    imgs, masks = arrays()
    imgs["s2"][:] = imgs["s2"][:, :1]  # identical dates
    imgs["dem"][..., 0] = imgs["s2"][:, 0, ..., 0]
    ld = GPULoader(imgs, masks, 4, {"s2": [0.0, 0.0], "dem": [0.0]},
                   {"s2": [1.0, 1.0], "dem": [1.0]}, torch.device("cpu"), train=True,
                   stack=False, out_size=(12, 12), seed=3)
    (x, y), = list(ld)
    for t in range(1, 3):
        torch.testing.assert_close(x["s2"][:, t], x["s2"][:, 0])
    torch.testing.assert_close(x["dem"][:, 0], x["s2"][:, 0, 0])
    assert y.shape == (4, 12, 12)


def test_series_extras_pass_as_vectors_in_dict_mode():
    dates = np.arange(12, dtype=np.float32).reshape(4, 3)
    (x, _), *_ = loader(False, vectors={"s2_dates": dates})
    assert torch.equal(x["s2_dates"], torch.from_numpy(dates[:2]))


def test_provider_builds_timeseries_inputs():
    spec = ModelSpec("unet", "resnet18", weights=None, num_classes=3, inputs={"s2": 2, "dem": 1},
                     input_mode="dict", timeseries=["s2"],
                     options={"input_options": {"s2": {"temporal": "attention"},
                                                "dem": {"alignment": "same_extent"}}})
    model = get_provider("manyseg").build(spec).model.eval()
    out = model({"s2": torch.randn(1, 3, 2, 64, 64), "dem": torch.randn(1, 1, 32, 32),
                 "s2_dates": torch.tensor([[1.0, 50.0, 90.0]])})
    assert out.shape == (1, 3, 64, 64)


DATES = ["20181120", "20190105", "20190220", "20190410", "20190525", "20190710", "20190820",
         "20191005"]


def fake_pastis(root, patches=(("101", "t30uxv"), ("102", "t31tfj"), ("103", "t31tfj"))):
    import json

    from manyseg_bench.datasets import pastis
    from manyseg_bench.datasets.base import save_split

    cache = root / "cache" / "pastis"
    rng = np.random.default_rng(0)
    for split in ("train", "val", "test"):
        names, imgs, masks = [], [], []
        for p, tile in patches:
            lab = rng.integers(0, 19, (32, 32)).astype(np.uint8)
            for d in reversed(DATES):  # unordered on purpose
                names.append(pastis.sample_name(p, tile, d))
                imgs.append(np.full((32, 32, 10), int(d[4:6]), np.float16))  # value = month
                masks.append(lab)
        save_split(cache, split, {"s2": np.stack(imgs)}, np.stack(masks), names)
    (cache / "meta.json").write_text(json.dumps({"mean": {"s2": [0] * 10},
                                                 "std": {"s2": [1] * 10}}))
    return root


def test_pastis_ts_prepare_orders_dates(tmp_path):
    from manyseg_bench.datasets import load_split, pastis_ts

    root = fake_pastis(tmp_path)
    pastis_ts.prepare(root / "raw" / "pastis_ts", root / "cache" / "pastis_ts")
    s = load_split(root / "cache" / "pastis_ts", "train")
    assert s.images["s2"].shape == (3, 8, 32, 32, 10) and s.masks.shape == (3, 32, 32)
    months = s.images["s2"][0, :, 0, 0, 0].astype(int).tolist()
    assert months == [11, 1, 2, 4, 5, 7, 8, 10]  # ordered through the season
    dates = np.load(root / "cache" / "pastis_ts" / "train_dates.npy")
    assert dates.shape == (3, 8) and (np.diff(dates, axis=1) > 0).all() and dates[0, 0] == 19


@pytest.mark.parametrize("dataset,mode,temporal", [
    ("pastis_ts", "dict", "attention"), ("pastis_ts", "stack", None),
    ("pastis_ts_single", "stack", None)])
def test_run_training_on_series(tmp_path, dataset, mode, temporal):
    from manyseg_bench.datasets import pastis_ts
    from manyseg_bench.experiments import run_training

    root = fake_pastis(tmp_path)
    pastis_ts.prepare(root / "raw" / "pastis_ts", root / "cache" / "pastis_ts")
    opts = {"input_options": {"s2": {"temporal": temporal}}} if temporal else {}
    spec = ModelSpec("unet", "resnet18", weights=None, input_mode=mode, options=opts)
    rec = run_training("manyseg", spec, dataset, "smoke", device="cpu", data=root,
                       out_dir=tmp_path / "runs", evaluate_test=True,
                       recipe_overrides={"batch_size": 2, "precision": "fp32"})
    expect = {("pastis_ts", "dict"): 10, ("pastis_ts", "stack"): 80,
              ("pastis_ts_single", "stack"): 10}[(dataset, mode)]
    assert rec["spec"]["in_channels"] == expect or rec["spec"]["inputs"] == {"s2": 10}
    assert rec["spec"]["timeseries"] == (["s2"] if mode == "dict" else None)
    assert 0 <= rec["test"]["miou"] <= 1 and rec["img_size"] == [32, 32]


def test_augmentation_keeps_labels_aligned_with_every_date():
    imgs, _ = arrays(n=4, h=32, w=32)
    imgs["s2"][:] = 0
    imgs["dem"][:] = 0
    masks = np.zeros((4, 32, 32), np.uint8)
    imgs["s2"][:, :, 5:15, 18:28, :] = 1  # a marker in every date ...
    imgs["dem"][:, 5:15, 18:28, :] = 1    # ... in the other raster ...
    masks[:, 5:15, 18:28] = 1             # ... and in the labels
    ld = GPULoader(imgs, masks, 4, {"s2": [0.0, 0.0], "dem": [0.0]},
                   {"s2": [1.0, 1.0], "dem": [1.0]}, torch.device("cpu"), train=True,
                   stack=False, out_size=(24, 24), vflip=True, seed=7)
    (x, y), = list(ld)
    lab = y == 1
    assert lab.any()
    for t in range(3):
        img = x["s2"][:, t, 0] > 0.5
        assert (img == lab).float().mean() > 0.97  # nearest vs bilinear edges only
    assert ((x["dem"][:, 0] > 0.5) == lab).float().mean() > 0.97


def fake_a4b(src, names=("AT_1", "FR_2")):
    import io
    import zipfile

    h5py = pytest.importorskip("h5py")
    tifffile = pytest.importorskip("tifffile")
    src.mkdir(parents=True)
    rng = np.random.default_rng(0)
    with zipfile.ZipFile(src / "masks.zip", "w") as zm:
        for split in ("train", "val", "test"):
            (src / f"{split}.csv").write_text(
                "image,country\n" + "".join(f"{n}_S2_10m_256.nc,{n[:2]}\n" for n in names))
            with zipfile.ZipFile(src / f"{split}.zip", "w") as zi:
                for n in names:
                    buf = io.BytesIO()
                    with h5py.File(buf, "w") as f:
                        for b in ("B2", "B3", "B4", "B8", "NDVI"):
                            a = rng.integers(1, 3000, (6, 256, 256)).astype(np.float32)
                            if b == "B2":
                                a[5] = 0  # August missing
                            f[b] = a
                        f["time"] = np.array([0, 31, 61, 92, 122, 153])
                    zi.writestr(f"{split}/{n}_S2_10m_256.nc", buf.getvalue())
                    m = np.zeros((256, 256, 4), np.float32)
                    m[20:60, 20:60, 3] = 7
                    m[20:60, 20:60, 0] = 1
                    m[..., 3][m[..., 3] == 0] = np.nan
                    t = io.BytesIO()
                    tifffile.imwrite(t, m)
                    zm.writestr(f"{split}/{n}_S2_10m_256.tif", t.getvalue())  # real naming


def test_ai4boundaries_s2_prepare_and_variants(tmp_path):
    from manyseg_bench.datasets import ai4boundaries_s2, get_dataset, load_split
    from manyseg_bench.experiments.runner import split_getter

    src = tmp_path / "raw" / "ai4boundaries" / "sentinel2"
    fake_a4b(src)
    cache = tmp_path / "cache" / "ai4boundaries_s2"
    s = ai4boundaries_s2.prepare(tmp_path / "raw" / "ai4boundaries_s2", cache, workers=1,
                                 expected=None)
    assert s["train"]["n"] == 2 and s["train"]["extent_agreement"] == 1.0
    tr = load_split(cache, "train")
    assert tr.images["s2"].shape == (2, 6, 256, 256, 4) and tr.masks.max() == 1
    get = split_getter(get_dataset("ai4boundaries_s2"), cache, tmp_path / "raw")
    sp = get("val")
    assert sp.vectors["s2_dates"][0].tolist() == [0, 31, 61, 92, 122, 153]
    assert sp.vectors["s2_mask"][0].tolist() == [1, 1, 1, 1, 1, 0]  # August B2 missing
    single = split_getter(get_dataset("ai4boundaries_s2_single"), cache, tmp_path / "raw")("val")
    assert single.images["s2"].shape == (2, 256, 256, 4) and single.vectors is None
