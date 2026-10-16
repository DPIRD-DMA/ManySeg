"""Change-detection dataset preparation on small fake copies of each source, and end-to-end
runs (binary change with a Siamese difference fusion; semantic change with per-input heads)."""

import io
import json
import zipfile

import numpy as np
import pytest
from manyseg_bench.datasets import change_detection as cd
from manyseg_bench.datasets import load_split
from manyseg_bench.providers import ModelSpec

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def png(a):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return {"bytes": buf.getvalue(), "path": None}


def fake_parquet(root, split, n, size, semantic=False):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        cm = np.zeros((size, size), np.uint8)
        cm[:size // 2, :size // 2] = 255
        r = {"t1_image": png(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)),
             "t2_image": png(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)),
             "change_mask": png(cm), "image_name": f"{split}_{i}.png"}
        if semantic:
            lc = np.where(cm > 0, 1 + i % 6, 0).astype(np.uint8)
            r["t1_mask"], r["t2_mask"] = png(lc), png(np.where(cm > 0, 2, 0).astype(np.uint8))
        rows.append(r)
    (root / "data").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), root / "data" / f"{split}-00000-of-00001.parquet")


def test_levircd_prepare_tiles(tmp_path):
    raw, cache = tmp_path / "raw" / "levircd", tmp_path / "cache" / "levircd"
    for s, n in (("train", 2), ("val", 1), ("test", 1)):
        fake_parquet(raw, s, n, 512)
    summary = cd.prepare_levircd(raw, cache, expected=None)
    tr = load_split(cache, "train")
    assert summary["train"]["tiles"] == 8 and tr.images["t1"].shape == (8, 256, 256, 3)
    assert set(np.unique(tr.masks)) == {0, 1}


def test_second_prepare_label_stack(tmp_path):
    raw, cache = tmp_path / "raw" / "second", tmp_path / "cache" / "second"
    fake_parquet(raw, "train", 12, 512, semantic=True)
    fake_parquet(raw, "test", 2, 512, semantic=True)
    s = cd.prepare_second(raw, cache)
    assert s["train"]["n"] + s["val"]["n"] == 12 and s["test"]["n"] == 2
    tr = load_split(cache, "train")
    assert tr.masks.shape[1:] == (3, 512, 512) and tr.masks[:, 2].max() == 1
    assert s["train"]["lc_vs_change_agreement"] == 1.0


def test_oscd_prepare(tmp_path, monkeypatch):
    tifffile = pytest.importorskip("tifffile")
    raw = tmp_path / "raw" / "oscd"
    raw.mkdir(parents=True)
    monkeypatch.setattr(cd, "OSCD_TRAIN", ("a", "b"))
    monkeypatch.setattr(cd, "OSCD_VAL", ("b",))
    monkeypatch.setattr(cd, "OSCD_TEST", ("c",))
    rng = np.random.default_rng(0)

    def tif(a):
        buf = io.BytesIO()
        tifffile.imwrite(buf, a)
        return buf.getvalue()

    with zipfile.ZipFile(raw / "Onera Satellite Change Detection dataset - Images.zip",
                         "w") as zi:
        for c in ("a", "b", "c"):
            for d in ("imgs_1_rect", "imgs_2_rect"):
                for b in cd.OSCD_BANDS:
                    zi.writestr(f"{cd._IMG}{c}/{d}/{b}.tif",
                                tif(rng.integers(0, 5000, (150, 140)).astype(np.uint16)))
    for kind, cs in (("Train", ("a", "b")), ("Test", ("c",))):
        root = f"Onera Satellite Change Detection dataset - {kind} Labels"
        with zipfile.ZipFile(raw / f"{root}.zip", "w") as zl:
            for c in cs:
                cm = np.ones((150, 140), np.uint8)
                cm[:20] = 2
                zl.writestr(f"{root}/{c}/cm/{c}-cm.tif", tif(cm))
    s = cd.prepare_oscd(raw, tmp_path / "cache" / "oscd")
    assert s["train"]["tiles"] == 4 and s["val"]["cities"] == ["b"]
    tr = load_split(tmp_path / "cache" / "oscd", "train")
    assert tr.images["t1"].shape == (4, 128, 128, 13) and 255 in tr.masks
    meta = json.loads((tmp_path / "cache" / "oscd" / "meta.json").read_text())
    assert len(meta["mean"]["t1"]) == 13


@pytest.mark.parametrize("dataset,mode,options,task_kwargs", [
    ("levircd", "dict", {"fusion": {"type": "intermediate", "op": "diff"},
                         "input_options": {"t2": {"backbone": "shared:t1", "share_norm": True}}},
     {}),
    ("levircd", "stack", {}, {}),
    ("second", "dict", {"fusion": {"type": "intermediate", "op": "concat_diff"},
                        "input_options": {"t2": {"backbone": "shared:t1", "share_norm": True}}},
     {"per_input": True, "share": True}),
    ("second", "stack", {}, {"per_input": False}),
])
def test_run_training_change(tmp_path, dataset, mode, options, task_kwargs):
    from manyseg_bench.experiments import run_training

    raw = tmp_path / "raw" / dataset
    if dataset == "levircd":
        for s, n in (("train", 1), ("val", 1), ("test", 1)):
            fake_parquet(raw, s, n, 512)
        cd.prepare_levircd(raw, tmp_path / "cache" / dataset, expected=None)
    else:
        fake_parquet(raw, "train", 12, 512, semantic=True)
        fake_parquet(raw, "test", 2, 512, semantic=True)
        cd.prepare_second(raw, tmp_path / "cache" / dataset)
    spec = ModelSpec("unet", "resnet18", weights=None, input_mode=mode, options=options)
    rec = run_training("manyseg", spec, dataset, "smoke", device="cpu", data=tmp_path,
                       out_dir=tmp_path / "runs", evaluate_test=True, task_overrides=task_kwargs,
                       recipe_overrides={"batch_size": 2, "precision": "fp32"})
    pm = rec["primary_metric"]
    assert pm == ("f1_change" if dataset == "levircd" else "score")
    assert np.isfinite(rec["test"][pm])
