"""BigEarthNet v2 preparation (fake LMDB + metadata + reference maps), variants and the joint
segmentation + classification task."""

import json

import numpy as np
import pytest
import torch
from manyseg_bench.datasets import bigearthnet as ben
from manyseg_bench.datasets import get_dataset, load_split
from manyseg_bench.tasks import build_task

lmdb = pytest.importorskip("lmdb")
pd = pytest.importorskip("pandas")
tifffile = pytest.importorskip("tifffile")
st = pytest.importorskip("safetensors.numpy")

CODES = [(311, 312), (211, 999), (523, 423), (112, 122), (241, 231), (313,)]


def pid(i):
    return f"S2A_MSIL2A_2017{6 + i % 3:02d}13T101031_N9999_R022_T33UUP_{26 + i}_57"


def fake_raw(raw, n=6):
    rng = np.random.default_rng(0)
    rows, maps = [], {}
    env = lmdb.open(str(raw / "BENv2.lmdb"), map_size=2**28)
    with env.begin(write=True) as txn:
        for i in range(n):
            p, s1 = pid(i), f"S1B_IW_{i}"
            bands = {b: rng.integers(0, 3000, (120 // (r // 10),) * 2).astype(np.uint16)
                     for b, r in ben.S2_NATIVE.items()}
            txn.put(p.encode(), st.save(bands))
            txn.put(s1.encode(), st.save({b: rng.normal(-12, 3, (120, 120)).astype(np.float32)
                                          for b in ben.S1_BANDS}))
            m = np.full((120, 120), CODES[i][0], np.uint16)
            m[:, 60:] = CODES[i][-1]
            maps[p] = m
            path = ben.reference_map_path(raw, p)
            path.parent.mkdir(parents=True)
            tifffile.imwrite(path, m)
            labels = sorted({ben.CLASSES[ben.CLC_TO_CLASS[c]] for c in CODES[i]
                             if c in ben.CLC_TO_CLASS})
            rows.append({"patch_id": p, "labels": labels, "split": ["train", "validation",
                                                                     "test"][i % 3],
                         "country": ben.COUNTRIES[i % 4], "s1_name": s1})
    pd.DataFrame(rows).to_parquet(raw / "metadata.parquet")
    return rows, maps


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    raw, cache = tmp_path / "raw" / "bigearthnet", tmp_path / "cache" / "bigearthnet"
    raw.mkdir(parents=True)
    rows, maps = fake_raw(raw)
    monkeypatch.setattr(ben, "SUBSET", {"train": 2, "val": 2, "test": 2})
    summary = ben.prepare(raw, cache)
    return raw, cache, rows, maps, summary


def test_prepare_arrays_and_label_agreement(prepared):
    raw, cache, rows, maps, summary = prepared
    tr = load_split(cache, "train")
    assert tr.images["s2"].shape == (2, 120, 120, 12) and tr.images["s1"].shape[-1] == 2
    assert all(summary[s]["label_agreement"] == 1.0 for s in ("train", "val", "test"))
    # 60 m band B01 (index 0) repeated in 6x6 blocks; block average recovers it
    b01 = tr.images["s2"][0, :, :, 0].astype(np.float32)
    assert np.all(b01.reshape(20, 6, 20, 6).std(axis=(1, 3)) == 0)
    p = tr.names[0]
    assert np.array_equal(tr.masks[0], ben.map_codes(maps[p]))
    meta = json.loads((cache / "meta.json").read_text())
    assert set(meta["mean"]) == {"s2", "s1"} and len(meta["mean"]["s2"]) == 12


def test_clc_mapping_quirks():
    assert ben.map_codes(np.array([241, 423, 421, 999, 122, 112, 523])).tolist() == \
        [3, 255, 16, 255, 255, 0, 18]


def test_variant_labels(prepared):
    from manyseg_bench.experiments.runner import split_getter

    raw, cache, rows, *_ = prepared
    by_id = {r["patch_id"]: r for r in rows}
    cls = split_getter(get_dataset("bigearthnet"), cache, raw)("train")
    assert cls.masks.shape == (2, 19) and cls.masks.dtype == np.uint8
    for name, hot in zip(cls.names, cls.masks):
        assert {ben.CLASSES[i] for i in np.flatnonzero(hot)} == set(by_id[name]["labels"])
    seg = split_getter(get_dataset("bigearthnet_seg"), cache, raw)("train")
    assert seg.masks.shape == (2, 120, 120)
    frac = split_getter(get_dataset("bigearthnet_frac"), cache, raw)("train")
    assert frac.masks.dtype == np.float32 and np.allclose(frac.masks.sum(1), 1.0)
    meta = split_getter(get_dataset("bigearthnet_meta"), cache, raw)("val")
    assert meta.vectors["meta"].shape == (2, 2)
    assert get_dataset("bigearthnet_seg").TASK == "seg_multilabel"
    assert get_dataset("bigearthnet_frac").TASK == "regression"
    assert set(split_getter(get_dataset("bigearthnet_s1s2"), cache, raw)("test").images) == \
        {"s2", "s1"}


def test_load_vectors_doy_and_country(prepared):
    raw, _, rows, *_ = prepared
    v = ben.load_vectors(raw, [rows[1]["patch_id"]], {"m": {"fields": ben.META_FIELDS}})["m"]
    assert v.tolist() == [[31 + 28 + 31 + 30 + 31 + 30 + 13, 1.0]]  # 13 July; Belgium


def test_shuffled_metadata_is_a_row_permutation(prepared):
    raw, _, rows, *_ = prepared
    names = [r["patch_id"] for r in rows]
    real = ben.load_vectors(raw, names, {"m": {"fields": ben.META_FIELDS}})["m"]
    shuf = ben.load_vectors(raw, names, get_dataset("bigearthnet_meta_shuffled").INFO.vectors)
    shuf = shuf["meta"]
    perm = np.random.default_rng(ben.SHUFFLE_SEED).permutation(len(names))
    assert np.array_equal(shuf, real[perm]) and not np.array_equal(shuf, real)


def test_seg_multilabel_targets_ignore_and_class_zero():
    t = build_task("seg_multilabel", 4, 255)
    y = torch.tensor([[[0, 255], [2, 2]], [[255, 255], [3, 3]]])
    tg = t.targets(y)
    assert tg["labels"].tolist() == [[1, 0, 1, 0], [0, 0, 0, 1]]
    out = {"seg": torch.randn(2, 4, 2, 2), "logits": torch.randn(2, 4)}
    assert t.loss(out, tg).ndim == 0
    ev = t.evaluator()
    ev.update(out, y)
    m = ev.compute()
    assert "map_macro" in m and "miou" in m["seg"]
    assert t.heads()["logits"]["scope"] == "image" and t.primary_metric == "map_macro"


def test_subset_is_deterministic_and_sorted():
    df = pd.DataFrame({"patch_id": [f"p{i}" for i in range(50)], "split": ["train"] * 50})
    a, b = ben.subset_ids(df, "train", 10), ben.subset_ids(df, "train", 10)
    assert a == b == sorted(a) and len(a) == 10
