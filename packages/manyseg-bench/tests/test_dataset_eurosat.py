"""EuroSAT preparation on a tiny synthetic archive."""

import io
import json
import zipfile

import numpy as np
import pytest
from manyseg_bench.datasets import eurosat, get_dataset, load_split

tifffile = pytest.importorskip("tifffile")


def fake_raw(raw, per_split=2):
    raw.mkdir(parents=True)
    rng = np.random.default_rng(0)
    names = {s: [f"{eurosat.CLASSES[i % 10]}_{s}{i}" for i in range(per_split)]
             for s in eurosat.EXPECTED}
    with zipfile.ZipFile(raw / "EuroSATallBands.zip", "w") as z:
        for split, ns in names.items():
            for n in ns:
                buf = io.BytesIO()
                tifffile.imwrite(buf, rng.integers(0, 4000, (64, 64, 13)).astype(np.uint16))
                z.writestr(f"ds/tif/{n.rsplit('_', 1)[0]}/{n}.tif", buf.getvalue())
            (raw / f"eurosat-spatial-{split}.txt").write_text("\n".join(f"{n}.jpg" for n in ns))
    return names


def test_prepare(tmp_path, monkeypatch):
    names = fake_raw(tmp_path / "raw")
    monkeypatch.setattr(eurosat, "EXPECTED", {s: 2 for s in eurosat.EXPECTED})
    summary = eurosat.prepare(tmp_path / "raw", tmp_path / "cache")
    tr = load_split(tmp_path / "cache", "train")
    assert tr.images.shape == (2, 64, 64, 13) and tr.images.dtype == np.float16
    assert tr.masks.tolist() == [0, 1] and tr.names == names["train"]
    assert float(tr.images.max()) < 0.41  # DN / 1e4
    meta = json.loads((tmp_path / "cache" / "meta.json").read_text())
    assert len(meta["mean"]) == 13 and summary["test"]["n"] == 2
    assert get_dataset("eurosat").TASK == "classification"


def test_prepare_rejects_missing_patches(tmp_path, monkeypatch):
    fake_raw(tmp_path / "raw")
    monkeypatch.setattr(eurosat, "EXPECTED", {s: 2 for s in eurosat.EXPECTED})
    (tmp_path / "raw" / "eurosat-spatial-val.txt").write_text("Forest_x.jpg\nForest_y.jpg")
    with pytest.raises(RuntimeError, match="missing"):
        eurosat.prepare(tmp_path / "raw", tmp_path / "cache")
