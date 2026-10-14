"""AI4Boundaries (orthophoto) preparation on a tiny synthetic GeoTIFF tree."""

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from manyseg_bench.datasets import ai4boundaries  # noqa: E402
from manyseg_bench.datasets.base import load_split  # noqa: E402


def _write_tile(root, split, name, k, size=32):
    img = np.full((size, size, 3), 50 + k, np.uint8)
    ids = np.full((size, size), -10000, np.float32)
    ids[2:12, 2:14] = 2
    ids[2:12, 14:28] = 3
    ids[18:30, 4:20] = 4
    ext = (ids > 0).astype(np.float32)
    mask = np.stack([ext, np.zeros_like(ext), ext * 0.5, ids], -1)
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "masks" / split).mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(root / "images" / split / f"{name}_ortho_1m_512.tif", img)
    tifffile.imwrite(root / "masks" / split / f"{name}_ortholabel_1m_512.tif", mask)


@pytest.fixture
def raw(tmp_path):
    root = tmp_path / "raw" / "orthos"
    for split, n in (("train", 3), ("val", 2), ("test", 2)):
        for i in range(n):
            _write_tile(root, split, f"{['NL', 'AT'][i % 2]}_{split}{i}", i)
    return tmp_path / "raw"


def test_prepare_instance_ids_and_splits(raw, tmp_path):
    summary = ai4boundaries.prepare(raw, tmp_path / "cache", size=32, expected=None)
    assert {k: v["n"] for k, v in summary.items()} == {"train": 3, "val": 2, "test": 2}
    s = load_split(tmp_path / "cache", "train")
    assert s.images.shape == (3, 32, 32, 3) and s.masks.dtype == np.int32
    m = np.asarray(s.masks[0])
    assert set(np.unique(m).tolist()) == {0, 1, 2, 3}  # background + 3 fields, sequential
    assert summary["train"]["extent_agreement"] == 1.0
    assert s.names[0].split("_")[0] in ("NL", "AT")


def test_integrity_rejects_wrong_counts(raw, tmp_path):
    with pytest.raises(RuntimeError, match="integrity"):
        ai4boundaries.prepare(raw, tmp_path / "cache", size=32, expected={"train": 5319})
