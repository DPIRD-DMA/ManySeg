"""OAM-TCD preparation on a tiny synthetic parquet that mirrors the real schema."""

import io
import json

import numpy as np
import pytest
import torch

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from manyseg_bench.datasets import oam_tcd  # noqa: E402
from manyseg_bench.datasets.base import load_split  # noqa: E402


def _png(arr):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _coco_rle(mask):
    """Uncompressed COCO RLE (column-major) of a binary mask."""
    flat = mask.flatten(order="F")
    counts, cur, n = [], 0, 0
    for v in flat:
        if v != cur:
            counts.append(n)
            n, cur = 0, v
        n += 1
    counts.append(n)
    return {"size": list(mask.shape), "counts": counts}


def _row(i, fold, size=64):
    img = np.full((size, size, 3), 40, np.uint8)
    ann = []
    # two trees (category 2) as polygons, one canopy group (category 1)
    for k, (y, x) in enumerate([(10, 10), (40, 44)]):
        img[y:y + 8, x:x + 8] = 200
        poly = [x, y, x + 8, y, x + 8, y + 8, x, y + 8]
        ann.append({"id": str(k), "image_id": str(i), "category_id": "2",
                    "segmentation": json.dumps([poly]), "area": "64", "bbox": "[]",
                    "iscrowd": "0", "extra": "{}"})
    canopy = np.zeros((size, size), np.uint8)
    canopy[30:60, 2:20] = 1
    ann.append({"id": "9", "image_id": str(i), "category_id": "1",
                "segmentation": json.dumps(_coco_rle(canopy)), "area": "540", "bbox": "[]",
                "iscrowd": "0", "extra": "{}"})
    return {"image_id": i, "image": {"bytes": _png(img), "path": f"{i}.tif"},
            "height": size, "width": size, "annotation": {"bytes": b"", "path": ""},
            "license": "CC-BY 4.0", "validation_fold": fold, "biome": 1,
            "coco_annotations": json.dumps(ann)}


@pytest.fixture
def raw(tmp_path):
    d = tmp_path / "raw" / "data"
    d.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([_row(i, i % 5) for i in range(10)]),
                   d / "train-00000-of-00001.parquet")
    pq.write_table(pa.Table.from_pylist([_row(100 + i, -1) for i in range(3)]),
                   d / "test-00000-of-00001.parquet")
    return tmp_path / "raw"


def test_prepare_splits_points_and_ignore(raw, tmp_path):
    summary = oam_tcd.prepare(raw, tmp_path / "cache", size=32, expected=None,
                              workers=1)
    assert summary["train"]["n"] == 8 and summary["val"]["n"] == 2  # fold 0 -> val
    assert summary["test"]["n"] == 3
    s = load_split(tmp_path / "cache", "train")
    assert s.images.shape == (8, 32, 32, 3) and s.masks.dtype == np.int16
    m = torch.from_numpy(np.asarray(s.masks[0]).astype(np.int64))
    assert set(m.unique().tolist()) == {-1, 0, 1, 2}  # canopy ignore, background, two trees
    from manyseg.points import centers_from_instances

    centers, _ = centers_from_instances(m.clamp(min=0))[0]
    torch.testing.assert_close(centers, torch.tensor([[6.75, 6.75], [21.75, 23.75]]),
                               atol=0.6, rtol=0)


def test_integrity_check_rejects_wrong_counts(raw, tmp_path):
    with pytest.raises(RuntimeError, match="integrity"):
        oam_tcd.prepare(raw, tmp_path / "cache", size=32, expected={"train_rows": 99})


def _rle_to_string(counts):
    """Reference encoder (port of pycocotools rleToString) for the decoder test."""
    out = []
    for i, x in enumerate(counts):
        if i > 2:
            x -= counts[i - 2]
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return "".join(out)


def test_rle_decode_compressed_matches_uncompressed():
    rng = np.random.default_rng(0)
    m = (rng.random((23, 17)) > 0.6).astype(np.uint8)
    m[5:15, 3:12] = 1
    rle = _coco_rle(m)
    assert np.array_equal(oam_tcd.rle_decode(rle), m)
    comp = {"size": rle["size"], "counts": _rle_to_string(rle["counts"])}
    assert np.array_equal(oam_tcd.rle_decode(comp), m)
