"""OAM-TCD tree detection (Veitch-Michaelis et al. 2024), CC BY 4.0 subset (HF ``restor/tcd``).

Point task: individual trees (COCO category 2) become instance ids whose centroids are the
target points; tree *groups* (category 1, "canopy") become ignore regions (label ``-1``),
because their trees are not individually annotated. The category mapping is verified from the
data at preparation (trees are much smaller than canopy groups) rather than assumed.

Images are 2048x2048 at 10 cm; they are stored at ``size`` (default 1024, i.e. 20 cm) to keep
the cache at ~22 GB. Splits: official ``test``; ``val`` = training fold 0 of the official 5-fold
assignment (tiles from one OAM source image share a fold, limiting leakage); ``train`` = folds
1-4. Only CC BY 4.0 rows are used (``restor/tcd`` is CC BY; NC/SA subsets are not downloaded).
"""

from __future__ import annotations

import io
import json
import statistics
import subprocess
from pathlib import Path

import numpy as np

from .base import DatasetInfo

HF = "https://huggingface.co/datasets/restor/tcd/resolve/main/"
FILES = ["data/test-00000-of-00001.parquet"] + [
    f"data/train-0000{i}-of-00007.parquet" for i in range(7)]
EXPECTED = {"train_rows": 4169, "test_rows": 439}
TREE, CANOPY = 2, 1

INFO = DatasetInfo(
    name="oam_tcd",
    num_classes=1,
    in_channels=3,
    class_names=("tree",),
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    license="CC BY 4.0 (imagery: OpenAerialMap contributors; annotations: Restor)",
    source="https://huggingface.co/datasets/restor/tcd",
    splits=("train", "val", "test"),
    test_splits=("test",),
    notes=("labels: tree instance ids > 0, canopy groups = -1 (ignored)",
           "val = official training fold 0", "stored at 1024 px (20 cm)"),
)
TASK = "points"
TASK_KWARGS = {"sigma": 2.0, "radius": 5.0}
LOADER_KWARGS = {"device_resident": False, "out_size": (512, 512), "scale": (0.2, 0.3),
                 "ratio": (0.9, 1.1), "vflip": True}
EVAL_BATCH_SIZE = 2


def download(raw: Path) -> None:
    for f in FILES:
        out = raw / f
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            subprocess.run(["curl", "-sSfL", "-o", str(out), HF + f], check=True)


def rle_decode(rle: dict) -> np.ndarray:
    """COCO RLE (compressed string or uncompressed list) -> (H, W) uint8 mask."""
    h, w = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, str):  # pycocotools' compressed LEB128-like encoding
        cnts, p, s = [], 0, counts.encode()
        while p < len(s):
            x, k, more = 0, 0, True
            while more:
                c = s[p] - 48
                x |= (c & 0x1F) << (5 * k)
                more = bool(c & 0x20)
                p += 1
                k += 1
                if not more and (c & 0x10):
                    x |= -1 << (5 * k)
            if len(cnts) > 2:
                x += cnts[-2]
            cnts.append(x)
        counts = cnts
    flat = np.zeros(h * w, np.uint8)
    pos, val = 0, 0
    for c in counts:
        flat[pos:pos + c] = val
        pos += c
        val ^= 1
    return flat.reshape((w, h)).T  # column-major


def _rasterise(anns: list, h: int, w: int, size: int) -> np.ndarray:
    from PIL import Image, ImageDraw

    scale = size / max(h, w)
    canopy = Image.new("L", (size, size), 0)
    trees = Image.new("I", (size, size), 0)
    dc, dt = ImageDraw.Draw(canopy), ImageDraw.Draw(trees)
    k = 0
    for a in sorted(anns, key=lambda a: int(a["category_id"])):  # canopy (1) before trees (2)
        cat = int(a["category_id"])
        seg = a["segmentation"]
        seg = json.loads(seg) if isinstance(seg, str) else seg
        if cat == TREE:
            k += 1
        if isinstance(seg, dict):
            m = Image.fromarray(rle_decode(seg) * 255).resize((size, size), Image.NEAREST)
            (dc if cat == CANOPY else dt).bitmap((0, 0), m, fill=1 if cat == CANOPY else k)
        else:
            for poly in seg:
                pts = [(x * scale, y * scale) for x, y in zip(poly[0::2], poly[1::2])]
                if len(pts) >= 3:
                    (dc if cat == CANOPY else dt).polygon(pts, fill=1 if cat == CANOPY else k)
    t = np.asarray(trees, dtype=np.int32)
    out = np.where(np.asarray(canopy) > 0, -1, 0).astype(np.int32)
    out[t > 0] = t[t > 0]
    if out.max() > np.iinfo(np.int16).max:
        raise RuntimeError("integrity: more trees than int16 ids")
    return out.astype(np.int16)


def _decode(row: dict, size: int):
    from PIL import Image

    img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
    h, w = row["height"], row["width"]
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR, reducing_gap=None)
    anns = json.loads(row["coco_annotations"] or "[]")
    return np.asarray(img), _rasterise(anns, h, w, size), anns


def prepare(raw: Path, cache: Path, size: int = 1024, workers: int = 1,
            expected: dict | None = EXPECTED) -> dict:
    import pyarrow.parquet as pq

    raw, cache = Path(raw), Path(cache)
    files = sorted((raw / "data").glob("*.parquet"))
    if not files:
        download(raw)
        files = sorted((raw / "data").glob("*.parquet"))
    rows = {"train": [], "test": []}
    for f in files:
        key = "test" if f.name.startswith("test") else "train"
        meta = pq.read_table(f, columns=["image_id", "validation_fold", "license"]).to_pylist()
        rows[key] += [(f, i, m) for i, m in enumerate(meta)]
    if expected is not None:
        for k in ("train", "test"):
            if len(rows[k]) != expected[f"{k}_rows"]:
                raise RuntimeError(f"integrity: {k} has {len(rows[k])} rows, "
                                   f"expected {expected[f'{k}_rows']}")
    bad = [m["license"] for _, _, m in rows["train"] + rows["test"] if m["license"] != "CC-BY 4.0"]
    if bad:
        raise RuntimeError(f"integrity: non CC-BY rows present: {set(bad)}")
    splits = {"train": [r for r in rows["train"] if r[2]["validation_fold"] != 0],
              "val": [r for r in rows["train"] if r[2]["validation_fold"] == 0],
              "test": rows["test"]}
    cache.mkdir(parents=True, exist_ok=True)
    summary, areas = {}, {TREE: [], CANOPY: []}
    for split, items in splits.items():
        n = len(items)
        imgs = np.lib.format.open_memmap(cache / f"{split}_images.npy", "w+", np.uint8,
                                         (n, size, size, 3))
        masks = np.lib.format.open_memmap(cache / f"{split}_masks.npy", "w+", np.int16,
                                          (n, size, size))
        names, n_trees = [], 0
        by_file: dict = {}
        for j, (f, i, m) in enumerate(items):
            by_file.setdefault(f, []).append((j, i, m))
        for f, lst in by_file.items():
            pf = pq.ParquetFile(f)
            wanted = {i: j for j, i, _ in lst}
            offset = 0
            for rg in range(pf.num_row_groups):
                tbl = pf.read_row_group(rg, columns=["image_id", "image", "height", "width",
                                                     "coco_annotations"])
                for r_idx, row in enumerate(tbl.to_pylist()):
                    gi = offset + r_idx
                    if gi not in wanted:
                        continue
                    j = wanted[gi]
                    imgs[j], masks[j], anns = _decode(row, size)
                    names_j = str(row["image_id"])
                    names.append((j, names_j))
                    for a in anns:
                        areas.setdefault(int(a["category_id"]), []).append(float(a["area"]))
                    n_trees += int(masks[j].max())
                offset += tbl.num_rows
        imgs.flush()
        masks.flush()
        (cache / f"{split}_names.json").write_text(
            json.dumps([nm for _, nm in sorted(names)]))
        summary[split] = {"n": n, "trees": n_trees}
    if areas.get(TREE) and areas.get(CANOPY) and \
            statistics.median(areas[TREE]) >= statistics.median(areas[CANOPY]):
        raise RuntimeError("integrity: category 2 (tree) should be smaller than category 1")
    (cache / "meta.json").write_text(json.dumps({"summary": summary}, indent=1))
    return summary
