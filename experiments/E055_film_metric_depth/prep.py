"""Convert the jagennath-hari/nyuv2 parquet files into one .npz per split.

rgb: (N, 480, 640, 3) uint8; depth: (N, 480, 640) uint16 in millimetres (hole-filled).
Usage: python prep.py <hf_snapshot_dir> <out_dir>
"""

import io
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def main(src: str, dst: str) -> None:
    out = Path(dst)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        rows = pq.read_table(Path(src) / "data" / f"{split}-00000-of-00001.parquet",
                             columns=["id", "rgb", "depth"]).to_pylist()
        rgb = np.stack([np.array(Image.open(io.BytesIO(r["rgb"]["bytes"])).convert("RGB"))
                        for r in rows])
        depth = np.stack([np.array(Image.open(io.BytesIO(r["depth"]["bytes"])))
                          for r in rows]).astype(np.uint16)
        ids = np.array([r["id"] for r in rows])
        np.savez(out / f"{split}.npz", rgb=rgb, depth=depth, ids=ids)
        print(split, rgb.shape, depth.shape, f"depth m: {depth.min() / 1e3:.2f}-{depth.max() / 1e3:.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
