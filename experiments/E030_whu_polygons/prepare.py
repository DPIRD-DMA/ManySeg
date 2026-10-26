"""Cache WHU aerial (0.3 m, 512² tiles) for E030.

The tiles are not georeferenced, so ground truth comes from the raster labels (README amendment
1): buildings are the label's **False** pixels (checked visually: empty tiles are all True),
instances are 4-connected components, and GT polygons are their exact pixel-edge outlines
simplified by 1 px (``polygons_from_labels(simplify=1.0, min_area=16)``).

    python prepare.py "<raw>/whu_building/tiles" "<cache>/whu_aerial"
"""

import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

from common import save_polygons
from manyseg.vectors import polygons_from_labels

GT_SIMPLIFY = 1.0


def main(src: str, out: str) -> None:
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        if (out / f"{split}_poly.npz").exists():
            continue
        names = sorted((src / split / "image").glob("*.tif"),
                       key=lambda p: int(p.stem.rsplit("_", 1)[-1]))  # "123" or "val_123"
        n = len(names)
        img = np.lib.format.open_memmap(out / f"{split}_img.npy", "w+", np.uint8, (n, 512, 512, 3))
        inst = np.lib.format.open_memmap(out / f"{split}_inst.npy", "w+", np.int32, (n, 512, 512))
        polys = []
        for i, p in enumerate(names):
            img[i] = tifffile.imread(p)
            building = ~tifffile.imread(src / split / "label" / p.name).astype(bool)
            lab, _ = ndimage.label(building)
            inst[i] = lab
            polys.append(polygons_from_labels(lab, simplify=GT_SIMPLIFY, min_area=16))
        img.flush()
        inst.flush()
        save_polygons(out / f"{split}_poly.npz", polys)
        print(split, n, "tiles,", sum(len(t) for t in polys), "polygons", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
