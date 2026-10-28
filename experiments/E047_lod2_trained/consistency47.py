"""E047: per-tile consistency between 3DBAG faces and the AHN nDSM (README amendment 2).

face_support: fraction of face pixels with nDSM > 1.5 m (low = faces on flat ground: built after
the AHN survey). unlabelled: area of smooth (5x5 nDSM std < 0.3 m) elevated (> 2.5 m) components
>= 25 m^2 at least 1 m from any face, relative to face area (high = buildings missing from 3DBAG,
e.g. demolished after the survey)."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from rasterio.features import rasterize
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "E031_lod2_oracle"))
from prepare47 import CACHE, SPLITS  # noqa: E402


def stats(nd: np.ndarray, faces) -> tuple[float, float]:
    m = (rasterize([(f, 1) for f in faces], out_shape=nd.shape).astype(bool) if faces
         else np.zeros(nd.shape, bool))
    fs = float((nd[m] > 1.5).mean()) if m.any() else 0.0
    mu = ndimage.uniform_filter(nd, 5)
    sd = np.sqrt(np.maximum(ndimage.uniform_filter(nd * nd, 5) - mu * mu, 0))
    cand = (nd > 2.5) & (sd < 0.3) & ~ndimage.binary_dilation(m, iterations=4)
    lab, n = ndimage.label(cand)
    if n:
        size = np.bincount(lab.ravel())
        size[0] = 0
        un = float(size[size >= 400].sum())
    else:
        un = 0.0
    return fs, un / max(float(m.sum()), 1.0)


def main():
    out = {}
    for split in SPLITS:
        tiles = pickle.loads((CACHE / f"{split}_tiles.pkl").read_bytes())
        nds = np.load(CACHE / f"{split}_ndsm.npy", mmap_mode="r")
        out[split] = [stats(np.asarray(nds[i], np.float32), t.faces) for i, t in enumerate(tiles)]
        fs = np.array(out[split])
        print(split, len(tiles), "face_support p10/p50", np.percentile(fs[:, 0], [10, 50]).round(3),
              "unlabelled p50/p90", np.percentile(fs[:, 1], [50, 90]).round(3), flush=True)
    (CACHE / "consistency.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
