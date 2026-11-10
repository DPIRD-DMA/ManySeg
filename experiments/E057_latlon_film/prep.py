"""Build the E057 chip set from the existing global Sentinel-2 L1C chips on the 3090.

Source: ~/data_s2_l1c_chips (60k chips, 4 bands, 256 px, from 931 scenes / 909 MGRS tiles,
56S-72N). Keeps chips with cloud fraction <= 0.2 (a cloudy chip carries no location signal),
locates each chip at the centre of its MGRS 100 km square (good to ~70 km, far below the 10 deg
longitude bins), splits train / val / test 70 / 15 / 15 **by tile** so a scene never spans splits,
and 2x2-averages to 128 px.

Run with mgrs available, e.g.
  uv run --no-project --with mgrs --with numpy python prep.py ~/data_s2_l1c_chips ~/data_e057/npz
"""

import hashlib
import json
import sys
from pathlib import Path

import mgrs
import numpy as np

MAX_CLOUD = 0.2


def tile_of(scene: str) -> str:
    return scene.split("_")[5][1:]  # e.g. T25LCF -> 25LCF


def split_of(tile: str) -> str:
    h = int(hashlib.md5(tile.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else "val" if h < 85 else "test"


def main(src: str, dst: str) -> None:
    src_p, out = Path(src), Path(dst)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(src_p / "meta.json"))
    stats = np.load(src_p / "clear_stats.npz")
    cloud = stats["cloud"]
    assert len(meta) == len(cloud)
    assert all(m["scene"] == s for m, s in zip(meta, stats["scene"]))
    chips = np.load(src_p / "chips.npy", mmap_mode="r")
    conv = mgrs.MGRS()
    centre = {}
    keep = np.flatnonzero(cloud <= MAX_CLOUD)
    rows = {"train": [], "val": [], "test": []}
    for i in keep:
        t = tile_of(meta[i]["scene"])
        if t not in centre:
            centre[t] = conv.toLatLon(t + "5000050000")  # centre of the 100 km square
        rows[split_of(t)].append(i)
    for split, idx in rows.items():
        idx = np.array(idx)
        x = np.empty((len(idx), 4, 128, 128), np.uint16)
        for j0 in range(0, len(idx), 512):
            sl = idx[j0:j0 + 512]
            a = np.asarray(chips[sl], dtype=np.float32)
            x[j0:j0 + len(sl)] = a.reshape(len(sl), 4, 128, 2, 128, 2).mean((3, 5)).round()
        tiles = [tile_of(meta[i]["scene"]) for i in idx]
        lat = np.array([centre[t][0] for t in tiles], np.float32)
        lon = np.array([centre[t][1] for t in tiles], np.float32)
        gsd = np.array([meta[i]["gsd"] for i in idx], np.float32)
        month = np.array([int(meta[i]["datetime"][5:7]) for i in idx], np.int8)
        np.savez(out / f"{split}.npz", x=x, lat=lat, lon=lon, gsd=gsd, month=month,
                 tile=np.array(tiles), src_index=idx)
        print(split, x.shape, "tiles", len(set(tiles)), f"lat {lat.min():.0f}..{lat.max():.0f}",
              f"lon {lon.min():.0f}..{lon.max():.0f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
