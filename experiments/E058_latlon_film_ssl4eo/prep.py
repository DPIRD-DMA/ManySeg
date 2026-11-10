"""Pack SSL4EO-S12 v1.1 S2RGB shards into one npz per split for E058.

Each sample: 4 seasons x RGB (B04 B03 B02, uint8) x 264 px, a cloud mask and the location's
centre lat/lon. The 256 px centre is 2x2-averaged to 128 px. Per-season cloud fraction is kept
so training can prefer clear seasons. Split 70 / 15 / 15 by 2 x 2 degree geographic block; sites
within 5.5 km of a neighbouring block in a different split are dropped (~5%), so no two sites
closer than that straddle splits.

Usage (zarr v2 needed):
  uv run --no-project --with "zarr<3" --with numpy python prep.py <shard dir> <out dir> [workers]
"""

import hashlib
import io
import sys
import tarfile
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import zarr


BLOCK = 2.0  # degrees
BUFFER_KM = 5.5  # chips are 2.6 km wide; a site this close to a differently split block is dropped


def split_of_block(bi: int, bj: int) -> str:
    bj %= int(360 / BLOCK)  # wrap longitude blocks at the antimeridian
    h = int(hashlib.md5(f"{bi}_{bj}".encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else "val" if h < 85 else "test"


def block_split(lat: float, lon: float) -> str | None:
    """The split of the site's 2-degree block, or None if the site lies within BUFFER_KM of a
    neighbouring block that belongs to a different split (so no two sites closer than the buffer
    straddle splits)."""
    bi, bj = int(np.floor(lat / BLOCK)), int(np.floor(lon / BLOCK))
    own = split_of_block(bi, bj)
    dlat = BUFFER_KM / 111.0
    dlon = BUFFER_KM / (111.0 * max(np.cos(np.radians(lat)), 0.05))
    near_i = [bi - 1] * bool(lat - bi * BLOCK < dlat) + [bi + 1] * bool((bi + 1) * BLOCK - lat < dlat)
    near_j = [bj - 1] * bool(lon - bj * BLOCK < dlon) + [bj + 1] * bool((bj + 1) * BLOCK - lon < dlon)
    for i in [bi] + near_i:
        for j in [bj] + near_j:
            if (i, j) != (bi, bj) and split_of_block(i, j) != own:
                return None
    return own


def read_shard(path: str):
    xs, clouds, lats, lons, ids = [], [], [], [], []
    with tarfile.open(path) as tar:
        for m in tar:
            if not m.name.endswith(".zarr.zip"):
                continue
            # zarr v2 takes any mapping as a store; ZipStore needs a path, so read the zip into one
            with zipfile.ZipFile(io.BytesIO(tar.extractfile(m).read())) as z:
                store = {n: z.read(n) for n in z.namelist() if not n.endswith("/")}
            g = zarr.open_group(store, mode="r")
            b = g["bands"][...]  # (4, 3, 264, 264) uint8
            if b.shape != (4, 3, 264, 264):
                continue
            c = b[:, :, 4:260, 4:260].astype(np.float32)
            xs.append(c.reshape(4, 3, 128, 2, 128, 2).mean((3, 5)).round().astype(np.uint8))
            clouds.append((g["cloud_mask"][...][:, 4:260, 4:260] > 0).mean((1, 2)))
            lats.append(float(g["center_lat"][...]))
            lons.append(float(g["center_lon"][...]))
            ids.append(str(g["sample"][...]))
    return xs, clouds, lats, lons, ids


def main(src: str, dst: str, workers: int = 6) -> None:
    out = Path(dst)
    out.mkdir(parents=True, exist_ok=True)
    shards = sorted(str(p) for p in Path(src).glob("*.tar"))
    acc = {s: ([], [], [], [], []) for s in ("train", "val", "test")}
    dropped = 0
    with Pool(workers) as pool:
        for k, (xs, cl, la, lo, ids) in enumerate(pool.imap_unordered(read_shard, shards)):
            for x, c, a, o, i in zip(xs, cl, la, lo, ids):
                split = block_split(a, o)
                if split is None:
                    dropped += 1
                    continue
                for lst, v in zip(acc[split], (x, c, a, o, i)):
                    lst.append(v)
            print(f"shard {k + 1}/{len(shards)}", flush=True)
    print("dropped near differently split blocks:", dropped, flush=True)
    for split, (xs, cl, la, lo, ids) in acc.items():
        lat = np.array(la, np.float32)
        lon = np.array(lo, np.float32)
        np.savez(out / f"{split}.npz", x=np.stack(xs), cloud=np.stack(cl).astype(np.float32),
                 lat=lat, lon=lon, ids=np.array(ids))
        blocks = {f"{int(np.floor(a / 2))}_{int(np.floor(o / 2))}" for a, o in zip(la, lo)}
        print(split, len(xs), "locations", len(blocks), "blocks",
              f"lat {lat.min():.0f}..{lat.max():.0f} lon {lon.min():.0f}..{lon.max():.0f}",
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6)
