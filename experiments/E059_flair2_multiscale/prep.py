"""Build the E059 subset of FLAIR #2: aerial 0.2 m patches + Sentinel-2 10 m time series.

Aerial (FLAIR #1 train, already on disk): 5 bands (R G B NIR elevation), 512 px at 0.2 m,
2x2-averaged to 256 px (0.4 m). Labels 1-19 -> 0-12 (FLAIR #2 convention: 13-19 merge into
"other" = 12, which is trained on but left out of mIoU); label pixel = the top-left of each 2x2
block (a 0.2 m offset, far below a label's accuracy).

Sentinel-2 (flair_sen_train.zip): per zone, (T, 10, H, W) uint16 L2A + (T, 2, H, W) cloud / snow
probability. Each patch is located in its zone's S2 grid by flair-2_centroids_sp_to_patch.json
(first value = row, second = column, as in the FLAIR #2 loader); the patch footprint is the 10x10
pixels around it (100 m, the patch is 102.4 m). From the clear dates (max cloud and snow
probability in the crop below 20): a monthly median for each of the 12 months (missing months
masked) and an annual median.

Split by departement (domain), as FLAIR does: 8 train, 2 val, 2 test domains, chosen by a fixed
hash; up to PER_DOMAIN patches sampled per domain.

Usage: python prep.py <flair dir> <sen dir> <centroids json> <out dir>
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

PER_DOMAIN = {"train": 1250, "val": 1000, "test": 1000}
HALF = 5  # S2 crop half-size (px)
CLEAR = 20  # max cloud / snow probability (%) for a date to count as clear in the crop


def domain_splits(domains):
    order = sorted(domains, key=lambda d: hashlib.md5(d.encode()).hexdigest())[:12]
    return {d: "train" if i < 8 else "val" if i < 10 else "test" for i, d in enumerate(order)}


def month_of(product: str) -> int:
    return int(product.split("_")[2][4:6])  # S2A_MSIL2A_20200104T... -> 1


def main(flair: str, sen: str, centroids: str, out: str) -> None:
    flair_p, sen_p, out_p = Path(flair), Path(sen), Path(out)
    out_p.mkdir(parents=True, exist_ok=True)
    cent = json.load(open(centroids))
    domains = sorted(p.name for p in (flair_p / "flair_aerial_train").iterdir())
    splits = domain_splits(domains)
    print({s: [d for d, v in splits.items() if v == s] for s in ("train", "val", "test")})
    rng = np.random.default_rng(0)
    data = {s: {k: [] for k in ("aerial", "label", "s2", "s2_valid", "s2_med", "domain",
                                 "corr")} for s in ("train", "val", "test")}
    for dom, split in splits.items():
        imgs = sorted((flair_p / "flair_aerial_train" / dom).glob("*/img/*.tif"))
        imgs = [p for p in imgs if p.name in cent]
        pick = sorted(rng.choice(len(imgs), min(PER_DOMAIN[split], len(imgs)), replace=False))
        by_zone = {}
        for i in pick:
            by_zone.setdefault(imgs[i].parent.parent.name, []).append(imgs[i])
        for zone, paths in by_zone.items():
            sdir = sen_p / dom / zone / "sen"
            stem = f"SEN2_sp_{dom}-{zone}"
            s2 = np.load(sdir / f"{stem}_data.npy", mmap_mode="r")
            masks = np.load(sdir / f"{stem}_masks.npy", mmap_mode="r")
            months = np.array([month_of(p) for p in
                               (sdir / f"{stem}_products.txt").read_text().split()])
            for p in paths:
                r, c = cent[p.name]
                if r - HALF < 0 or c - HALF < 0 or r + HALF > s2.shape[2] or c + HALF > s2.shape[3]:
                    continue
                crop = np.asarray(s2[:, :, r - HALF:r + HALF, c - HALF:c + HALF])
                cm = np.asarray(masks[:, :, r - HALF:r + HALF, c - HALF:c + HALF]).max((1, 2, 3))
                clear = cm < CLEAR
                if clear.sum() < 3:
                    continue
                monthly = np.zeros((12, 10, 2 * HALF, 2 * HALF), np.uint16)
                valid = np.zeros(12, bool)
                for m in range(1, 13):
                    sel = clear & (months == m)
                    if sel.any():
                        monthly[m - 1] = np.median(crop[sel], 0)
                        valid[m - 1] = True
                med = np.median(crop[clear], 0).astype(np.uint16)
                with rasterio.open(p) as f:
                    a = f.read().astype(np.float32)
                msk = Path(str(p).replace("flair_aerial_train", "flair_labels_train")
                           .replace("/img/", "/msk/").replace("IMG_", "MSK_"))
                with rasterio.open(msk) as f:
                    lab = f.read(1)
                aerial = a.reshape(5, 256, 2, 256, 2).mean((2, 4)).round().astype(np.uint8)
                lab = np.minimum(lab[::2, ::2].astype(np.int16) - 1, 12).clip(0).astype(np.uint8)
                # alignment check: NIR of the aerial patch (block-averaged to 10x10) vs S2 B8
                a_nir = a[3, :500, :500].reshape(10, 50, 10, 50).mean((1, 3))
                corr = np.corrcoef(a_nir.ravel(), med[6].astype(np.float32).ravel())[0, 1]
                d = data[split]
                for k, v in (("aerial", aerial), ("label", lab), ("s2", monthly),
                             ("s2_valid", valid), ("s2_med", med), ("domain", dom),
                             ("corr", corr)):
                    d[k].append(v)
        print(dom, split, sum(len(v) for v in by_zone.values()), "picked;",
              len(data[split]["aerial"]), "kept so far", flush=True)
    for split, d in data.items():
        arrays = {k: np.stack(v) if k not in ("domain", "corr") else np.array(v)
                  for k, v in d.items()}
        np.savez(out_p / f"{split}.npz", **arrays)
        print(split, arrays["aerial"].shape, "months valid", arrays["s2_valid"].mean().round(2),
              "aerial-S2 NIR corr median", np.nanmedian(arrays["corr"]).round(3), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:5])
