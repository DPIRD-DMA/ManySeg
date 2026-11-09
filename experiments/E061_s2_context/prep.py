"""Build the E061 subset of FLAIR #2: E059's patches with a 40 x 40 px Sentinel-2 context crop.

Same domains, split and patch sampling as E059 (prep.py there; same rng and order). The S2 crop
is the 40 x 40 px (400 m) window centred on the patch centroid instead of E059's 10 x 10; smaller
contexts (20, 10) are its centre, sliced in train.py, so every arm sees the same patches and the
same dates. Dates are chosen as in E059: clear when max cloud and snow probability over the
central 10 x 10 (the patch footprint) is below 20%, and patches need >= 3 clear dates (whole-window
clearness dropped ~70% of patches). Each pixel of the window takes the median over those dates
on which it is itself clear (probability < 20%), or over all of them if it never is, so the
centre 10 x 10 equals E059's median exactly. Patches whose 40 x 40 window leaves the zone are
dropped. Only the annual median is kept (E059's mid_median arm).

Usage: python prep.py <flair dir> <sen dir> <centroids json> <out dir>
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

PER_DOMAIN = {"train": 1250, "val": 1000, "test": 1000}
HALF = 20  # S2 crop half-size (px): 40 x 40 = 400 m
CLEAR = 20


def domain_splits(domains):
    order = sorted(domains, key=lambda d: hashlib.md5(d.encode()).hexdigest())[:12]
    return {d: "train" if i < 8 else "val" if i < 10 else "test" for i, d in enumerate(order)}


def main(flair: str, sen: str, centroids: str, out: str) -> None:
    flair_p, sen_p, out_p = Path(flair), Path(sen), Path(out)
    out_p.mkdir(parents=True, exist_ok=True)
    cent = json.load(open(centroids))
    domains = sorted(p.name for p in (flair_p / "flair_aerial_train").iterdir())
    splits = domain_splits(domains)
    print({s: [d for d, v in splits.items() if v == s] for s in ("train", "val", "test")})
    rng = np.random.default_rng(0)
    data = {s: {k: [] for k in ("aerial", "label", "s2_med", "n_clear", "domain", "corr")}
            for s in ("train", "val", "test")}
    drops = {"edge": 0, "cloud": 0}
    n_pix_clear = []  # per patch: fraction of window pixels clear on at least one chosen date
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
            for p in paths:
                r, c = cent[p.name]
                if r - HALF < 0 or c - HALF < 0 or r + HALF > s2.shape[2] or c + HALF > s2.shape[3]:
                    drops["edge"] += 1
                    continue
                crop = np.asarray(s2[:, :, r - HALF:r + HALF, c - HALF:c + HALF])
                pm = np.asarray(masks[:, :, r - HALF:r + HALF, c - HALF:c + HALF]).max(1)  # (T,40,40)
                clear = pm[:, HALF - 5:HALF + 5, HALF - 5:HALF + 5].max((1, 2)) < CLEAR
                if clear.sum() < 3:
                    drops["cloud"] += 1
                    continue
                sub = crop[clear].astype(np.float32)  # (Tc, 10, 40, 40)
                ok = (pm[clear] < CLEAR)[:, None]  # (Tc, 1, 40, 40): pixel clear on that date
                ok = ok | ~ok.any(0, keepdims=True)  # never clear -> use every chosen date
                med = np.nanmedian(np.where(ok, sub, np.nan), 0).astype(np.uint16)  # truncates, as E059
                n_pix_clear.append(float((pm[clear] < CLEAR).any(0).mean()))
                with rasterio.open(p) as f:
                    a = f.read().astype(np.float32)
                msk = Path(str(p).replace("flair_aerial_train", "flair_labels_train")
                           .replace("/img/", "/msk/").replace("IMG_", "MSK_"))
                with rasterio.open(msk) as f:
                    lab = f.read(1)
                aerial = a.reshape(5, 256, 2, 256, 2).mean((2, 4)).round().astype(np.uint8)
                lab = np.minimum(lab[::2, ::2].astype(np.int16) - 1, 12).clip(0).astype(np.uint8)
                # alignment check: aerial NIR (block-averaged to 10 x 10) vs the centre of S2 B8
                a_nir = a[3, :500, :500].reshape(10, 50, 10, 50).mean((1, 3))
                centre = med[6, HALF - 5:HALF + 5, HALF - 5:HALF + 5].astype(np.float32)
                corr = np.corrcoef(a_nir.ravel(), centre.ravel())[0, 1]
                d = data[split]
                for k, v in (("aerial", aerial), ("label", lab), ("s2_med", med),
                             ("n_clear", int(clear.sum())), ("domain", dom), ("corr", corr)):
                    d[k].append(v)
        print(dom, split, sum(len(v) for v in by_zone.values()), "picked;",
              len(data[split]["aerial"]), "kept so far", drops, flush=True)
    for split, d in data.items():
        arrays = {k: np.stack(v) if k in ("aerial", "label", "s2_med") else np.array(v)
                  for k, v in d.items()}
        np.savez(out_p / f"{split}.npz", **arrays)
        print(split, arrays["aerial"].shape, "clear dates median", np.median(arrays["n_clear"]),
              "aerial-S2 NIR corr median", np.nanmedian(arrays["corr"]).round(3), flush=True)
    print("window pixels clear on >= 1 chosen date: mean", np.mean(n_pix_clear).round(3))


if __name__ == "__main__":
    main(*sys.argv[1:5])
