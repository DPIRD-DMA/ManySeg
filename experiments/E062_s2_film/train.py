"""E062: does Sentinel-2 help as FiLM metadata instead of an image? (FLAIR #2, after E059/E061)

ResNet-34 U-Net on the aerial input (5 bands, 256 px at 0.4 m) in every arm. Arms:
  aerial     aerial only (baseline)
  s2enc      + separate ResNet-18 S2 encoder on the 10 x 10 px annual median (E059 mid_median)
  film_med   + FiLM from S2 annual-median statistics over the patch footprint (10 x 10 px):
             per-band mean and std, 20 values
  film_ts    + FiLM from monthly per-band means over the footprint, 12 months x 10 bands = 120
             values; months without a clear date are NaN (ManySeg's learned "unknown")
  film_ctx   + FiLM from the film_med statistics plus the same statistics over the 40 x 40 px
             (400 m) window around the patch, 40 values
  *_shuf     the same arm with each patch's vector taken from another patch of the same split
             (fixed random permutation): a control for gains that don't need the true S2 values

FiLM is ManySeg's default for a pixel head (vector input, conditioning "film": the encoder
levels entering the decoder and the decoder output). Vector features are z-scored with
training-set statistics. The statistics are invariant to the flips / 90-degree rotations used
for augmentation, so the vector needs no augmentation.

Data: E061's npz (aerial, labels, 40 x 40 median) and E059's npz (monthly medians), which hold
the same patches in the same order (checked: labels, domains and the centre of the 40 x 40
median equal E059's).

Usage: python train.py --arm film_ts --seed 0 --out runs/film_ts_s0
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from manyseg import create_model

N_CLASSES, SCORED = 13, 12
UP = 64  # s2enc: S2 10 x 10 -> 64 px, as E059 / E061
BASE = ["aerial", "s2enc", "film_med", "film_ts", "film_ctx"]
ARMS = BASE + [a + "_shuf" for a in BASE if a.startswith("film_")]
D59, D61 = Path("~/data_e059/npz").expanduser(), Path("~/data_e061/npz").expanduser()


def stats(x):
    """(B, 10, H, W) float -> (B, 20): per-band mean and std."""
    return torch.cat([x.mean((2, 3)), x.std((2, 3))], 1)


def load(split: str):
    a, b = np.load(D61 / f"{split}.npz"), np.load(D59 / f"{split}.npz")
    # the two files must hold the same patches in the same order
    assert np.array_equal(a["label"], b["label"]) and (a["domain"] == b["domain"]).all()
    assert np.array_equal(a["s2_med"][..., 15:25, 15:25], b["s2_med"])
    med40 = torch.from_numpy(b_to_float(a["s2_med"]))  # (N, 10, 40, 40)
    med10 = med40[..., 15:25, 15:25]
    monthly = torch.from_numpy(b_to_float(b["s2"])).mean((3, 4))  # (N, 12, 10)
    monthly[~torch.from_numpy(b["s2_valid"])] = float("nan")
    return {"aerial": torch.from_numpy(a["aerial"]), "label": torch.from_numpy(a["label"]),
            "s2_med": torch.from_numpy(a["s2_med"][..., 15:25, 15:25]),
            "film_med": stats(med10),
            "film_ts": monthly.flatten(1),
            "film_ctx": torch.cat([stats(med10), stats(med40)], 1)}


def b_to_float(x):
    return x.astype(np.float32)


def build(arm, n_feat):
    inputs = {"aerial": {"in_channels": 5}}
    kw = {}
    if arm == "s2enc":
        inputs["s2"] = {"in_channels": 10, "backbone": "resnet18", "alignment": "same_extent"}
        kw["fusion"] = {"type": "intermediate", "op": "concat"}
    elif arm.startswith("film_"):
        inputs["meta"] = {"kind": "vector", "num_features": n_feat, "conditioning": "film"}
    return create_model(backbone="resnet34", pretrained=True, inputs=inputs,
                        heads={"seg": {"type": "segmentation", "out_channels": N_CLASSES}}, **kw)


def batch(d, idx, arm, st, gen=None):
    aer = (d["aerial"][idx].cuda().float() - st["a_mean"]) / st["a_std"]
    lab = d["label"][idx].cuda().long()
    med = None
    if arm == "s2enc":
        med = (d["s2_med"][idx].cuda().float() - st["s_mean"]) / st["s_std"]
    if gen is not None:
        k = int(torch.randint(0, 4, (1,), generator=gen))
        flip = bool(torch.rand(1, generator=gen) < 0.5)

        def tf(x):
            x = torch.rot90(x, k, (-2, -1))
            return x.flip(-1) if flip else x
        aer, lab = tf(aer), tf(lab)
        if med is not None:
            med = tf(med)
    feed = {"aerial": aer}
    if med is not None:
        feed["s2"] = F.interpolate(med, (UP, UP), mode="bilinear", align_corners=False)
    if arm.startswith("film_"):
        key = arm.removesuffix("_shuf")
        v = d[key][d["perm"][idx] if arm.endswith("_shuf") else idx].cuda()
        feed["meta"] = (v - st["v_mean"]) / st["v_std"]  # NaN (missing month) stays NaN
    return feed, lab


@torch.no_grad()
def evaluate(model, arm, d, st, bs=32):
    model.eval()
    cm = torch.zeros(N_CLASSES, N_CLASSES, dtype=torch.long, device="cuda")
    for i in range(0, len(d["aerial"]), bs):
        idx = torch.arange(i, min(i + bs, len(d["aerial"])))
        feed, lab = batch(d, idx, arm, st)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(feed)["seg"].argmax(1)
        cm += torch.bincount((lab * N_CLASSES + pred).flatten(),
                             minlength=N_CLASSES ** 2).view(N_CLASSES, N_CLASSES)
    cm = cm.double()
    inter = cm.diag()
    union = cm.sum(0) + cm.sum(1) - inter
    iou = inter[:SCORED] / union[:SCORED].clamp_min(1)
    valid = union[:SCORED] > 0
    return {"miou": iou[valid].mean().item(),
            "per_class": [v if ok else None for v, ok in zip(iou.tolist(), valid.tolist())]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tr, va, te = (load(s) for s in ("train", "val", "test"))
    for i, d in enumerate((tr, va, te)):  # fixed per split, the same for every seed
        d["perm"] = torch.randperm(len(d["aerial"]), generator=torch.Generator().manual_seed(100 + i))
    sa = tr["aerial"][::10].float()
    ss = tr["s2_med"][::10].float()
    st = {"a_mean": sa.mean((0, 2, 3)).view(1, 5, 1, 1).cuda(),
          "a_std": sa.std((0, 2, 3)).view(1, 5, 1, 1).cuda(),
          "s_mean": ss.mean((0, 2, 3)).view(1, 10, 1, 1).cuda(),
          "s_std": ss.std((0, 2, 3)).view(1, 10, 1, 1).cuda()}
    n_feat = None
    if a.arm.startswith("film_"):
        v = tr[a.arm.removesuffix("_shuf")]
        st["v_mean"] = torch.nanmean(v, 0).cuda()
        st["v_std"] = torch.from_numpy(np.nanstd(v.numpy(), 0)).clamp_min(1e-6).cuda()
        n_feat = v.shape[1]

    model = build(a.arm, n_feat).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    spe = len(tr["aerial"]) // a.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * spe,
                                                pct_start=0.05)
    gen = torch.Generator().manual_seed(a.seed)
    log, t0 = [], time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(tr["aerial"]), generator=gen)
        tot = 0.0
        for s in range(spe):
            feed, lab = batch(tr, perm[s * a.bs:(s + 1) * a.bs], a.arm, st, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(feed)["seg"]
            loss = F.cross_entropy(logits.float(), lab)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        rec = {"epoch": ep, "loss": tot / spe, "time": time.time() - t0}
        if (ep + 1) % 5 == 0 or ep == a.epochs - 1:
            rec["val"] = evaluate(model, a.arm, va, st)["miou"]
        log.append(rec)
        print(json.dumps(rec), flush=True)

    result = {"arm": a.arm, "seed": a.seed, "epochs": a.epochs, "n_feat": n_feat,
              "params": sum(p.numel() for p in model.parameters()),
              "train_min": (time.time() - t0) / 60,
              "val": evaluate(model, a.arm, va, st), "test": evaluate(model, a.arm, te, st)}
    (out / "result.json").write_text(json.dumps(result, indent=2))
    (out / "log.json").write_text(json.dumps(log))
    print("RESULT", json.dumps({"arm": a.arm, "seed": a.seed, "val": result["val"]["miou"],
                                "test": result["test"]["miou"]}), flush=True)


if __name__ == "__main__":
    main()
