"""E059: separate encoders for inputs at different scales (FLAIR #2: aerial 0.4 m + S2 10 m).

Arms (ResNet-34 U-Net on the aerial input; ResNet-18 for the S2 encoder where there is one):
  aerial      aerial only (5 bands) (baseline)
  early       aerial + the S2 annual median upsampled to 256 px, stacked as 15 channels
  mid_median  aerial encoder + S2-median encoder (same_extent), intermediate concat fusion
  mid_ts      aerial encoder + S2 monthly time-series encoder (ManySeg timeseries, attention over
              months, masked missing months, day-of-year dates; same_extent), concat fusion

The S2 crop (10 x 10 px) is upsampled to 64 px before its encoder, so a stride-32 ResNet still
has a 2 x 2 deepest map. mIoU over the 12 FLAIR #2 classes ("other" = 12 is trained on, not
scored).

Usage: python train.py --arm mid_ts --seed 0 --data ~/data_e059/npz --out runs/mid_ts_s0
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
S2_UP = 64
MONTH_DOY = torch.tensor([15 + 30.4 * m for m in range(12)])


def load(path: Path):
    d = np.load(path)
    return {k: torch.from_numpy(d[k]) for k in ("aerial", "label", "s2", "s2_valid", "s2_med")}


def build(arm):
    inputs = {"aerial": {"in_channels": 15 if arm == "early" else 5}}
    kw = {}
    if arm == "mid_median":
        inputs["s2"] = {"in_channels": 10, "backbone": "resnet18", "alignment": "same_extent"}
    elif arm == "mid_ts":
        inputs["s2"] = {"kind": "timeseries", "in_channels": 10, "backbone": "resnet18",
                        "temporal": "attention", "alignment": "same_extent"}
    if arm.startswith("mid_"):
        kw["fusion"] = {"type": "intermediate", "op": "concat"}
    return create_model(backbone="resnet34", pretrained=True, inputs=inputs,
                        heads={"seg": {"type": "segmentation", "out_channels": N_CLASSES}}, **kw)


def batch(d, idx, arm, st, gen=None):
    """Normalised model inputs and labels for samples idx, with one shared random flip / rot90
    (applied to every raster of the sample, so the inputs stay aligned) when gen is given."""
    aer = (d["aerial"][idx].cuda().float() - st["a_mean"]) / st["a_std"]
    lab = d["label"][idx].cuda().long()
    med = (d["s2_med"][idx].cuda().float() - st["s_mean"]) / st["s_std"]  # (B, 10, 10, 10)
    ts = (d["s2"][idx].cuda().float() - st["s_mean"][:, None]) / st["s_std"][:, None]
    valid = d["s2_valid"][idx].cuda()
    ts = ts * valid[:, :, None, None, None]  # masked months are ignored by the model anyway
    if gen is not None:
        k = int(torch.randint(0, 4, (1,), generator=gen))
        flip = bool(torch.rand(1, generator=gen) < 0.5)

        def tf(x):
            x = torch.rot90(x, k, (-2, -1))
            return x.flip(-1) if flip else x
        aer, lab, med, ts = tf(aer), tf(lab), tf(med), tf(ts)
    feed = {"aerial": aer}
    if arm == "early":
        up = F.interpolate(med, aer.shape[-2:], mode="bilinear", align_corners=False)
        feed["aerial"] = torch.cat([aer, up], 1)
    elif arm == "mid_median":
        feed["s2"] = F.interpolate(med, (S2_UP, S2_UP), mode="bilinear", align_corners=False)
    elif arm == "mid_ts":
        b, t = ts.shape[:2]
        up = F.interpolate(ts.flatten(0, 1), (S2_UP, S2_UP), mode="bilinear",
                           align_corners=False)
        feed["s2"] = up.view(b, t, 10, S2_UP, S2_UP)
        feed["s2_mask"] = valid
        feed["s2_dates"] = MONTH_DOY.cuda().expand(b, t)
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
    # IoU from the full matrix (a scored-class pixel predicted "other" is still a miss), then
    # average the 12 scored classes, as FLAIR #2 does
    cm = cm.double()
    inter = cm.diag()
    union = cm.sum(0) + cm.sum(1) - inter
    iou = inter[:SCORED] / union[:SCORED].clamp_min(1)
    valid = union[:SCORED] > 0
    return {"miou": iou[valid].mean().item(),
            "per_class": [v if ok else None for v, ok in zip(iou.tolist(), valid.tolist())]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["aerial", "early", "mid_median", "mid_ts"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(a.data)
    tr, va, te = (load(data / f"{s}.npz") for s in ("train", "val", "test"))
    sa = tr["aerial"][::10].float()
    ss = tr["s2_med"][::10].float()
    st = {"a_mean": sa.mean((0, 2, 3)).view(1, 5, 1, 1).cuda(),
          "a_std": sa.std((0, 2, 3)).view(1, 5, 1, 1).cuda(),
          "s_mean": ss.mean((0, 2, 3)).view(1, 10, 1, 1).cuda(),
          "s_std": ss.std((0, 2, 3)).view(1, 10, 1, 1).cuda()}

    model = build(a.arm).cuda()
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

    result = {"arm": a.arm, "seed": a.seed, "epochs": a.epochs,
              "params": sum(p.numel() for p in model.parameters()),
              "train_min": (time.time() - t0) / 60,
              "val": evaluate(model, a.arm, va, st), "test": evaluate(model, a.arm, te, st)}
    (out / "result.json").write_text(json.dumps(result, indent=2))
    (out / "log.json").write_text(json.dumps(log))
    print("RESULT", json.dumps({"arm": a.arm, "seed": a.seed, "val": result["val"]["miou"],
                                "test": result["test"]["miou"]}), flush=True)


if __name__ == "__main__":
    main()
