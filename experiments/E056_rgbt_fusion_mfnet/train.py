"""E056: separate encoders vs stacking for RGB + thermal segmentation (MFNet, day and night).

E005 found no gain from separate encoders on Sen1Floods11, where S2 alone carried the task.
MFNet is the opposite case: at night RGB is close to useless and thermal carries the signal,
and thermal statistics are far from the ImageNet RGB the stem was trained on.

Arms (all ResNet-34 U-Net, ImageNet weights):
  rgb         RGB only
  thermal     thermal only
  early       RGB + thermal stacked as 4 channels into one encoder (baseline)
  mid_concat  one encoder per input, intermediate concat fusion
  mid_gated   one encoder per input, intermediate gated fusion

Usage: python train.py --arm early --seed 0 --data ~/data_e056_mfnet/npz --out runs/early_s0
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from manyseg import create_model

NUM_CLASSES = 9  # 0 = unlabelled, then car, person, bike, curve, car stop, guardrail, cone, bump
CROP_H, CROP_W = 384, 512
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load(path: Path, device):
    d = np.load(path)
    rgb = torch.from_numpy(d["rgb"]).permute(0, 3, 1, 2).contiguous().to(device)  # uint8
    th = torch.from_numpy(d["thermal"]).unsqueeze(1).contiguous().to(device)  # uint8
    lab = torch.from_numpy(d["label"]).long().to(device)
    night = torch.from_numpy(d["night"]).to(device)
    return rgb, th, lab, night


def build(arm):
    if arm == "rgb":
        inputs = {"rgb": {"in_channels": 3}}
    elif arm == "thermal":
        inputs = {"thermal": {"in_channels": 1}}
    elif arm == "early":
        inputs = {"rgbt": {"in_channels": 4}}
    else:
        inputs = {"rgb": {"in_channels": 3}, "thermal": {"in_channels": 1}}
    kw = {}
    if arm.startswith("mid_"):
        kw["fusion"] = {"type": "intermediate", "op": arm.split("_")[1]}
    return create_model(backbone="resnet34", pretrained=True, inputs=inputs,
                        heads={"seg": {"type": "segmentation", "out_channels": NUM_CLASSES}},
                        **kw)


def feed(arm, rgb_u8, th_u8, th_stats):
    rgb = (rgb_u8.float() / 255.0 - MEAN.to(rgb_u8.device)) / STD.to(rgb_u8.device)
    th = (th_u8.float() / 255.0 - th_stats[0]) / th_stats[1]
    if arm == "rgb":
        return {"rgb": rgb}
    if arm == "thermal":
        return {"thermal": th}
    if arm == "early":
        return {"rgbt": torch.cat([rgb, th], 1)}
    return {"rgb": rgb, "thermal": th}


def augment(rgb, th, lab, gen):
    """Random scale in [0.75, 1.5], random crop CROP_H x CROP_W, random horizontal flip.
    Per-sample, on GPU."""
    n, _, H, W = rgb.shape
    xs, ts, ys = [], [], []
    for i in range(n):
        s = float(torch.empty(1).uniform_(0.75, 1.5, generator=gen))
        h, w = max(CROP_H, round(H * s)), max(CROP_W, round(W * s))
        r = F.interpolate(rgb[i:i + 1].float(), (h, w), mode="bilinear", align_corners=False)
        t = F.interpolate(th[i:i + 1].float(), (h, w), mode="bilinear", align_corners=False)
        y = F.interpolate(lab[i:i + 1, None].float(), (h, w), mode="nearest-exact")[:, 0].long()
        top = int(torch.randint(0, h - CROP_H + 1, (1,), generator=gen))
        left = int(torch.randint(0, w - CROP_W + 1, (1,), generator=gen))
        sl = (slice(top, top + CROP_H), slice(left, left + CROP_W))
        r, t, y = r[..., sl[0], sl[1]], t[..., sl[0], sl[1]], y[..., sl[0], sl[1]]
        if torch.rand(1, generator=gen) < 0.5:
            r, t, y = r.flip(-1), t.flip(-1), y.flip(-1)
        xs.append(r)
        ts.append(t)
        ys.append(y)
    return (torch.cat(xs).round().clamp(0, 255).to(torch.uint8),
            torch.cat(ts).round().clamp(0, 255).to(torch.uint8), torch.cat(ys))


def confusion(pred, lab):
    k = lab * NUM_CLASSES + pred
    return torch.bincount(k.flatten(), minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES)


def iou_from(cm):
    cm = cm.double()
    inter = cm.diag()
    union = cm.sum(0) + cm.sum(1) - inter
    iou = inter / union.clamp_min(1)
    valid = union > 0  # a class predicted but absent from the labels counts as IoU 0
    valid8 = valid.clone()
    valid8[0] = False
    return {"miou9": iou[valid].mean().item(), "miou8": iou[valid8].mean().item(),
            "per_class": [v if ok else None for v, ok in zip(iou.tolist(), valid.tolist())]}


@torch.no_grad()
def evaluate(model, arm, rgb, th, lab, night, th_stats, bs=16):
    model.eval()
    cms = {"all": 0, "day": 0, "night": 0}
    for i in range(0, len(rgb), bs):
        sl = slice(i, i + bs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(feed(arm, rgb[sl], th[sl], th_stats))["seg"]
        pred = logits.argmax(1)
        for j in range(pred.shape[0]):
            cm = confusion(pred[j], lab[sl][j])
            cms["all"] = cms["all"] + cm
            key = "night" if bool(night[sl][j]) else "day"
            cms[key] = cms[key] + cm
    return {k: iou_from(v) for k, v in cms.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["rgb", "thermal", "early", "mid_concat", "mid_gated"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(a.data)
    tr = load(data / "train.npz", dev)
    va = load(data / "val.npz", dev)
    te = load(data / "test.npz", dev)
    # thermal mean / SD over the train set, in chunks (a full float copy can OOM a shared GPU)
    s = s2 = 0.0
    for i in range(0, len(tr[1]), 64):
        c = tr[1][i:i + 64].double() / 255.0
        s += c.sum().item()
        s2 += (c * c).sum().item()
    cnt = tr[1].numel()
    th_stats = (s / cnt, (s2 / cnt - (s / cnt) ** 2) ** 0.5)

    model = build(a.arm).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps_per_epoch = len(tr[0]) // a.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                                total_steps=a.epochs * steps_per_epoch,
                                                pct_start=0.05)
    gen = torch.Generator().manual_seed(a.seed)
    log = []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(tr[0]), generator=gen)
        tot = 0.0
        for s in range(steps_per_epoch):
            idx = perm[s * a.bs:(s + 1) * a.bs].to(dev)
            r, t, y = augment(tr[0][idx], tr[1][idx], tr[2][idx], gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(feed(a.arm, r, t, th_stats))["seg"]
            loss = F.cross_entropy(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        rec = {"epoch": ep, "loss": tot / steps_per_epoch, "time": time.time() - t0}
        if (ep + 1) % 20 == 0 or ep == a.epochs - 1:
            v = evaluate(model, a.arm, *va, th_stats)
            rec["val"] = {k: v[k]["miou9"] for k in v}
        log.append(rec)
        print(json.dumps(rec), flush=True)

    result = {"arm": a.arm, "seed": a.seed, "epochs": a.epochs,
              "params": sum(p.numel() for p in model.parameters()),
              "train_min": (time.time() - t0) / 60,
              "val": evaluate(model, a.arm, *va, th_stats),
              "test": evaluate(model, a.arm, *te, th_stats)}
    (out / "result.json").write_text(json.dumps(result, indent=2))
    (out / "log.json").write_text(json.dumps(log))
    print("RESULT", json.dumps({"arm": a.arm, "seed": a.seed,
                                "test": {k: result["test"][k]["miou9"] for k in result["test"]}}),
          flush=True)


if __name__ == "__main__":
    main()
