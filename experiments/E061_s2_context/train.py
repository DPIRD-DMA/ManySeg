"""E061: does a wider Sentinel-2 context help a separate S2 encoder? (FLAIR #2, E059 follow-up)

Arms (ResNet-34 U-Net on the aerial input; ResNet-18 S2 encoder, intermediate concat fusion):
  aerial    aerial only (baseline)
  ctx10     S2 annual median, 10 x 10 px (100 m, the patch footprint) -> E059's mid_median
  ctx20     20 x 20 px (200 m) around the patch
  ctx40     40 x 40 px (400 m) around the patch

All S2 crops are upsampled by 6.4 (10 px -> 64, 20 -> 128, 40 -> 256), so the ground per
input pixel is the same in every arm. ManySeg's same_extent alignment assumes both inputs cover
the same ground, so for ctx20 / ctx40 a forward hook on the S2 encoder keeps the centre 1/k of
every feature map (k = ctx / 10: the patch footprint, since the crop is centred on the patch),
and ManySeg is told the S2 input's valid size is 64 px. Its same_extent resize then maps the
cropped 64/s features onto the aerial grid with the same single bilinear resize as ctx10. The
encoder sees the whole context, so the kept features carry information from around the patch.

Usage: python train.py --arm ctx40 --seed 0 --data ~/data_e061/npz --out runs/ctx40_s0
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
FULL = 40  # stored crop size (px)
UP = 6.4  # S2 upsampling factor before its encoder
CTX = {"aerial": None, "ctx10": 10, "ctx20": 20, "ctx40": 40}


def load(path: Path):
    d = np.load(path)
    return {k: torch.from_numpy(d[k]) for k in ("aerial", "label", "s2_med")}


def centre_hook(k: int):
    """Forward hook: keep the centre 1/k of each S2 feature map (the patch footprint)."""
    def hook(module, args, out):
        res = []
        for f in out:
            h, w = f.shape[-2:]
            assert h % k == 0 and w % k == 0, (h, w, k)
            ch, cw = h // k, w // k
            t, l = (h - ch) // 2, (w - cw) // 2
            res.append(f[..., t:t + ch, l:l + cw])
        return res
    return hook


def footprint_valid(enc, size: int):
    """Make ManySeg treat the S2 input as size x size px (the footprint the hook keeps), so its
    same_extent resize maps the cropped features onto the aerial grid exactly as for ctx10: one
    bilinear resize from 64/s to 256/s at every stride."""
    orig = enc._encode

    def _encode(inputs):
        feats, valid = orig(inputs)
        valid = dict(valid)
        valid["s2"] = (size, size)
        return feats, valid
    enc._encode = _encode


def build(arm):
    inputs = {"aerial": {"in_channels": 5}}
    kw = {}
    if arm != "aerial":
        inputs["s2"] = {"in_channels": 10, "backbone": "resnet18", "alignment": "same_extent"}
        kw["fusion"] = {"type": "intermediate", "op": "concat"}
    model = create_model(backbone="resnet34", pretrained=True, inputs=inputs,
                         heads={"seg": {"type": "segmentation", "out_channels": N_CLASSES}}, **kw)
    if arm != "aerial" and CTX[arm] > 10:
        model.backbone.encoders["s2"].register_forward_hook(centre_hook(CTX[arm] // 10))
        footprint_valid(model.backbone, round(10 * UP))
    return model


def batch(d, idx, arm, st, gen=None):
    """Normalised inputs and labels for samples idx, with one shared random flip / rot90 when
    gen is given (both crops are centred on the patch, so rotating each about its own centre
    keeps them aligned)."""
    aer = (d["aerial"][idx].cuda().float() - st["a_mean"]) / st["a_std"]
    lab = d["label"][idx].cuda().long()
    feed_s2 = None
    if CTX[arm]:
        c = CTX[arm]
        o = (FULL - c) // 2
        med = d["s2_med"][idx][..., o:o + c, o:o + c].cuda().float()
        feed_s2 = (med - st["s_mean"]) / st["s_std"]
    if gen is not None:
        k = int(torch.randint(0, 4, (1,), generator=gen))
        flip = bool(torch.rand(1, generator=gen) < 0.5)

        def tf(x):
            x = torch.rot90(x, k, (-2, -1))
            return x.flip(-1) if flip else x
        aer, lab = tf(aer), tf(lab)
        if feed_s2 is not None:
            feed_s2 = tf(feed_s2)
    feed = {"aerial": aer}
    if feed_s2 is not None:
        n = round(CTX[arm] * UP)
        feed["s2"] = F.interpolate(feed_s2, (n, n), mode="bilinear", align_corners=False)
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
    ap.add_argument("--arm", required=True, choices=list(CTX))
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
    ss = tr["s2_med"][::10].float()  # stats over the full 40 x 40 window, shared by all arms
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
