"""E060: FiLM between encoder stages on a dense task (E055's NYUv2 metric depth under zoom).

E058 found that, for an image-level head, FiLM after every encoder stage (`film_encoder`) beat
ManySeg's FiLM, which there only reaches the pooled feature. For pixel heads ManySeg's FiLM
already modulates every encoder level entering the decoder and the decoder output. This asks
whether FiLM between stages also helps there, before building it into ManySeg.

Arms (all re-run on this code; data, model and recipe as E055):
  none          no vector (baseline)
  film          ManySeg FiLM, on: features + output (diagram A)
  film_encoder  FiLM after each ResNet stage via forward hooks, no ManySeg conditioning (C)
  film_both     ManySeg FiLM + FiLM after each stage (A + C)

The rest of this docstring is E055's:

Each sample is a crop of the 480x640 frame covering 1/z of its side, resized (antialiased,
always downscaling) to 192x256. Zoom changes the effective focal length by z, so for the same
pixels the metric depth is ambiguous unless the model is told z.

Arms (vector input = log z):
  none           no vector input (baseline)
  film           FiLM conditioning (ManySeg default)
  channels       log z broadcast as an extra input channel
  film_shuffled  FiLM, but fed a log z drawn independently of the image zoom from the same
                 log-uniform distribution (train and eval): same input distribution, no
                 information. Should match `none`.
  nozoom         reference only: no vector, trained at z = 1 only (the usual task); evaluated at
                 every zoom, so its z1.0 entry is the reference and the others show the failure.
  probe          diagnostic: an image-scope head predicts log z from pixels alone. If it can,
                 the `none` baseline can infer zoom too (resampling cues), and a tie between
                 `none` and `film` would not mean FiLM is useless. Reports log z RMSE against the
                 prior std (predicting the mean).

Usage: python train.py --arm film --seed 0 --data ~/data_e055_nyu/npz --out runs/film_s0
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from manyseg import create_model

OUT_H, OUT_W = 192, 256
MARGIN = 8  # NYU frames have an invalid border from undistortion
Z_MIN, Z_MAX = 1.0, 2.4  # largest crop 464 px high -> smallest 193 px, so always downscaling
EVAL_Z = (1.0, 1.4, 1.8, 2.4)
HAS_META = ("film", "channels", "film_shuffled", "film_both")  # arms with a ManySeg vector input
HOOKED = ("film_encoder", "film_both")  # arms with FiLM after each encoder stage


class EncoderFiLM(torch.nn.Module):
    """FiLM after each ResNet stage: x * (1 + gamma(log z)) + beta(log z) per channel. Same MLP
    shape as ManySeg's metadata encoder (1 -> 64 -> GELU -> 64) plus a zero-initialised projection,
    so the model is unchanged at init. As E058's, with log z normalised like ManySeg's field."""

    STAGES = ("layer1", "layer2", "layer3", "layer4")
    CHANNELS = (64, 128, 256, 512)

    def __init__(self, model):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(1, 64), torch.nn.GELU(),
                                       torch.nn.Linear(64, 64),
                                       torch.nn.Linear(64, 2 * sum(self.CHANNELS)))
        torch.nn.init.zeros_(self.mlp[-1].weight)
        torch.nn.init.zeros_(self.mlp[-1].bias)
        self.params = None
        stages = {n.rsplit(".", 1)[-1]: m for n, m in model.named_modules()
                  if n.rsplit(".", 1)[-1] in self.STAGES and "backbone" in n}
        assert sorted(stages) == sorted(self.STAGES), sorted(stages)
        for i, name in enumerate(self.STAGES):
            stages[name].register_forward_hook(self._hook(i))

    def set(self, logz):
        self.params = self.mlp(((logz - LOGZ_MEAN) / LOGZ_STD).view(-1, 1).float())

    def _hook(self, i):
        def hook(_, __, out):
            o = 2 * sum(self.CHANNELS[:i])
            c = self.CHANNELS[i]
            g = self.params[:, o:o + c].view(-1, c, 1, 1).to(out.dtype)
            b = self.params[:, o + c:o + 2 * c].view(-1, c, 1, 1).to(out.dtype)
            return out * (1 + g) + b
        return hook
LOGZ_MEAN = 0.5 * (math.log(Z_MIN) + math.log(Z_MAX))
LOGZ_STD = (math.log(Z_MAX) - math.log(Z_MIN)) / math.sqrt(12)  # std of a uniform
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load(path: Path, device):
    d = np.load(path)
    rgb = torch.from_numpy(d["rgb"]).permute(0, 3, 1, 2).contiguous().to(device)  # uint8
    depth = torch.from_numpy(d["depth"].astype(np.float32) / 1000.0).unsqueeze(1).to(device)
    return rgb, depth


DEGRADE = False  # set by --degrade: random blur + noise independent of z


def degrade(x, gen):
    """Per-sample Gaussian blur (sigma U(0, 1.5) px) and noise (SD U(0, 0.03)), both drawn
    independently of z, to mask the resampling signature that differs between zoom levels."""
    out = []
    for i in range(x.shape[0]):
        xi = x[i:i + 1]
        sigma = float(torch.empty(1).uniform_(0.0, 1.5, generator=gen))
        if sigma > 0.05:
            r = int(math.ceil(3 * sigma))
            t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
            k = torch.exp(-0.5 * (t / sigma) ** 2)
            k = (k / k.sum()).view(1, 1, 1, -1).repeat(3, 1, 1, 1)
            xi = F.pad(xi, (r, r, r, r), mode="reflect")
            xi = F.conv2d(F.conv2d(xi, k, groups=3), k.transpose(-1, -2), groups=3)
        sd = float(torch.empty(1).uniform_(0.0, 0.03, generator=gen))
        noise = torch.randn(xi.shape, generator=gen).to(x.device) * sd
        out.append((xi + noise).clamp(0, 1))
    return torch.cat(out)


def crop_resize(rgb, depth, z, cy, cx, gen=None):
    """Crop a 3:4 window of side 1/z of the usable frame centred at (cy, cx) in [0, 1]
    of the free range, resize to OUT_H x OUT_W with antialiasing. Per-sample z, cy, cx."""
    H, W = rgb.shape[-2:]
    h0 = H - 2 * MARGIN
    w0 = min(W - 2 * MARGIN, round(h0 * OUT_W / OUT_H))
    xs, ys = [], []
    for i in range(rgb.shape[0]):
        h = round(h0 / float(z[i]))
        w = round(w0 / float(z[i]))
        y = MARGIN + round(float(cy[i]) * (H - 2 * MARGIN - h))
        x = MARGIN + round(float(cx[i]) * (W - 2 * MARGIN - w))
        r = rgb[i:i + 1, :, y:y + h, x:x + w].float() / 255.0
        d = depth[i:i + 1, :, y:y + h, x:x + w]
        xs.append(F.interpolate(r, (OUT_H, OUT_W), mode="bilinear", antialias=True,
                                align_corners=False))
        ys.append(F.interpolate(d, (OUT_H, OUT_W), mode="area"))
    x = torch.cat(xs)
    if DEGRADE:
        x = degrade(x, gen)
    return x, torch.cat(ys)


def build(arm):
    inputs = {"rgb": {"in_channels": 3}}
    if arm in HAS_META:
        inputs["meta"] = {"kind": "vector",
                          "fields": {"logz": {"type": "continuous", "mean": LOGZ_MEAN,
                                              "std": LOGZ_STD}},
                          "conditioning": "channels" if arm == "channels" else "film"}
    if arm == "probe":
        heads = {"logz": {"type": "regression", "out_channels": 1, "scope": "image"}}
    else:
        heads = {"depth": {"type": "regression", "out_channels": 1}}
    model = create_model(backbone="resnet34", pretrained=True, inputs=inputs, heads=heads)
    if arm in HOOKED:
        model.add_module("enc_film", EncoderFiLM(model))
    return model


def sample_z(n, gen):
    return torch.exp(torch.empty(n).uniform_(math.log(Z_MIN), math.log(Z_MAX), generator=gen))


def forward(model, arm, x, z, gen):
    feed = {"rgb": (x - MEAN.to(x.device)) / STD.to(x.device)}
    if arm in HOOKED:
        model.enc_film.set(torch.log(z))
    if arm in HAS_META:
        zc = sample_z(len(z), gen).to(z.device) if arm == "film_shuffled" else z
        feed["meta"] = model.pack_metadata("meta", {"logz": torch.log(zc)})
    if arm == "probe":
        return model(feed)["logz"].view(-1)  # predicted log z
    return model(feed)["depth"]  # predicted log depth (m)


@torch.no_grad()
def evaluate_probe(model, rgb, depth, seed=1234, bs=32):
    """log z RMSE on random zooms and crop positions, against the prior std."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    err = []
    for i in range(0, len(rgb), bs):
        n = min(bs, len(rgb) - i)
        z = sample_z(n, gen)
        x, _ = crop_resize(rgb[i:i + n], depth[i:i + n], z, torch.rand(n, generator=gen),
                           torch.rand(n, generator=gen), gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = forward(model, "probe", x, z.to(rgb.device), gen)
        err.append((p.float().cpu() - z.log()) ** 2)
    rmse = torch.cat(err).mean().sqrt().item()
    return {"logz_rmse": rmse, "prior_std": LOGZ_STD, "r2": 1 - rmse ** 2 / LOGZ_STD ** 2}


def metrics(pred_log, gt):
    """Per-image metrics averaged over images. pred_log: log metres; gt: metres."""
    out = {k: [] for k in ("absrel", "rmse", "delta1", "si_log", "scale_err")}
    for p, g in zip(pred_log, gt):
        p, g = p.flatten().float(), g.flatten().float()
        m = g > 0.05
        p, g = p[m], g[m]
        d = p.exp()
        out["absrel"].append(((d - g).abs() / g).mean().item())
        out["rmse"].append(((d - g) ** 2).mean().sqrt().item())
        out["delta1"].append((torch.maximum(d / g, g / d) < 1.25).float().mean().item())
        e = p - g.log()
        # scale-invariant log error (shape only) and |median log ratio| (scale only)
        out["si_log"].append((e.var(unbiased=False)).sqrt().item())
        out["scale_err"].append(e.median().abs().item())
    return {k: float(np.mean(v)) for k, v in out.items()}


@torch.no_grad()
def evaluate(model, arm, rgb, depth, zs, bs=32):
    model.eval()
    gen = torch.Generator().manual_seed(1234)
    res = {}
    for zv in zs:
        preds, gts = [], []
        for i in range(0, len(rgb), bs):
            n = min(bs, len(rgb) - i)
            z = torch.full((n,), zv, device=rgb.device)
            c = torch.full((n,), 0.5)
            x, y = crop_resize(rgb[i:i + n], depth[i:i + n], z, c, c, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = forward(model, arm, x, z, gen)
            preds.append(p.float())
            gts.append(y)
        res[f"z{zv}"] = metrics(torch.cat(preds), torch.cat(gts))
    res["mean"] = {k: float(np.mean([res[f"z{z}"][k] for z in zs])) for k in res[f"z{zs[0]}"]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["none", "film", "channels", "film_shuffled", "nozoom", "probe",
                             "film_encoder", "film_both"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--degrade", action="store_true")
    a = ap.parse_args()
    global DEGRADE
    DEGRADE = a.degrade

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(a.data)
    tr_rgb, tr_d = load(data / "train.npz", dev)
    va_rgb, va_d = load(data / "val.npz", dev)
    te_rgb, te_d = load(data / "test.npz", dev)

    model = build(a.arm).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps_per_epoch = len(tr_rgb) // a.bs
    total = a.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total,
                                                pct_start=0.05)
    gen = torch.Generator().manual_seed(a.seed)
    log = []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(tr_rgb), generator=gen)
        tot = 0.0
        for s in range(steps_per_epoch):
            idx = perm[s * a.bs:(s + 1) * a.bs].to(dev)
            n = len(idx)
            z = torch.ones(n) if a.arm == "nozoom" else sample_z(n, gen)
            cy, cx = torch.rand(n, generator=gen), torch.rand(n, generator=gen)
            x, y = crop_resize(tr_rgb[idx], tr_d[idx], z, cy, cx, gen)
            flip = torch.rand(n, generator=gen) < 0.5
            x[flip] = x[flip].flip(-1)
            y[flip] = y[flip].flip(-1)
            z = z.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = forward(model, a.arm, x, z, gen)
            if a.arm == "probe":
                loss = (p.float() - z.log()).abs().mean()
            else:
                loss = (p.float() - y.clamp_min(0.05).log()).abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        rec = {"epoch": ep, "loss": tot / steps_per_epoch, "time": time.time() - t0}
        if a.arm == "probe" and ((ep + 1) % 10 == 0 or ep == a.epochs - 1):
            rec["val"] = evaluate_probe(model, va_rgb, va_d)
        elif (ep + 1) % 10 == 0 or ep == a.epochs - 1:
            zs = (1.0,) if a.arm == "nozoom" else EVAL_Z
            rec["val"] = evaluate(model, a.arm, va_rgb, va_d, zs)["mean"]
        log.append(rec)
        print(json.dumps(rec), flush=True)

    ev = evaluate_probe if a.arm == "probe" else (
        lambda m, r, d: evaluate(m, a.arm, r, d, EVAL_Z))
    result = {"arm": a.arm, "seed": a.seed, "epochs": a.epochs, "degrade": a.degrade,
              "params": sum(p.numel() for p in model.parameters()),
              "train_min": (time.time() - t0) / 60,
              "val": ev(model, va_rgb, va_d), "test": ev(model, te_rgb, te_d)}
    (out / "result.json").write_text(json.dumps(result, indent=2))
    (out / "log.json").write_text(json.dumps(log))
    print("RESULT", json.dumps({"arm": a.arm, "seed": a.seed, "test": result["test"].get("mean", result["test"])}),
          flush=True)


if __name__ == "__main__":
    main()
