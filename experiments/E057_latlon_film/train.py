"""E057: predict longitude from a Sentinel-2 chip, with latitude given as a vector input.

Latitude changes the answer for the same pixels (it narrows where on Earth a scene can be), so
any way of injecting it should help. The question is which injection works: FiLM on the pooled
feature (multiplicative), `head` (an additive logit term), or `channels` (latitude as an input
band). Longitude is predicted as one of 36 bins of 10 deg (cross-entropy; wraps naturally).

Arms:
  none           image only (baseline)
  film           FiLM conditioning on latitude
  head           latitude added to the image head (Linear([pool(f), e]))
  channels       latitude broadcast as an extra input channel
  film_shuffled  FiLM fed a latitude drawn at random from the training latitudes (control)
  film_encoder   FiLM after every encoder stage (written here with forward hooks; ManySeg's
                 FiLM only reaches the pooled feature for image heads). Tests whether reach,
                 not mechanism, explains channels > film.
  probe          diagnostic: predict the latitude bin (18 x 10 deg) from pixels alone; how much
                 latitude the image already carries

Augmentation: random flips and 90-degree rotations. These hide the shadow direction, which
otherwise gives away the hemisphere; documented in the README.

Usage: python train.py --arm film --seed 0 --data ~/data_e057/npz --out runs/film_s0
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from manyseg import create_model

N_LON, N_LAT = 36, 18
HAS_META = ("film", "head", "channels", "film_shuffled")


def lon_bin(lon):
    return torch.floor(torch.remainder(lon + 180.0, 360.0) / 10.0).long().remainder(N_LON)


def lat_bin(lat):
    return ((lat + 90.0) // 10.0).long().clamp(0, N_LAT - 1)


def load(path: Path):
    d = np.load(path)
    return (torch.from_numpy(d["x"]), torch.from_numpy(d["lat"]), torch.from_numpy(d["lon"]))


class EncoderFiLM(torch.nn.Module):
    """FiLM inside the encoder: after each ResNet stage, x * (1 + gamma(lat)) + beta(lat) per
    channel. Same form as ManySeg's FiLM (MLP on the normalised field, zero-initialised output, so
    identity at init), but reaching every stage instead of only the pooled feature."""

    STAGES = ("layer1", "layer2", "layer3", "layer4")
    CHANNELS = (64, 128, 256, 512)

    def __init__(self, encoder):
        super().__init__()
        # same shape as ManySeg's metadata encoder (1 -> 64 -> GELU -> 64) + a zero-init projection
        self.mlp = torch.nn.Sequential(torch.nn.Linear(1, 64), torch.nn.GELU(),
                                       torch.nn.Linear(64, 64),
                                       torch.nn.Linear(64, 2 * sum(self.CHANNELS)))
        torch.nn.init.zeros_(self.mlp[-1].weight)
        torch.nn.init.zeros_(self.mlp[-1].bias)
        self.params = None
        for i, name in enumerate(self.STAGES):
            getattr(encoder, name).register_forward_hook(self._hook(i))

    def set(self, lat):
        self.params = self.mlp((lat / 35.0).view(-1, 1).float())

    def _hook(self, i):
        def hook(_, __, out):
            o = 2 * sum(self.CHANNELS[:i])
            c = self.CHANNELS[i]
            g = self.params[:, o:o + c].view(-1, c, 1, 1).to(out.dtype)
            b = self.params[:, o + c:o + 2 * c].view(-1, c, 1, 1).to(out.dtype)
            return out * (1 + g) + b
        return hook


def build(arm):
    if arm == "film_encoder":
        model = build("none")
        model.add_module("enc_film", EncoderFiLM(model.backbone.encoders["s2"].model))
        return model
    inputs = {"s2": {"in_channels": 4}}
    if arm in HAS_META:
        mode = {"film_shuffled": "film"}.get(arm, arm)
        inputs["meta"] = {"kind": "vector",
                          "fields": {"lat": {"type": "continuous", "mean": 0.0, "std": 35.0}},
                          "conditioning": mode}
    n = N_LAT if arm == "probe" else N_LON
    return create_model(backbone="resnet34", pretrained=True, inputs=inputs,
                        heads={"out": {"type": "classification", "out_channels": n,
                                       "scope": "image"}})


def forward(model, arm, x, lat, lat_pool, gen):
    feed = {"s2": x}
    if arm == "film_encoder":
        model.enc_film.set(lat.to(x.device))
    if arm in HAS_META:
        if arm == "film_shuffled":
            lat = lat_pool[torch.randint(0, len(lat_pool), (len(lat),), generator=gen)]
        feed["meta"] = model.pack_metadata("meta", {"lat": lat.to(x.device)})
    return model(feed)["out"]


def augment(x, gen):
    k = int(torch.randint(0, 4, (1,), generator=gen))
    x = torch.rot90(x, k, (-2, -1))
    if torch.rand(1, generator=gen) < 0.5:
        x = x.flip(-1)
    return x


def lon_err_deg(pred_bin, lon):
    centre = pred_bin.float() * 10.0 - 175.0
    d = (centre - lon).abs() % 360.0
    return torch.minimum(d, 360.0 - d)


@torch.no_grad()
def evaluate(model, arm, data, norm, lat_pool, bs=256):
    model.eval()
    x_all, lat_all, lon_all = data
    gen = torch.Generator().manual_seed(1234)
    logits = []
    for i in range(0, len(x_all), bs):
        x = ((x_all[i:i + bs].cuda().float() / 10000.0) - norm[0]) / norm[1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits.append(forward(model, arm, x, lat_all[i:i + bs], lat_pool, gen).float().cpu())
    lg = torch.cat(logits)
    if arm == "probe":
        y = lat_bin(lat_all)
        pred = lg.argmax(1)
        lat_err = (pred.float() * 10.0 - 85.0 - lat_all).abs()
        return {"acc": (pred == y).float().mean().item(), "lat_mae": lat_err.mean().item(),
                "lat_median": lat_err.median().item()}
    y = lon_bin(lon_all)
    top3 = lg.topk(3, 1).indices
    err = lon_err_deg(lg.argmax(1), lon_all)
    return {"acc": (lg.argmax(1) == y).float().mean().item(),
            "top3": (top3 == y[:, None]).any(1).float().mean().item(),
            "nll": F.cross_entropy(lg, y).item(),
            "lon_median": err.median().item(), "lon_mae": err.mean().item()}


def lat_only_prior(tr, te):
    """Accuracy of predicting the most common training longitude bin for the test chip's
    latitude bin: what latitude alone gives, with no image."""
    tl, tg = lat_bin(tr[1]), lon_bin(tr[2])
    table = torch.zeros(N_LAT, N_LON)
    table.index_put_((tl, tg), torch.ones(len(tl)), accumulate=True)
    pred = table.argmax(1)[lat_bin(te[1])]
    return {"acc": (pred == lon_bin(te[2])).float().mean().item(),
            "lon_median": lon_err_deg(pred, te[2]).median().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=list(HAS_META) + ["none", "probe", "film_encoder"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(a.data)
    tr, va, te = (load(data / f"{s}.npz") for s in ("train", "val", "test"))
    sample = tr[0][::20].float() / 10000.0
    norm = (sample.mean((0, 2, 3)).view(1, 4, 1, 1).cuda(),
            sample.std((0, 2, 3)).view(1, 4, 1, 1).cuda())
    lat_pool = tr[1].clone()
    x_gpu = tr[0].to(device="cuda", dtype=torch.int32)  # CUDA cannot index uint16; ~4 GB

    model = build(a.arm).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    spe = len(tr[0]) // a.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * spe,
                                                pct_start=0.05)
    gen = torch.Generator().manual_seed(a.seed)
    y_all = lat_bin(tr[1]) if a.arm == "probe" else lon_bin(tr[2])
    log, t0 = [], time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(tr[0]), generator=gen)
        tot = 0.0
        for s in range(spe):
            idx = perm[s * a.bs:(s + 1) * a.bs]
            x = ((x_gpu[idx.cuda()].float() / 10000.0) - norm[0]) / norm[1]
            x = augment(x, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = forward(model, a.arm, x, tr[1][idx], lat_pool, gen)
            loss = F.cross_entropy(lg.float(), y_all[idx].cuda())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        rec = {"epoch": ep, "loss": tot / spe, "time": time.time() - t0}
        if (ep + 1) % 5 == 0 or ep == a.epochs - 1:
            rec["val"] = evaluate(model, a.arm, va, norm, lat_pool)
        log.append(rec)
        print(json.dumps(rec), flush=True)

    result = {"arm": a.arm, "seed": a.seed, "epochs": a.epochs,
              "params": sum(p.numel() for p in model.parameters()),
              "train_min": (time.time() - t0) / 60,
              "val": evaluate(model, a.arm, va, norm, lat_pool),
              "test": evaluate(model, a.arm, te, norm, lat_pool),
              "lat_only_prior_test": lat_only_prior(tr, te)}
    (out / "result.json").write_text(json.dumps(result, indent=2))
    (out / "log.json").write_text(json.dumps(log))
    print("RESULT", json.dumps({"arm": a.arm, "seed": a.seed, "test": result["test"],
                                "lat_prior": result["lat_only_prior_test"]}), flush=True)


if __name__ == "__main__":
    main()
