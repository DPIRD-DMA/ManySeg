"""E044: train one model with the unified road heads on City-Scale (README).

    uv run python train44.py --cache <e044 cache> --seed 0 [--epochs 40] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "E030_whu_polygons"))
from common import DeviceLoader, seed_everything  # noqa: E402
from manyseg import create_model  # noqa: E402
from manyseg_bench import losses  # noqa: E402
from manyseg_bench.providers.base import split_params  # noqa: E402
from manyseg_bench.training.recipe import get_recipe  # noqa: E402
from manyseg_bench.training.trainer import train  # noqa: E402

HEADS = {"mask": {"type": "segmentation", "out_channels": 1},
         "line": {"type": "segmentation", "out_channels": 1},
         "kp": {"type": "segmentation", "out_channels": 3},  # junction, end, crossing
         "direction": {"type": "regression", "out_channels": 4}}  # two (cos 2t, sin 2t) modes


class Tiles(torch.utils.data.Dataset):
    """1024 px tiles; ``augment``: random h/v flips and transpose, with the double-angle direction
    transformed analytically (flip: sin 2t -> -sin 2t; transpose: cos 2t -> -cos 2t)."""

    def __init__(self, cache: Path, split: str, augment: bool):
        self.img = np.load(cache / f"{split}_img.npy", mmap_mode="r")
        self.prob = np.load(cache / f"{split}_prob.npy", mmap_mode="r")
        self.dir = np.load(cache / f"{split}_dir.npy", mmap_mode="r")
        self.augment = augment

    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        img = np.asarray(self.img[i]).transpose(2, 0, 1)  # (3, H, W)
        prob = np.asarray(self.prob[i]).astype(np.float32) / 255
        d = np.asarray(self.dir[i]).astype(np.float32) / 127
        if self.augment:
            # per-sample generator from the worker's torch seed (derived from the DataLoader's seeded
            # generator) and the sample index: reproducible for a given --seed (Codex E044 #1)
            rng = np.random.default_rng([torch.initial_seed() % 2 ** 32, i])
            if rng.random() < 0.5:  # horizontal flip
                img, prob, d = img[..., ::-1], prob[..., ::-1], d[..., ::-1]
                d = d * np.array([1, -1, 1, -1], np.float32)[:, None, None]
            if rng.random() < 0.5:  # vertical flip
                img, prob, d = img[:, ::-1], prob[:, ::-1], d[:, ::-1]
                d = d * np.array([1, -1, 1, -1], np.float32)[:, None, None]
            if rng.random() < 0.5:  # transpose
                img, prob, d = img.swapaxes(1, 2), prob.swapaxes(1, 2), d.swapaxes(1, 2)
                d = d * np.array([-1, 1, -1, 1], np.float32)[:, None, None]
        x = torch.from_numpy(np.ascontiguousarray(img))
        y = {"mask": torch.from_numpy(np.ascontiguousarray(prob[0])), "line": torch.from_numpy(np.ascontiguousarray(prob[1])),
             "kp": torch.from_numpy(np.ascontiguousarray(prob[2:5])), "direction": torch.from_numpy(np.ascontiguousarray(d))}
        return x, y


def _bce_dice(logit, target):
    p = torch.sigmoid(logit)
    inter = (p * target).sum((-2, -1))
    dice = 1 - (2 * inter + 1) / (p.sum((-2, -1)) + target.sum((-2, -1)) + 1)
    return F.binary_cross_entropy_with_logits(logit, target) + dice.mean()


class RoadTask:
    """``w``: loss weights (mask, line, kp, direction)."""
    name = "unified_roads"

    def __init__(self, w=(1.0, 1.0, 1.0, 1.0)):
        self.w = w
    primary_metric = "mask_iou"
    higher_is_better = True

    def heads(self):
        return HEADS

    def targets(self, y):
        return y

    def loss(self, out, t):
        lm = _bce_dice(out["mask"][:, 0].float(), t["mask"].float())
        ll = _bce_dice(out["line"][:, 0].float(), t["line"].float())
        kp = t["kp"].float()
        pos = (kp > 0.5) & (kp == F.max_pool2d(kp, 3, 1, 1))
        # one focal loss over all three channels, normalised by their total positives (floor 16): a
        # channel with no positives in the batch (crossings are rare) was divided by 1, its summed
        # negatives swamped the decoder and the mask collapsed (debug runs, README)
        pl = out["kp"].float().sigmoid().clamp(1e-6, 1 - 1e-6)
        pos_l = -((1 - pl) ** 2) * torch.log(pl) * pos
        neg_l = -((1 - kp) ** 4) * (pl ** 2) * torch.log(1 - pl) * (~pos)
        lk = (pos_l.sum() + neg_l.sum()) / pos.sum().clamp(min=16)
        d, pd = t["direction"].float(), out["direction"].float()
        m1 = (t["mask"] > 0.5)[:, None].expand(-1, 2, -1, -1)
        m2 = (d[:, 2:].abs().sum(1, keepdim=True) > 0).expand(-1, 2, -1, -1)
        ld = F.l1_loss(pd[:, :2][m1], d[:, :2][m1]) if m1.any() else pd.sum() * 0
        if m2.any():
            ld = ld + F.l1_loss(pd[:, 2:][m2], d[:, 2:][m2])
        w = self.w
        return w[0] * lm + w[1] * ll + w[2] * lk + w[3] * ld

    def evaluator(self):
        task = self

        class _Eval:
            def __init__(self):
                self.loss = self.n = self.inter = self.union = 0

            def update(self, out, y):
                b = y["mask"].shape[0]
                self.loss += float(task.loss(out, y)) * b
                self.n += b
                p, g = out["mask"][:, 0] > 0, y["mask"] > 0.5
                self.inter += int((p & g).sum())
                self.union += int((p | g).sum())

            def compute(self):
                return {"loss": self.loss / max(1, self.n), "mask_iou": self.inter / max(1, self.union)}

        return _Eval()


def build():
    model = create_model(backbone="resnet34.tv_in1k", in_channels=3, pretrained=True, heads=HEADS)
    torch.nn.init.constant_(model.heads.kp.conv.bias, -2.19)  # CenterNet prior (E036 lesson)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--w", default="1,1,1,1", help="loss weights mask,line,kp,direction")
    args = ap.parse_args()
    out = HERE / "runs"
    out.mkdir(exist_ok=True)
    name = f"U_s{args.seed}{args.tag}"
    seed_everything(args.seed)
    device = torch.device("cuda")

    def loader(split, train_mode):
        ds = Tiles(args.cache, split, augment=train_mode)
        if args.limit:
            ds = torch.utils.data.Subset(ds, range(min(args.limit, len(ds))))
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=train_mode, drop_last=train_mode,
                                         num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
                                         generator=torch.Generator().manual_seed(args.seed))
        return DeviceLoader(dl, device)

    model = build()
    enc, dec = split_params(model, model.backbone)
    recipe = get_recipe("fastai_like", epochs=args.epochs, batch_size=args.batch_size, eval_every=min(5, args.epochs))
    t0 = time.time()
    res = train(model, enc, dec, loader("train", True), loader("val", False), num_classes=1, recipe=recipe,
                device=device, task=RoadTask(tuple(float(x) for x in args.w.split(","))))
    torch.save(model.state_dict(), out / f"{name}.pt")
    res.update(seed=args.seed, recipe=recipe.to_dict(), wall_s=time.time() - t0, limit=args.limit)
    (out / f"{name}_train.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"done {name}: {res['final_val']} in {res['wall_s'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
