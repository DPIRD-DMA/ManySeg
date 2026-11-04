"""E046: train B0 or B3 on FTW Netherlands (README).

    uv run python train46.py --arm {B0,B3} --seed 0 [--epochs 40] [--limit N]
"""

from __future__ import annotations

import os
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "E030_whu_polygons"))
from common import CornerLineTask, dihedral_array, dihedral_polygon, load_polygons, seed_everything  # noqa: E402
from manyseg import create_model  # noqa: E402
from manyseg.instances import instance_heads  # noqa: E402
from manyseg.vectors import corner_targets  # noqa: E402
from manyseg_bench.providers.base import split_params  # noqa: E402
from manyseg_bench.tasks import InstanceTask  # noqa: E402
from manyseg_bench.training.recipe import get_recipe  # noqa: E402
from manyseg_bench.training.trainer import train  # noqa: E402

CACHE = Path(os.environ.get("MANYSEG_BENCH_DATA", "data"), "cache/e046_ftw_nl")
TILE = 256


class FTWChips(torch.utils.data.Dataset):
    """(image uint16 (8, H, W), target) with a random dihedral transform when ``augment``."""

    def __init__(self, split: str, arm: str, augment: bool):
        self.img = np.load(CACHE / f"{split}_img.npy", mmap_mode="r")
        self.inst = np.load(CACHE / f"{split}_inst.npy", mmap_mode="r")
        self.polys = load_polygons(CACHE / f"{split}_poly.npz") if arm == "B3" else None
        self.arm, self.augment = arm, augment

    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        d = int(torch.randint(8, (1,))) if self.augment else 0
        img = dihedral_array(np.asarray(self.img[i]), d)
        inst = dihedral_array(np.asarray(self.inst[i]), d)
        x = torch.from_numpy(img.astype(np.int32)).permute(2, 0, 1)
        if self.arm == "B0":
            return x, torch.from_numpy(inst.astype(np.int64))
        ct = corner_targets([dihedral_polygon(p, d, TILE) for p in self.polys[i]], (TILE, TILE))
        return x, {"inst": torch.from_numpy(inst.astype(np.int64)),
                   "corner": torch.from_numpy(ct["corner"]),
                   "direction": torch.from_numpy(ct["direction"].astype(np.float16)),
                   "direction_mask": torch.from_numpy(ct["direction_mask"])}


class S2Loader:
    """Moves batches to the device and standardises each band with the train statistics."""

    def __init__(self, loader, device):
        st = json.loads((CACHE / "stats.json").read_text())
        self.mean = torch.tensor(st["mean"], device=device).view(1, 8, 1, 1)
        self.std = torch.tensor(st["std"], device=device).view(1, 8, 1, 1)
        self.loader, self.device = loader, device

    def __len__(self):
        return len(self.loader)

    def norm(self, x):
        return (x.to(self.device, non_blocking=True).float() - self.mean) / self.std

    def __iter__(self):
        for x, y in self.loader:
            y = ({k: v.to(self.device, non_blocking=True) for k, v in y.items()} if isinstance(y, dict)
                 else y.to(self.device, non_blocking=True))
            yield self.norm(x), y


def build(arm: str):
    heads = instance_heads() if arm == "B0" else CornerLineTask().heads()
    model = create_model(backbone="resnet34.tv_in1k", in_channels=8, pretrained=True, heads=heads)
    if arm == "B3":  # CenterNet prior (E036)
        torch.nn.init.constant_(model.heads.corner.conv.bias, -2.19)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["B0", "B3"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out = HERE / "runs"
    out.mkdir(exist_ok=True)
    name = f"{args.arm}_s{args.seed}{args.tag}"
    seed_everything(args.seed)
    device = torch.device("cuda")

    def loader(split, train_mode):
        ds = FTWChips(split, args.arm, augment=train_mode)
        if args.limit:
            ds = torch.utils.data.Subset(ds, range(min(args.limit, len(ds))))
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=train_mode, drop_last=train_mode,
                                         num_workers=args.workers, pin_memory=True,
                                         persistent_workers=args.workers > 0,
                                         generator=torch.Generator().manual_seed(args.seed))
        return S2Loader(dl, device)

    model = build(args.arm)
    enc, dec = split_params(model, model.backbone)
    task = InstanceTask(boundary_width=2.0) if args.arm == "B0" else CornerLineTask()
    recipe = get_recipe("fastai_like", epochs=args.epochs, batch_size=args.batch_size, eval_every=min(10, args.epochs))
    t0 = time.time()
    res = train(model, enc, dec, loader("train", True), loader("val", False), num_classes=1, recipe=recipe,
                device=device, task=task)
    torch.save(model.state_dict(), out / f"{name}.pt")
    res.update(arm=args.arm, seed=args.seed, recipe=recipe.to_dict(), wall_s=time.time() - t0, limit=args.limit)
    (out / f"{name}_train.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"done {name}: {res['final_val']} in {res['wall_s'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
