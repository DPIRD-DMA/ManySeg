"""E052: train E046's B0 with a normalised (N) or absolute (A) distance target (README).

    uv run python train52.py --arm {N,A} --seed 0 [--epochs 40] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "E030_whu_polygons"))
sys.path.insert(0, str(HERE.parent / "E046_ftw_trained"))
from common import seed_everything  # noqa: E402
from manyseg import create_model  # noqa: E402
from manyseg_bench.providers.base import split_params  # noqa: E402
from manyseg_bench.tasks import InstanceTask  # noqa: E402
from manyseg_bench.training.recipe import get_recipe  # noqa: E402
from manyseg_bench.training.trainer import train  # noqa: E402
from train46 import FTWChips, S2Loader  # noqa: E402

DISTANCE = {"N": "normalized", "A": "absolute"}


def task(arm: str) -> InstanceTask:
    return InstanceTask(boundary_width=2.0, distance=DISTANCE[arm])


def build(arm: str):
    return create_model(backbone="resnet34.tv_in1k", in_channels=8, pretrained=True, heads=task(arm).heads())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["N", "A"], required=True)
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
        ds = FTWChips(split, "B0", augment=train_mode)
        if args.limit:
            ds = torch.utils.data.Subset(ds, range(min(args.limit, len(ds))))
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=train_mode, drop_last=train_mode,
                                         num_workers=args.workers, pin_memory=True,
                                         persistent_workers=args.workers > 0,
                                         generator=torch.Generator().manual_seed(args.seed))
        return S2Loader(dl, device)

    model = build(args.arm)
    enc, dec = split_params(model, model.backbone)
    recipe = get_recipe("fastai_like", epochs=args.epochs, batch_size=args.batch_size, eval_every=min(10, args.epochs))
    t0 = time.time()
    res = train(model, enc, dec, loader("train", True), loader("val", False), num_classes=1, recipe=recipe,
                device=device, task=task(args.arm))
    torch.save(model.state_dict(), out / f"{name}.pt")
    res.update(arm=args.arm, distance=DISTANCE[args.arm], seed=args.seed, recipe=recipe.to_dict(),
               wall_s=time.time() - t0, limit=args.limit)
    (out / f"{name}_train.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"done {name}: {res['final_val']} in {res['wall_s'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
