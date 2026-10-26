"""E030: train one arm and seed with the bench trainer; saves the weights and the history.

    uv run python experiments/E030_whu_polygons/train.py --arm B2 --seed 0 --cache "$E030_CACHE"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from common import DeviceLoader, GraphTensorTask, WHUTiles, seed_everything
from manyseg import create_model
from manyseg.instances import instance_heads
from manyseg_bench.providers.base import split_params
from manyseg_bench.tasks import InstanceTask
from manyseg_bench.training.recipe import get_recipe
from manyseg_bench.training.trainer import train

HERE = Path(__file__).resolve().parent


def build(arm: str):
    heads = instance_heads() if arm == "B0" else GraphTensorTask().heads()
    model = create_model(backbone="resnet34.tv_in1k", in_channels=3, pretrained=True, heads=heads)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["B0", "B2"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N tiles per split")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out = HERE / "runs"
    out.mkdir(exist_ok=True)
    name = f"{args.arm}_s{args.seed}{args.tag}"
    seed_everything(args.seed)
    device = torch.device("cuda")

    def loader(split, train_mode):
        ds = WHUTiles(args.cache, split, args.arm, augment=train_mode)
        if args.limit:
            ds = torch.utils.data.Subset(ds, range(min(args.limit, len(ds))))
        g = torch.Generator().manual_seed(args.seed)
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=train_mode,
                                         drop_last=train_mode, num_workers=args.workers,
                                         pin_memory=True, persistent_workers=True, generator=g)
        return DeviceLoader(dl, device)

    model = build(args.arm)
    enc, dec = split_params(model, model.backbone)
    task = InstanceTask(boundary_width=2.0) if args.arm == "B0" else GraphTensorTask()
    recipe = get_recipe("fastai_like", epochs=args.epochs, batch_size=args.batch_size,
                        eval_every=10)
    t0 = time.time()
    res = train(model, enc, dec, loader("train", True), loader("val", False), num_classes=1,
                recipe=recipe, device=device, task=task)
    torch.save(model.state_dict(), out / f"{name}.pt")
    res.update(arm=args.arm, seed=args.seed, recipe=recipe.to_dict(), wall_s=time.time() - t0,
               limit=args.limit)
    (out / f"{name}_train.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"done {name}: {res['final_val']} in {res['wall_s'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
