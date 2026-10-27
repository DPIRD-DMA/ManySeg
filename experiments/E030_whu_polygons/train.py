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


def task_for(run_arm: str):
    """Training task from the run's arm name: B0, B2, or B2 + E032 / E033 loss letters
    (w = decode-cost weighting, e = w + endpoint, d = distance-weighted vertex loss,
    s = snap offset head)."""
    if run_arm == "B0":
        return InstanceTask(boundary_width=2.0)
    opts = run_arm[2:]
    return GraphTensorTask(weighted=bool(set(opts) & {"w", "e"}), endpoint=0.1 if "e" in opts else 0.0,
                           vertex_loss="distance" if "d" in opts else "bce", snap="s" in opts)


def build(run_arm: str):
    heads = instance_heads() if run_arm == "B0" else task_for(run_arm).heads()
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
    ap.add_argument("--loss", choices=["base", "w", "e", "d", "s"], default="base",
                    help="B2 loss: base = E030; E032: w = decode-cost weighting, e = w + endpoint; "
                         "E033: d = distance-weighted vertex BCE, s = snap offset head")
    args = ap.parse_args()
    if args.arm == "B0" and args.loss != "base":
        ap.error("--loss options apply to --arm B2 only")
    out = HERE / "runs"
    out.mkdir(exist_ok=True)
    suffix = "" if args.loss == "base" else args.loss
    name = f"{args.arm}{suffix}_s{args.seed}{args.tag}"
    seed_everything(args.seed)
    device = torch.device("cuda")

    def loader(split, train_mode):
        extras = tuple(x for x, f in (("vdist", "d"), ("snap", "s")) if f == args.loss)
        ds = WHUTiles(args.cache, split, args.arm, augment=train_mode, extras=extras)
        if args.limit:
            ds = torch.utils.data.Subset(ds, range(min(args.limit, len(ds))))
        g = torch.Generator().manual_seed(args.seed)
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=train_mode,
                                         drop_last=train_mode, num_workers=args.workers,
                                         pin_memory=True, persistent_workers=True, generator=g)
        return DeviceLoader(dl, device)

    run_arm = name.split("_")[0]
    model = build(run_arm)
    enc, dec = split_params(model, model.backbone)
    task = task_for(run_arm)
    recipe = get_recipe("fastai_like", epochs=args.epochs, batch_size=args.batch_size,
                        eval_every=10)
    t0 = time.time()
    res = train(model, enc, dec, loader("train", True), loader("val", False), num_classes=1,
                recipe=recipe, device=device, task=task)
    torch.save(model.state_dict(), out / f"{name}.pt")
    res.update(arm=args.arm, loss=args.loss, seed=args.seed, recipe=recipe.to_dict(), wall_s=time.time() - t0,
               limit=args.limit)
    (out / f"{name}_train.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"done {name}: {res['final_val']} in {res['wall_s'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
