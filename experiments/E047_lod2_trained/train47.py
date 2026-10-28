"""E047: train the B3 roof-face model (README).

    uv run python train47.py targets              # precompute corner / direction targets
    uv run python train47.py train --seed 0 [--epochs 40] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "E030_whu_polygons"))
sys.path.insert(0, str(HERE.parent / "E031_lod2_oracle"))  # unpickling e031.Tile
from common import CornerLineTask, dihedral_array, seed_everything  # noqa: E402
from manyseg import create_model  # noqa: E402
from manyseg_bench.providers.base import split_params  # noqa: E402
from manyseg_bench.training.recipe import get_recipe  # noqa: E402
from manyseg_bench.training.trainer import train  # noqa: E402

CACHE = Path("$MANYSEG_BENCH_DATA/cache/e047_lod2")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255
NDSM_SCALE = 10.0  # m


def make_targets():
    from manyseg.vectors import corner_targets

    for split in ("train", "val", "test"):
        tiles = pickle.loads((CACHE / f"{split}_tiles.pkl").read_bytes())
        c = np.lib.format.open_memmap(CACHE / f"{split}_corner.npy", "w+", np.float16, (len(tiles), 512, 512))
        d = np.lib.format.open_memmap(CACHE / f"{split}_dir.npy", "w+", np.int8, (len(tiles), 2, 512, 512))
        m = np.lib.format.open_memmap(CACHE / f"{split}_dmask.npy", "w+", np.bool_, (len(tiles), 512, 512))
        for i, t in enumerate(tiles):
            ct = corner_targets(t.faces, (512, 512))
            c[i], m[i] = ct["corner"], ct["direction_mask"]
            d[i] = np.round(ct["direction"] * 127).astype(np.int8)
        c.flush(), d.flush(), m.flush()
        print(split, len(tiles), flush=True)


class RoofTiles(torch.utils.data.Dataset):
    """(RGB + nDSM, B3 targets); dihedral ``augment`` with the direction transform verified in
    the README (each quarter turn negates (cos 2t, sin 2t); a flip negates sin 2t)."""

    def __init__(self, split: str, augment: bool, targets: bool = True):
        self.img = np.load(CACHE / f"{split}_img.npy", mmap_mode="r")
        self.nd = np.load(CACHE / f"{split}_ndsm.npy", mmap_mode="r")
        self.inst = np.load(CACHE / f"{split}_inst.npy", mmap_mode="r")
        if targets:
            self.c = np.load(CACHE / f"{split}_corner.npy", mmap_mode="r")
            self.d = np.load(CACHE / f"{split}_dir.npy", mmap_mode="r")
            self.m = np.load(CACHE / f"{split}_dmask.npy", mmap_mode="r")
        self.augment, self.targets = augment, targets

    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        k = int(torch.randint(8, (1,))) if self.augment else 0
        x = np.concatenate([np.asarray(self.img[i], np.float32),
                            np.asarray(self.nd[i], np.float32)[..., None]], -1)
        x = torch.from_numpy(dihedral_array(x, k)).permute(2, 0, 1)
        inst = torch.from_numpy(dihedral_array(np.asarray(self.inst[i]), k).astype(np.int64))
        if not self.targets:
            return x, inst
        d = dihedral_array(np.asarray(self.d[i]).transpose(1, 2, 0), k).transpose(2, 0, 1).astype(np.float32) / 127
        d = d * ((-1) ** (k % 4))
        if k >= 4:
            d[1] = -d[1]
        return x, {"inst": inst, "corner": torch.from_numpy(dihedral_array(np.asarray(self.c[i], np.float32), k)),
                   "direction": torch.from_numpy(np.ascontiguousarray(d)).half(),
                   "direction_mask": torch.from_numpy(dihedral_array(np.asarray(self.m[i]), k))}


class RoofLoader:
    def __init__(self, loader, device):
        self.loader, self.device = loader, device
        self.mean, self.std = MEAN.to(device), STD.to(device)

    def __len__(self):
        return len(self.loader)

    def norm(self, x):
        x = x.to(self.device, non_blocking=True).float()
        return torch.cat([(x[:, :3] - self.mean) / self.std, x[:, 3:] / NDSM_SCALE], 1)

    def __iter__(self):
        for x, y in self.loader:
            y = ({k: v.to(self.device, non_blocking=True) for k, v in y.items()} if isinstance(y, dict)
                 else y.to(self.device, non_blocking=True))
            yield self.norm(x), y


def build():
    model = create_model(backbone="resnet34.tv_in1k", in_channels=4, pretrained=True, heads=CornerLineTask().heads())
    torch.nn.init.constant_(model.heads.corner.conv.bias, -2.19)  # CenterNet prior (E036)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["targets", "train"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.cmd == "targets":
        return make_targets()
    out = HERE / "runs"
    out.mkdir(exist_ok=True)
    name = f"B3_s{args.seed}{args.tag}"
    seed_everything(args.seed)
    device = torch.device("cuda")

    def loader(split, train_mode):
        ds = RoofTiles(split, augment=train_mode)
        if args.limit:
            ds = torch.utils.data.Subset(ds, range(min(args.limit, len(ds))))
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=train_mode, drop_last=train_mode,
                                         num_workers=args.workers, pin_memory=True,
                                         persistent_workers=args.workers > 0,
                                         generator=torch.Generator().manual_seed(args.seed))
        return RoofLoader(dl, device)

    model = build()
    enc, dec = split_params(model, model.backbone)
    recipe = get_recipe("fastai_like", epochs=args.epochs, batch_size=args.batch_size, eval_every=min(10, args.epochs))
    t0 = time.time()
    res = train(model, enc, dec, loader("train", True), loader("val", False), num_classes=1, recipe=recipe,
                device=device, task=CornerLineTask())
    torch.save(model.state_dict(), out / f"{name}.pt")
    res.update(seed=args.seed, recipe=recipe.to_dict(), wall_s=time.time() - t0, limit=args.limit)
    (out / f"{name}_train.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"done {name}: {res['final_val']} in {res['wall_s'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
