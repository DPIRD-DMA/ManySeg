"""E049: RGB-only roof model with a height head (H0) or height + slope heads (H1) (README).

    uv run python train49.py targets                       # building-only height target (once)
    uv run python train49.py train --arm H0 --seed 0 [--epochs 150] [--limit N]
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
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
E47 = HERE.parent / "E047_lod2_trained"
for d in ("E030_whu_polygons", "E031_lod2_oracle"):
    sys.path.insert(0, str(HERE.parent / d))
sys.path.insert(0, str(E47))
import e031 as E  # noqa: E402
from common import CornerLineTask, dihedral_array, seed_everything  # noqa: E402
from manyseg import create_model  # noqa: E402
from manyseg_bench.providers.base import split_params  # noqa: E402
from manyseg_bench.training.recipe import get_recipe  # noqa: E402
from manyseg_bench.training.trainer import train  # noqa: E402
from train47 import CACHE, MEAN, STD  # noqa: E402

H_SCALE = 10.0  # m; the height head predicts metres / 10


def make_targets():
    """Building-only height: 3DBAG roof planes per pixel (E031 ``height_map``), 0 off roofs."""
    for split in ("train", "val", "test"):
        tiles = pickle.loads((CACHE / f"{split}_tiles.pkl").read_bytes())
        h = np.lib.format.open_memmap(CACHE / f"{split}_height.npy", "w+", np.float16, (len(tiles), 512, 512))
        for i, t in enumerate(tiles):
            hm, m = E.height_map(t.faces, t.planes)
            h[i] = np.where(m, np.clip(hm, 0, 100), 0)
        h.flush()
        print(split, len(tiles), flush=True)


class RGBRoofTiles(torch.utils.data.Dataset):
    """RGB only; E047's targets plus ``height`` (m), all under the same dihedral transform."""

    def __init__(self, split: str, augment: bool, targets: bool = True):
        self.img = np.load(CACHE / f"{split}_img.npy", mmap_mode="r")
        self.inst = np.load(CACHE / f"{split}_inst.npy", mmap_mode="r")
        self.h = np.load(CACHE / f"{split}_height.npy", mmap_mode="r")
        if targets:
            self.c = np.load(CACHE / f"{split}_corner.npy", mmap_mode="r")
            self.d = np.load(CACHE / f"{split}_dir.npy", mmap_mode="r")
            self.m = np.load(CACHE / f"{split}_dmask.npy", mmap_mode="r")
        self.augment, self.targets = augment, targets

    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        k = int(torch.randint(8, (1,))) if self.augment else 0
        x = torch.from_numpy(dihedral_array(np.asarray(self.img[i], np.float32), k)).permute(2, 0, 1)
        inst = torch.from_numpy(dihedral_array(np.asarray(self.inst[i]), k).astype(np.int64))
        if not self.targets:
            return x, inst
        d = dihedral_array(np.asarray(self.d[i]).transpose(1, 2, 0), k).transpose(2, 0, 1).astype(np.float32) / 127
        d = d * ((-1) ** (k % 4))  # E047's verified direction transform
        if k >= 4:
            d[1] = -d[1]
        return x, {"inst": inst, "corner": torch.from_numpy(dihedral_array(np.asarray(self.c[i], np.float32), k)),
                   "direction": torch.from_numpy(np.ascontiguousarray(d)).half(),
                   "direction_mask": torch.from_numpy(dihedral_array(np.asarray(self.m[i]), k)),
                   "height": torch.from_numpy(dihedral_array(np.asarray(self.h[i], np.float32), k))}


class RGBLoader:
    def __init__(self, loader, device):
        self.loader, self.device = loader, device
        self.mean, self.std = MEAN.to(device), STD.to(device)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for x, y in self.loader:
            y = ({k: v.to(self.device, non_blocking=True) for k, v in y.items()} if isinstance(y, dict)
                 else y.to(self.device, non_blocking=True))
            yield (x.to(self.device, non_blocking=True).float() - self.mean) / self.std, y


def slope_target(h: torch.Tensor, inst: torch.Tensor):
    """Central-difference (dz/dx, dz/dy) in m / px on interior roof pixels (the pixel and its 4
    neighbours in one face); returns (B, 2, H-2, W-2) target and (B, H-2, W-2) mask."""
    c = inst[:, 1:-1, 1:-1]
    m = (c > 0) & (inst[:, 1:-1, 2:] == c) & (inst[:, 1:-1, :-2] == c) & (inst[:, 2:, 1:-1] == c) & (inst[:, :-2, 1:-1] == c)
    gx = (h[:, 1:-1, 2:] - h[:, 1:-1, :-2]) / 2
    gy = (h[:, 2:, 1:-1] - h[:, :-2, 1:-1]) / 2
    return torch.stack([gx, gy], 1), m


class HeightTask(CornerLineTask):
    """B3 roof heads + height (L1, all pixels, m / 10) [+ slope (L1, interior roof pixels)]."""

    def __init__(self, slope: bool):
        super().__init__()
        self.slope = slope

    def heads(self):
        h = {**super().heads(), "height": {"type": "regression", "out_channels": 1}}
        if self.slope:
            h["slope"] = {"type": "regression", "out_channels": 2}
        return h

    def loss(self, out, t):
        loss = super().loss(out, t)
        h = t["height"].float()
        loss = loss + F.l1_loss(out["height"][:, 0].float(), h / H_SCALE)
        if self.slope:
            st, m = slope_target(h, t["inst"])
            ps = out["slope"].float()[:, :, 1:-1, 1:-1]
            mm = m[:, None].expand_as(ps)
            loss = loss + (F.l1_loss(ps[mm], st[mm]) if mm.any() else ps.sum() * 0)
        return loss


def build(arm: str):
    task = HeightTask(slope=arm == "H1")
    model = create_model(backbone="resnet34.tv_in1k", in_channels=3, pretrained=True, heads=task.heads())
    torch.nn.init.constant_(model.heads.corner.conv.bias, -2.19)  # CenterNet prior (E036)
    return model, task


def probabilities(out: dict, i: int) -> dict[str, np.ndarray]:
    from common import b3_probabilities

    p = b3_probabilities(out, i)
    p["height"] = np.clip(out["height"][i, 0].float().cpu().numpy() * H_SCALE, 0, None)
    if "slope" in out:
        p["slope"] = out["slope"][i].float().cpu().numpy()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["targets", "train"])
    ap.add_argument("--arm", choices=["H0", "H1"], default="H0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.cmd == "targets":
        return make_targets()
    out = HERE / "runs"
    out.mkdir(exist_ok=True)
    name = f"{args.arm}_s{args.seed}{args.tag}"
    seed_everything(args.seed)
    device = torch.device("cuda")

    def loader(split, train_mode):
        ds = RGBRoofTiles(split, augment=train_mode)
        if args.limit:
            ds = torch.utils.data.Subset(ds, range(min(args.limit, len(ds))))
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=train_mode, drop_last=train_mode,
                                         num_workers=args.workers, pin_memory=True,
                                         persistent_workers=args.workers > 0,
                                         generator=torch.Generator().manual_seed(args.seed))
        return RGBLoader(dl, device)

    model, task = build(args.arm)
    enc, dec = split_params(model, model.backbone)
    recipe = get_recipe("fastai_like", epochs=args.epochs, batch_size=args.batch_size, eval_every=min(10, args.epochs))
    t0 = time.time()
    res = train(model, enc, dec, loader("train", True), loader("val", False), num_classes=1, recipe=recipe,
                device=device, task=task)
    torch.save({"arm": args.arm, "seed": args.seed, "state_dict": model.state_dict()}, out / f"{name}.pt")
    res.update(arm=args.arm, seed=args.seed, recipe=recipe.to_dict(), wall_s=time.time() - t0, limit=args.limit)
    (out / f"{name}_train.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"done {name}: {res['final_val']} in {res['wall_s'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
