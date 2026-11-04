"""E051: end-to-end large-scene vectors through ManyPred (merge="cut", BlockPostprocess) and the
manyseg.vectors block decoders, against a whole-scene decode of the same stitched fields.

    PYTHONPATH=<ManyPred feature/vector-blocks>/src uv run python e051.py roofs|roads [--workers 4]
"""

from __future__ import annotations

import os
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from affine import Affine

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
DATA = Path(os.environ.get("MANYSEG_BENCH_DATA", "data"))
OUT = DATA / "cache" / "e051_out"
sys.path[:0] = [str(EXP / d) for d in ("E030_whu_polygons", "E031_lod2_oracle", "E047_lod2_trained",
                                       "E044_cityscale_unified", "E039_unified_roads", "E028_vector_oracle")]

from manypred import BlockPostprocess, predict_files  # noqa: E402
from manypred.geo import RasterioReader  # noqa: E402
from manyseg.vectors import block_decoder, decode_scene, write_records  # noqa: E402

PATCH, OVERLAP, BLOCK, HALO = 1024, 256, 1024, 256


class RoofModel(torch.nn.Module):
    """E047b roof-face model on raw RGB (0-255) + nDSM (m); returns activated heads plus the
    input nDSM as the "height" field the roof decoder fits planes to."""

    def __init__(self):
        super().__init__()
        import train47 as T
        self.T = T
        self.net = T.build()
        self.net.load_state_dict(torch.load(EXP / "E047_lod2_trained/runs/B3_s0_e150.pt", map_location="cpu"))
        self.net.eval()

    def forward(self, x):
        T, x = self.T, x.float()
        z = torch.cat([(x[:, :3] - T.MEAN.to(x.device)) / T.STD.to(x.device), x[:, 3:] / T.NDSM_SCALE], 1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            o = self.net(z)
        return {"extent": torch.sigmoid(o["extent"].float()), "boundary": torch.sigmoid(o["boundary"].float()),
                "distance": o["distance"].float(), "corner": torch.sigmoid(o["corner"].float()),
                "direction": o["direction"].float(), "height": x[:, 3:4]}


class RoadModel(torch.nn.Module):
    """E044 unified road model (seed 0) on raw RGB; library head names."""

    def __init__(self):
        super().__init__()
        from train44 import build
        self.net = build()
        runs = Path(os.environ.get("MANYSEG_RUNS", "../ManySeg-runs"), "e044-90736d4/experiments/E044_cityscale_unified/runs")
        self.net.load_state_dict(torch.load(runs / "U_s0.pt", map_location="cpu"))
        self.net.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)

    def forward(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            o = self.net((x.float() - self.mean) / self.std)
        return {"mask": torch.sigmoid(o["mask"].float()), "line": torch.sigmoid(o["line"].float()),
                "keypoints": torch.sigmoid(o["kp"].float()), "direction": o["direction"].float()}


def write_scene(task: str, crop: int | None = None) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    if task == "roofs":
        src = DATA / "cache" / "e051_large"
        meta = json.loads((src / "leidscherijn_4096.json").read_text())
        rgb = np.load(src / "leidscherijn_4096_rgb.npy")
        nd = np.load(src / "leidscherijn_4096_ndsm.npy").astype(np.float32)
        arr = np.concatenate([rgb.transpose(2, 0, 1).astype(np.float32), nd[None]], 0)
        prof = dict(crs=meta["crs"], transform=Affine(meta["gsd"], 0, meta["x0"], 0, -meta["gsd"], meta["y1"]))
    else:
        from PIL import Image
        arr = np.asarray(Image.open(DATA / "raw/rngdet_roads/cityscale/data/20cities/region_8_sat.png").convert("RGB"))
        arr = arr.transpose(2, 0, 1).astype(np.float32)
        prof = dict(crs=None, transform=Affine(1, 0, 0, 0, -1, 0))  # pixel grid (City-Scale has no georeferencing)
    if crop:
        arr = arr[:, :crop, :crop]
    path = OUT / f"{task}_scene{'_crop' if crop else ''}.tif"
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
                       dtype="float32", tiled=True, blockxsize=512, blockysize=512, **prof) as dst:
        dst.write(arr)
    return path


def load(item):
    reader = RasterioReader(item)
    return reader, reader.profile


def decoder(task):
    if task == "roofs":
        ev = json.loads((EXP / "E047_lod2_trained/runs/eval_B3_s0_e150.json").read_text())["selected"]["F3"]
        return block_decoder("roofs", **ev["thr"], **ev["params"])
    ev = json.loads(Path(os.environ.get("MANYSEG_RUNS", "../ManySeg-runs"), "e044-90736d4/experiments/E044_cityscale_unified/runs/eval_U_s0.json").read_text())
    r0, r4 = ev["r0_params"], ev["r4_params"]
    return block_decoder("roads", threshold=r0["thr"], keypoint_threshold=r4["thr_k"], zone_radius=r4["zr"],
                         pair_deg=r4["pair_deg"], spur=r0["spur"], gap=r0["gap"], simplify=r0["simplify"],
                         min_length=r0["min_len"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["roofs", "roads"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--crop", type=int, default=None)  # smoke runs: top-left crop size
    args = ap.parse_args()
    task = args.task
    path = write_scene(task, args.crop)
    model = RoofModel() if task == "roofs" else RoadModel()
    dec = decoder(task)
    saved = {}
    post = BlockPostprocess(dec, block=BLOCK, halo=HALO, combine=dec.combine, workers=args.workers,
                            save=lambda result, item, scene: saved.setdefault("hook", (result, scene)))
    t0 = time.time()
    predict_files(model, [path], load_func=load, output_dir=OUT / f"{path.stem}_pred", patch_size=PATCH, overlap=OVERLAP,
                  device=args.device, merge="cut", block_postprocess=post, progress=False, batch_size=2, dtype="float32")
    t_hook = time.time() - t0
    hook, scene = saved["hook"]
    # whole-scene decode of the same stitched rasters (written by the same run)
    fields = {}
    for f in sorted((OUT / f"{path.stem}_pred").glob(f"{path.stem}_pred_*.tif")):
        name = f.stem.split("_pred_")[1]
        with rasterio.open(f) as r:
            fields[name] = r.read().astype(np.float32)
    H, W = next(iter(fields.values())).shape[1:]
    t0 = time.time()
    whole = decode_scene(fields, dec, block=max(H, W), halo=HALO)
    t_whole = time.time() - t0
    t0 = time.time()
    blocks = decode_scene(fields, dec, block=BLOCK, halo=HALO)
    t_blocks = time.time() - t0
    res = {"task": task, "shape": [H, W], "hook_s": t_hook, "whole_decode_s": t_whole, "block_decode_s": t_blocks}
    if task == "roofs":
        def key(rs):
            return {(r["id"], r.get("part", 0)): r for r in rs}
        kh, kw, kb = key(hook), key(whole), key(blocks)
        res.update(n_hook=len(kh), n_whole=len(kw), n_blocks=len(kb),
                   same_keys=set(kh) == set(kw) == set(kb),
                   geom_equal=sum(kh[k]["geometry"].equals_exact(kw[k]["geometry"], 0) for k in kh if k in kw),
                   plane_max_diff=float(max((np.max(np.abs(np.asarray(kh[k]["plane"]) - np.asarray(kw[k]["plane"])))
                                             for k in kh if k in kw), default=0.0)))
        write_records(hook, OUT / "roofs_leidscherijn.gpkg", scene)
        write_records(hook, OUT / "roofs_leidscherijn_3d.gpkg", scene, mesh=True)
    else:
        def arrs(g):
            return np.asarray(g.xy), np.asarray(g.edges)
        hn, he = arrs(hook)
        wn, we = arrs(whole)
        res.update(nodes_hook=len(hn), nodes_whole=len(wn), edges_hook=len(he), edges_whole=len(we),
                   nodes_equal=bool(np.array_equal(hn, wn)), edges_equal=bool(np.array_equal(he, we)))
    print(json.dumps(res, indent=1, default=str))
    (OUT / f"{task}_result.json").write_text(json.dumps(res, indent=1, default=str))
    with open(OUT / f"{task}_vis.pkl", "wb") as fh:
        pickle.dump({"hook": hook, "scene": (scene.height, scene.width)}, fh)


if __name__ == "__main__":
    main()
