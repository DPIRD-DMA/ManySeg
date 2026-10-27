"""E031 visual check: ground-truth roof faces and heights, and decoded faces / heights per arm.

    uv run python experiments/E031_lod2_oracle/viz.py --area delft --tile 0 --levels 0,2 --out figs
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import e031 as E

DEFAULT = {"A0": dict(smooth=1.0, tg=0.06, tz=0.5, min_px=60, simplify=1.0),
           "A1": dict(vertex_threshold=0.2, link_radius=10.0, gap=25.0, simplify=1.5),
           "LoD1": {}}


def faces_plot(ax, faces, color, lw=0.8):
    for p in faces:
        x, y = p.exterior.xy
        ax.plot(x, y, color=color, lw=lw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="delft")
    ap.add_argument("--tile", type=int, default=0)
    ap.add_argument("--levels", default="0,2")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--params", type=Path, default=None)
    args = ap.parse_args()
    data = Path(os.environ["E031_DATA"])
    t = E.make_tiles(args.area, E.load_faces(data / args.area))[args.tile]
    best = json.loads(args.params.read_text()) if args.params else None
    levels = [int(x) for x in args.levels.split(",")]
    fig, axes = plt.subplots(len(levels), 5, figsize=(25, 5 * len(levels)), squeeze=False)
    vmax = float(t.height.max())
    for r, lv in enumerate(levels):
        a = E.corrupt(E.encode(t), t, E.LEVELS[lv], 0)
        ax = axes[r, 0]
        ax.imshow(a["height"], cmap="viridis", vmin=0, vmax=vmax)
        faces_plot(ax, t.faces, "w", 0.6)
        ax.set_title(f"L{lv}: input height model + GT faces ({len(t.faces)})")
        ax = axes[r, 1]
        ax.imshow(np.stack([a["vertex"], a["edge"].max(0) * 0.6, a["mask"] * 0.3], -1).clip(0, 1))
        ax.set_title(f"L{lv}: graph tensor + mask")
        for c, arm in enumerate(("A0", "A1", "LoD1"), start=2):
            p = best[f"L{lv}"][arm]["params"] if best and arm in best[f"L{lv}"] else DEFAULT[arm]
            faces, planes = E.DECODERS[arm](a, p)
            m = E.metrics(t, faces, planes)
            h, _ = E.height_map(faces, planes)
            ax = axes[r, c]
            ax.imshow(h, cmap="viridis", vmin=0, vmax=vmax)
            faces_plot(ax, t.faces, "w", 1.2)
            faces_plot(ax, faces, "r", 0.7)
            ax.set_title(f"{arm}: face F1 {m['face_f1']:.2f}, RMSE {m['h_rmse']:.2f} m, "
                         f"normal {m['normal_err_deg']:.1f}°")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"{args.area} tile {args.tile}: GT faces white, decoded red; colour = height (m)")
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / f"{args.area}_{args.tile}.png", dpi=55)


if __name__ == "__main__":
    main()
