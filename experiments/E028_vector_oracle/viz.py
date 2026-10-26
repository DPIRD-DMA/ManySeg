"""Visual checks for E028: GT vectors, dense targets, corrupted fields and decoded outputs.

    uv run python experiments/E028_vector_oracle/viz.py --data "$E028_DATA" --out figs \
        [--area murdoch --task roads --tile 5 --level 2 --params runs/best.json]
    uv run python experiments/E028_vector_oracle/viz.py --overview --data "$E028_DATA" --out figs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import e028 as E

DEFAULT = {
    "R0": dict(thr=0.5, spur=20, simplify=1, min_len=20, gap=25),
    "R1": dict(thr=0.5, spur=20, simplify=1, min_len=20, thr_j=0.4, merge_r=8, kernel=5, gap=25),
    "R2": dict(thr_v=0.3, thr_e=0.5, link_r=6, mutual=False, kernel=5, gap=25),
    "B0": dict(thr=0.5, thr_b=0.5, thr_s=0.3, simplify=1.0),
    "B1": dict(thr=0.5, thr_b=0.5, thr_s=0.3, simplify=1.0, thr_c=0.4, snap_r=4, kernel=3),
    "B2": dict(thr_v=0.4, thr_e=0.5, link_r=6, mutual=False, kernel=3, simplify=0.5, face_thr=0.5),
}
REPS = {"roads": ("R0", "R1", "R2"), "buildings": ("B0", "B1", "B2")}


def draw_graph(ax, g, color, lw=1.0, nodes=False):
    for u, v in g.edges():
        a, b = g.nodes[u]["xy"], g.nodes[v]["xy"]
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw)
    if nodes:
        j = E.junctions(g)
        if len(j):
            ax.scatter(j[:, 0], j[:, 1], s=8, c=color, edgecolors="k", linewidths=0.3, zorder=5)


def draw_polys(ax, polys, color, lw=1.0, vertices=False):
    for p in polys:
        x, y = p.exterior.xy
        ax.plot(x, y, color=color, lw=lw)
        if vertices:
            ax.scatter(x[:-1], y[:-1], s=4, c=color, zorder=5)


def draw_gt(ax, t, lw=0.8):
    if t.task == "roads":
        draw_graph(ax, t.graph, "#1f77b4", lw)
        if t.crossings:
            c = np.array(t.crossings)
            ax.scatter(c[:, 0], c[:, 1], marker="x", c="m", s=30, zorder=6)
    else:
        draw_polys(ax, t.polygons, "#1f77b4", lw)


def view(ax, title, crop):
    ax.set_title(title, fontsize=8)
    ax.set_xlim(crop[0], crop[1])
    ax.set_ylim(crop[3], crop[2])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def field_image(rep, a):
    """One display image per representation: the main probability channel(s)."""
    if rep in ("R0", "R1"):
        img = np.stack([a["mask"], a.get("junction", np.zeros_like(a["mask"])), np.zeros_like(a["mask"])], -1)
    elif rep in ("R2", "B2"):
        base = a.get("mask", np.zeros_like(a["vertex"])) * 0.4
        img = np.stack([np.maximum(base, a["vertex"]), a["edge"].max(0) * 0.6, base], -1)
    else:
        img = np.stack([a["extent"] * 0.6, a["boundary"], a.get("corner", a["distance"])], -1)
    return np.clip(img, 0, 1)


def tile_figure(t, level_cfgs, params, out: Path, crop=None, seed=0):
    reps = REPS[t.task]
    crop = crop or (0, E.TILE, 0, E.TILE)
    rows = len(level_cfgs)
    fig, axes = plt.subplots(rows, 2 * len(reps), figsize=(4 * len(reps) * 1.0, 2.1 * rows), squeeze=False)
    for i, (lname, cfg) in enumerate(level_cfgs):
        for k, rep in enumerate(reps):
            tg = E.encode(t, rep)
            a = E.corrupt(tg, t, cfg, seed)
            pred = E.DECODERS[rep](a, params[rep] if lname not in params.get("_by_level", {})
                                   else params["_by_level"][lname][rep])
            m = E.evaluate(t, rep, pred)
            ax = axes[i, 2 * k]
            ax.imshow(field_image(rep, a), interpolation="nearest")
            view(ax, f"{rep} fields, {lname}", crop)
            ax = axes[i, 2 * k + 1]
            ax.imshow(np.ones((E.TILE, E.TILE, 3)), interpolation="nearest")
            draw_gt(ax, t, lw=2.0)
            if t.task == "roads":
                draw_graph(ax, pred, "#d62728", 0.8, nodes=True)
                s = f"APLS {m['apls']:.3f} JF1 {m['junction_f1']:.2f}"
            else:
                draw_polys(ax, pred, "#d62728", 0.8, vertices=True)
                s = f"C-IoU {m['ciou']:.3f} F1 {m['det_f1']:.2f} N {m['n_ratio']:.2f}"
            view(ax, f"{rep} decoded ({s})", crop)
    fig.suptitle(f"{t.area} {t.task} tile {t.index}: GT blue, decoded red", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def overview(data: Path, out: Path):
    """Each area: all GT roads and buildings with the road and building tile grids."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4))
    for ax, name in zip(axes, ("innaloo", "murdoch", "northam")):
        d = E.load_area(data / f"{name}.json")
        for r in d["roads"]:
            c = np.array(r["coords"])
            ax.plot(c[:, 0], c[:, 1], lw=0.6, color="m" if r["bridge"] else "k")
        for b in d["buildings"]:
            c = np.array(b["coords"])
            ax.fill(c[:, 0], c[:, 1], color="tab:orange", lw=0)
        xmin, ymin, xmax, ymax = E.inner_box(d["bbox_wgs84"], d["crs"])
        ax.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], "b--", lw=1)
        for task, col in (("roads", "tab:blue"), ("buildings", "tab:green")):
            size = E.TILE * E.GSD[task]
            for x0, y1 in E.tile_origins((xmin, ymin, xmax, ymax), task):
                ax.plot([x0, x0 + size, x0 + size, x0, x0], [y1, y1, y1 - size, y1 - size, y1],
                        color=col, lw=0.4, alpha=0.6)
        ax.set_xlim(xmin - 100, xmax + 100)
        ax.set_ylim(ymin - 100, ymax + 100)
        ax.set_aspect("equal")
        ax.set_title(f"{name}: {len(d['roads'])} roads, {len(d['buildings'])} buildings "
                     "(bridges magenta)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--overview", action="store_true")
    ap.add_argument("--area", default="murdoch")
    ap.add_argument("--task", default="roads")
    ap.add_argument("--tile", type=int, default=None, help="tile position in the area's list")
    ap.add_argument("--levels", default="0,1,2,3")
    ap.add_argument("--params", type=Path, default=None, help="tuned params JSON {level: {rep: params}}")
    ap.add_argument("--crop", default=None, help="x0,x1,y0,y1 in px")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.overview:
        overview(args.data, args.out)
        return
    d = E.load_area(args.data / f"{args.area}.json")
    tiles = E.road_tiles(args.area, d) if args.task == "roads" else E.building_tiles(args.area, d)
    t = tiles[args.tile if args.tile is not None else len(tiles) // 2]
    levels = [int(x) for x in args.levels.split(",")]
    params = dict(DEFAULT)
    if args.params:
        best = json.loads(args.params.read_text())
        params["_by_level"] = {f"L{lv}": best[str(lv)] for lv in levels}
    crop = tuple(float(v) for v in args.crop.split(",")) if args.crop else None
    tile_figure(t, [(f"L{lv}", E.LEVELS[lv]) for lv in levels], params,
                args.out / f"{args.area}_{args.task}_{t.index}.png", crop)


if __name__ == "__main__":
    main()
