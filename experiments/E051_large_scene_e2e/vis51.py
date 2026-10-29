"""E051 visual check: decoded vectors over the image, with ManyPred's tile cut lines and the
decoder's block cores, overview + seam close-ups.

    uv run python vis51.py roofs|roads <out dir>
"""

import pickle
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PatchCollection  # noqa: E402
from matplotlib.patches import Polygon as MP  # noqa: E402

DATA = Path("$MANYSEG_BENCH_DATA")
OUT = DATA / "cache" / "e051_out"
PATCH, OVERLAP, BLOCK = 1024, 256, 1024
task, dest = sys.argv[1], Path(sys.argv[2])
v = pickle.load(open(OUT / f"{task}_vis.pkl", "rb"))
H, W = v["scene"]
if task == "roofs":
    img = np.load(DATA / "cache/e051_large/leidscherijn_4096_rgb.npy")[:H, :W]
else:
    from PIL import Image
    img = np.asarray(Image.open(DATA / "raw/rngdet_roads/cityscale/data/20cities/region_8_sat.png").convert("RGB"))[:H, :W]


def tile_cuts(n):  # ManyPred cut lines: middle of each overlap between neighbouring windows
    sys.path.insert(0, "$WORKSPACE/ManyPred-wt/blocks/src")
    from manypred.windows import make_patch_indexes
    starts = sorted({w[0] for w in make_patch_indexes(n, n, PATCH, OVERLAP)})
    return [(a + PATCH + b) / 2 for a, b in zip(starts[:-1], starts[1:])]


def draw(ax, box=None):
    ax.imshow((0.6 * img + 0.4 * 255).astype(np.uint8))
    if task == "roofs":
        rng = np.random.default_rng(0)
        polys = [MP(np.asarray(r["geometry"].exterior.coords), closed=True) for r in v["hook"]]
        pc = PatchCollection(polys, facecolor=rng.uniform(0.2, 0.9, (len(polys), 3)), edgecolor="k", lw=0.4, alpha=0.6)
        ax.add_collection(pc)
    else:
        g = v["hook"]
        ax.add_collection(LineCollection(g.xy[g.edges], colors="#0b63ce", lw=1.6))
        deg = np.bincount(g.edges.ravel(), minlength=len(g.xy))
        j = g.xy[deg >= 3]
        ax.scatter(j[:, 0], j[:, 1], s=10, c="#0b63ce", zorder=5)
    for c in tile_cuts(max(H, W)):
        ax.axvline(c, color="#ff00aa", lw=1.6, ls="--")
        ax.axhline(c, color="#ff00aa", lw=1.6, ls="--")
    for c in range(BLOCK, max(H, W), BLOCK):
        ax.axvline(c, color="#ffaa00", lw=1.6)
        ax.axhline(c, color="#ffaa00", lw=1.6)
    box = box or (0, W, 0, H)
    ax.set_xlim(box[0], box[1]); ax.set_ylim(box[3], box[2])
    ax.set_xticks([]); ax.set_yticks([])


fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 2, 1); draw(ax)
ax.set_title(f"{task}: whole scene {W} x {H} px; pink dashed = ManyPred cut lines (1024/256), orange = decoder block edges (1024)", fontsize=11)
c = tile_cuts(max(H, W))[0]
box = (BLOCK - 160, BLOCK + 160, BLOCK - 160, BLOCK + 160)
ax2 = fig.add_subplot(2, 2, 2); draw(ax2, box); ax2.set_title("close-up: a decoder block corner (orange)", fontsize=11)
box2 = (c - 160, c + 160, BLOCK + 300 - 160, BLOCK + 300 + 160)
ax3 = fig.add_subplot(2, 2, 4); draw(ax3, box2); ax3.set_title("close-up: a ManyPred cut line (pink)", fontsize=11)
plt.tight_layout(); plt.savefig(dest / f"e051_{task}.png", dpi=60)
print("ok")
