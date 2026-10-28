"""Trained F3 faces -> robust planes -> vertex snap (1.5 px) -> E043b J2 mesh; 3D render vs 3DBAG."""
import sys, json, numpy as np, torch, shapely, warnings, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree
warnings.filterwarnings("ignore")
sys.path[:0] = ["../E030_whu_polygons", "../E031_lod2_oracle", "../E043_roof_mesh", "."]
import evaluate47 as V, e043, train47 as T
from common import b3_probabilities
from manyseg.instances import instances_from_outputs
from manyseg.vectors import polygons_from_corners
S = sys.argv[1]
sel = json.load(open("runs/eval_B3_s0_e150.json"))["selected"]["F3"]
m = T.build(); m.load_state_dict(torch.load("runs/B3_s0_e150.pt", map_location="cpu")); m = m.cuda().eval()
tiles = V._tiles("test"); ds = T.RoofTiles("test", augment=False, targets=False)
nds = np.load(V.CACHE / "test_ndsm.npy", mmap_mode="r")
mean, std = T.MEAN.cuda(), T.STD.cuda()

def snap(faces, r=1.5):
    pts = [np.asarray(f.exterior.coords)[:-1] for f in faces]
    allp = np.concatenate(pts); owner = np.concatenate([[i] * len(p) for i, p in enumerate(pts)])
    tree = cKDTree(allp); parent = np.arange(len(allp))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, j in tree.query_pairs(r):
        if owner[i] != owner[j]: parent[find(i)] = find(j)
    roots = np.array([find(i) for i in range(len(allp))])
    cent = {r_: allp[roots == r_].mean(0) for r_ in np.unique(roots)}
    out, k = [], 0
    for p in pts:
        q = np.array([cent[roots[k + j]] for j in range(len(p))]); k += len(p)
        poly = shapely.Polygon(q)
        out.append(poly if poly.is_valid and poly.area > 1 else shapely.Polygon(p))
    return out

def draw(ax, rings, color):
    polys = [[(x * 0.25, -y * 0.25, z) for x, y, z in r] for r in rings if len(r) >= 3]
    ax.add_collection3d(Poly3DCollection(polys, facecolor=color, edgecolor="k", linewidths=0.3, alpha=0.9))

fig = plt.figure(figsize=(18, 12))
for row, (i, box) in enumerate(((5, None), (30, None))):
    x = ds[i][0][None].cuda().float(); x = torch.cat([(x[:, :3] - mean) / std, x[:, 3:] / T.NDSM_SCALE], 1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): out = m(x)
    p = b3_probabilities(out, 0); tt = lambda k: torch.from_numpy(p[k].astype(np.float32))[None, None]
    lab = instances_from_outputs(tt("extent"), tt("boundary"), tt("distance"), logits=False, min_size=16, **sel["thr"])[0]
    faces = polygons_from_corners(lab, p["corner"].astype(np.float32), p["direction"].astype(np.float32), snap_radius=4.0, simplify=1.0, min_area=16, **sel["params"])
    faces = [f for f in faces if f.geom_type == "Polygon"]
    a = {"height": np.asarray(nds[i], np.float32)}
    sf = snap(faces); P = e043.planes(sf, a, "robust"); rings, st = e043.mesh(sf, P, "J2")
    raw_rings, raw_st = e043.mesh(faces, e043.planes(faces, a, "robust"), "J2")
    t = tiles[i]; grings, _ = e043.mesh(t.faces, t.planes, "J1")
    # crop to the densest 48 m window
    c = np.array([f.centroid.coords[0] for f in t.faces]); cx, cy = np.median(c, 0)
    keep = lambda R: [r for r in R if abs(r[:, 0].mean() - cx) < 96 and abs(r[:, 1].mean() - cy) < 96]
    for col, (R, title, colr) in enumerate(((keep(grings), f"3DBAG LoD2.2 ({t.area})", "#9ab"), (keep(rings), f"trained F3 + planes + snap + J2: shared vertices {st['shared_vertices']} (no snap {raw_st['shared_vertices']}), steps {st['step_vertices']}", "#c96"))):
        ax = fig.add_subplot(2, 2, row * 2 + col + 1, projection="3d"); draw(ax, R, colr)
        ax.set_xlim((cx - 96) * 0.25, (cx + 96) * 0.25); ax.set_ylim(-(cy + 96) * 0.25, -(cy - 96) * 0.25); ax.set_zlim(0, 20)
        ax.view_init(35, -60); ax.set_title(title, fontsize=9); ax.set_box_aspect((1, 1, 0.4))
plt.tight_layout(); plt.savefig(f"{S}/e047_mesh3d.png", dpi=70)
print("ok")
