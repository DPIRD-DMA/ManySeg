"""E043 runner: test areas only (fixed parameters, no tuning); every plane arm x join on the same faces.

    E031_DATA=... E031B_BEST=<best_h.json> uv run python run43.py test|decide [--procs N] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import e043 as M

E = M.E
HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
TEST_AREAS, SEEDS = ("leidscherijn", "staphorst"), (0, 1, 2, 3, 4)
_T: dict = {}


def tiles(area):
    if area not in _T:
        _T[area] = E.make_tiles(area, E.load_faces(Path(os.environ["E031_DATA"]) / area))
    return _T[area]


def work(job):
    t = tiles(job["area"])[job["tile"]]
    lv = job["level"]
    best = json.loads(Path(os.environ["E031B_BEST"]).read_text())[f"L{lv}"]["H"]["params"]
    a = M.corrupt43(t, lv, job["seed"])
    faces, _ = E.decode_hybrid(a, best)
    rows = []
    for arm in ("fit", "robust", "slope"):
        P = M.planes(faces, a, arm)
        m = E.metrics(t, faces, P)
        for join in ("J1", "J2"):
            rings, st = M.mesh(faces, P, join)
            mm = M.mesh_metrics(t, faces, rings, P)
            rows.append(dict(area=t.area, tile=t.index, level=lv, seed=job["seed"], arm=arm, join=join,
                             h_rmse=m["h_rmse"], normal_err_deg=m["normal_err_deg"], face_f1=m["face_f1"], **st, **mm))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["test", "decide"])
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    RUNS.mkdir(exist_ok=True)
    sfx = "" if args.limit is None else f"_limit{args.limit}"
    if args.phase == "test":
        jobs = [dict(area=a, tile=i, level=lv, seed=s) for a in TEST_AREAS
                for i in range(min(len(tiles(a)), args.limit or 10 ** 6)) for lv in range(4) for s in SEEDS]
        t0 = time.time()
        with open(RUNS / f"test{sfx}.jsonl", "w") as f, Pool(args.procs, maxtasksperchild=8) as pool:
            for k, rows in enumerate(pool.imap_unordered(work, jobs), 1):
                for r in rows:
                    f.write(json.dumps(r, default=float) + "\n")
                if k % 50 == 0 or k == len(jobs):
                    print(f"{k}/{len(jobs)} {(time.time() - t0) / 60:.1f} min", flush=True)
        return
    rows = [json.loads(l) for l in open(RUNS / f"test{sfx}.jsonl")]
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k in ("h_rmse", "normal_err_deg", "vertex_z_err", "false_steps", "steps", "mean_gap"):
            if r[k] == r[k]:  # not NaN
                acc[(r["arm"], r["join"], r["level"])][k].append(r[k])
    out = {f"{a}_{j}_L{lv}": {k: float(np.mean(v)) for k, v in d.items()} for (a, j, lv), d in sorted(acc.items())}
    for k, v in out.items():
        print(k, {a: round(b, 3) for a, b in v.items()})
    g = lambda a, j, lv, k: out[f"{a}_{j}_L{lv}"][k]  # noqa: E731
    lv_hits = [lv for lv in (1, 2, 3) if g("slope", "J1", lv, "normal_err_deg") <= g("robust", "J1", lv, "normal_err_deg") - 1.0
               and g("slope", "J1", lv, "h_rmse") <= g("robust", "J1", lv, "h_rmse") - 0.05]
    dec = {"slope_rasters_worth_it": len(lv_hits) >= 2, "slope_levels": lv_hits,
           "robust_beats_fit": all(g("robust", "J1", lv, "h_rmse") <= g("fit", "J1", lv, "h_rmse") for lv in (1, 2, 3)),
           "J2_better": all(g(a, "J2", lv, "vertex_z_err") <= g(a, "J1", lv, "vertex_z_err") - 0.05
                            and g(a, "J2", lv, "false_steps") <= g(a, "J1", lv, "false_steps") for a in ("robust",) for lv in (1, 2))}
    out["decision"] = dec
    print(json.dumps(dec, indent=1))
    (RUNS / f"decision{sfx}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
