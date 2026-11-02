"""E053 (README): min_seed 0 vs 8 at the shipped decoder settings on trained checkpoints.

    uv run python e053.py --task {roofs,footprints,fields} [--limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RUNS = Path("$WORKSPACE/ManySeg-runs")
SEEDS = (0, 8)
CKPT = {
    "roofs": [RUNS / "e047b-54b73bc/experiments/E047_lod2_trained/runs/B3_s0_e150.pt"],
    "footprints": [RUNS / f"e036-f2d2f86/experiments/E030_whu_polygons/runs/B3_s{s}.pt" for s in (0, 1)],
    "fields": [EXP / f"E052_absolute_distance/runs/A_s{s}.pt" for s in (0, 1)],
}
THR = {"roofs": dict(extent_threshold=0.4, boundary_threshold=0.3, seed_threshold=0.15),
       "footprints": dict(extent_threshold=0.6, boundary_threshold=0.5, seed_threshold=0.15),
       "fields": dict(extent_threshold=0.4, boundary_threshold=0.3, seed_threshold=3.0)}
PARAMS = {"roofs": dict(corner_threshold=0.5, big_dev=4.0, agree=0.6, merge_deg=8.0),
          "footprints": dict(corner_threshold=0.5, big_dev=4.0, agree=0.4, merge_deg=8.0),
          "fields": dict(simplify=0.5)}
PRIMARY = {"roofs": ("face_f1", "face_f1"), "footprints": ("boundary_f", "det_f1"),
           "fields": ("boundary_f", "det_f1")}


def evaluate(task, ckpt, split, procs, limit):
    """[result at min_seed 0, result at min_seed 8] (+ per-area results for roofs)."""
    if task == "roofs":
        sys.path.insert(0, str(EXP / "E047_lod2_trained"))
        import evaluate47 as E

        m = E.build()
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        m = m.to("cuda").eval()
        allm, areas = E.run(m, "F3", split, [dict(THR[task], min_seed=s) for s in SEEDS], [PARAMS[task]],
                            procs, limit)
        return [dict(allm[i], areas={a: v[i] for a, v in areas.items()}) for i in range(len(SEEDS))]
    if task == "footprints":
        os.chdir(EXP / "E030_whu_polygons")  # evaluate38 imports common / train from the cwd
        sys.path.insert(0, str(EXP / "E038_fair_footprint_metrics"))
        import evaluate38 as E

        m = E.build("B3")
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        m = m.to("cuda").eval()
        out = []
        for s in SEEDS:  # labels() reads the module's thresholds (forked into the pool)
            E.B0_THR = dict(THR[task], min_seed=s)
            (r,) = E.run(m, E.b3_probabilities, "B3", split, [PARAMS[task]], {}, procs, limit)
            out.append(r)
        return out
    sys.path.insert(0, str(EXP / "E052_absolute_distance"))
    from evaluate52 import b0_probabilities, build, run

    m = build("A")
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m = m.to("cuda").eval()
    return run(m, b0_probabilities, "B0", split, [dict(THR[task], min_seed=s) for s in SEEDS], [PARAMS[task]],
               procs, limit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(CKPT), required=True)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    out = HERE / "runs" / f"e053_{args.task}{'_limit' if args.limit else ''}.json"
    if out.exists():
        raise SystemExit(f"{out} exists")
    res, t0 = {"task": args.task, "limit": args.limit, "thr": THR[args.task], "params": PARAMS[args.task]}, time.time()
    for ckpt in CKPT[args.task]:
        r = {"sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest()}
        for split in ("val", "test"):
            r[split] = dict(zip((f"min_seed_{s}" for s in SEEDS), evaluate(args.task, ckpt, split, args.procs,
                                                                          args.limit)))
            print(ckpt.name, split, {k: {m: round(v[m], 4) for m in set(PRIMARY[args.task]) | {"n_pred"}}
                                     for k, v in r[split].items()}, f"{time.time() - t0:.0f}s", flush=True)
        res[str(ckpt)] = r
    runs = [res[str(c)]["test"] for c in CKPT[args.task]]
    diff = {k: sum(t["min_seed_8"][k] - t["min_seed_0"][k] for t in runs) / len(runs)
            for k in set(PRIMARY[args.task]) | {"n_pred"}}
    p, d = PRIMARY[args.task]
    res["decision"] = {"min_seed_8_minus_0": diff, "adopt": diff[p] >= -0.005 and diff[d] >= -0.005}
    out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["decision"], indent=1))


if __name__ == "__main__":
    main()
