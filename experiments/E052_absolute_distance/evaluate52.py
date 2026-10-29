"""E052 evaluation (README): N and A runs, seeds 0 and 1, tuned on FTW NL val, then test once.

    uv run python evaluate52.py [--limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "E030_whu_polygons"))
sys.path.insert(0, str(HERE.parent / "E046_ftw_trained"))
from common import b0_probabilities  # noqa: E402
from evaluate46 import run  # noqa: E402
from train52 import build  # noqa: E402

SEEDS = {"N": (0.15, 0.3), "A": (1.5, 3.0)}
SIMPLIFY = [dict(simplify=s) for s in (0.5, 1.0, 1.5, 2.0)]


def thresholds(arm):
    return [dict(extent_threshold=e, boundary_threshold=b, seed_threshold=s)
            for e, b, s in itertools.product((0.4, 0.6), (0.3, 0.5), SEEDS[arm])]


def decide(test: dict) -> dict:
    """README rule: mean over seeds of A - N on test."""
    d = {k: float(np.mean([test[f"A_s{s}"][k] - test[f"N_s{s}"][k] for s in (0, 1)]))
         for k in ("boundary_f", "det_f1", "ciou", "polis_px")}
    default = d["boundary_f"] >= -0.01 and d["det_f1"] >= -0.01
    return {"a_minus_n": d, "absolute_default": default,
            "verdict": "absolute becomes the default" if default else "absolute stays an option"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    out = HERE / "runs" / f"eval{'_limit' if args.limit else ''}.json"
    if out.exists():  # test once
        raise SystemExit(f"{out} exists: the test split was already evaluated")
    dev = torch.device("cuda")
    runs = [f"{a}_s{s}" for a in ("N", "A") for s in (0, 1)]
    res, t0 = {"limit": args.limit}, time.time()

    def load(name):
        m = build(name[0])
        p = HERE / "runs" / f"{name}.pt"
        m.load_state_dict(torch.load(p, map_location="cpu"))
        return m.to(dev).eval(), p

    for name in runs:  # tune all runs on val before any test evaluation
        m, p = load(name)
        thr_list = thresholds(name[0])
        v1 = run(m, b0_probabilities, "B0", "val", thr_list, [dict(simplify=1.0)], args.procs, args.limit)
        thr = thr_list[max(range(len(thr_list)), key=lambda j: v1[j]["boundary_f"])]
        v2 = run(m, b0_probabilities, "B0", "val", [thr], SIMPLIFY, args.procs, args.limit)
        k = max(range(len(v2)), key=lambda j: v2[j]["boundary_f"])
        res[name] = {"checkpoint": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                     "thr": thr, "params": SIMPLIFY[k], "val": v2[k],
                     "val_thr": [dict(thr_list[j], **v1[j]) for j in range(len(thr_list))],
                     "val_grid": [dict(SIMPLIFY[j], **v2[j]) for j in range(len(v2))]}
        print(name, "val", round(v2[k]["boundary_f"], 4), f"{time.time() - t0:.0f}s", flush=True)
        del m
        torch.cuda.empty_cache()
    for name in runs:
        m, _ = load(name)
        (res[name]["test"],) = run(m, b0_probabilities, "B0", "test", [res[name]["thr"]], [res[name]["params"]],
                                   args.procs, args.limit)
        del m
        torch.cuda.empty_cache()
    res["decision"] = decide({n: res[n]["test"] for n in runs})
    out.write_text(json.dumps(res, indent=1))
    for name in runs:
        t = res[name]["test"]
        print(name, "test boundary F", round(t["boundary_f"], 4), "det F1", round(t["det_f1"], 4))
    print(json.dumps(res["decision"], indent=1))


if __name__ == "__main__":
    main()
