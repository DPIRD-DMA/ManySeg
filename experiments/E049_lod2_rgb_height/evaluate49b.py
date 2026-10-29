"""E049b (README): H1b (slope head with targets x10) against seed-0 H0; val gate on the slope head,
then test once.

    uv run python evaluate49b.py --h1b runs/H1b_s0.pt [--procs 6] [--limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

import evaluate49 as V9
import train49 as T9

HERE = Path(__file__).resolve().parent
AREAS = V9.AREAS


@torch.no_grad()
def slope_gate(model, limit):
    """Pearson correlation of predicted vs target slope on val interior roof pixels, per channel."""
    ds = T9.RGBRoofTiles("val", augment=False, targets=True)
    mean, std = T9.MEAN.cuda(), T9.STD.cuda()
    pr, tg = [[], []], [[], []]
    for i in range(min(len(ds), limit or len(ds))):
        x, t = ds[i]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model((x[None].cuda().float() - mean) / std)
        ps = o["slope"][0].float().cpu().numpy()[:, 1:-1, 1:-1] / T9.SLOPE_SCALE["H1b"]
        st, m = T9.slope_target(t["height"][None], t["inst"][None])
        st, m = st[0].numpy(), m[0].numpy()
        for c in (0, 1):
            pr[c].append(ps[c][m]); tg[c].append(st[c][m])
    return [float(np.corrcoef(np.concatenate(tg[c]), np.concatenate(pr[c]))[0, 1]) for c in (0, 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h1b", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    sha = hashlib.sha256(args.h1b.read_bytes()).hexdigest()
    ledger = HERE / "runs" / "test_ledger.json"
    led = json.loads(ledger.read_text()) if ledger.exists() else {}
    out = HERE / "runs" / f"eval_{args.h1b.stem}{'_limit' if args.limit else ''}.json"
    if not args.limit and (out.exists() or sha in led):
        raise SystemExit(f"test already opened for {args.h1b} ({led.get(sha)})")
    model, seed = V9.load(args.h1b, "H1b")
    h0 = json.loads((HERE / "runs" / "eval_H0_s0_H1_s0.json").read_text())["test"]["H0"]
    res = {"seed": seed, "sha256": sha}
    t0 = time.time()
    res["slope_corr_val"] = corr = slope_gate(model, args.limit)
    print("val slope correlation (x, y)", [round(c, 3) for c in corr], flush=True)
    res["gate_passed"] = min(corr) >= 0.5
    if not res["gate_passed"] and not args.limit:  # smoke runs continue past the gate (val only)
        out.write_text(json.dumps(res, indent=1))
        raise SystemExit("slope head gate failed on val; test not opened")
    sel = V9.tune(model, args.procs, args.limit, kind="H1b")
    res["selected"] = sel
    split = "val" if args.limit else "test"
    if not args.limit:
        led[sha] = {"opened": time.strftime("%Y-%m-%d %H:%M"), "arm": "H1b", "out": str(out)}
        ledger.write_text(json.dumps(led, indent=1))
    allm, areas, hmae = V9.run(model, "H1b", split, [sel["thr"]], [sel["params"]], ["slope", "pred"], True, args.procs, args.limit)
    res["test"] = {k: {"all": allm[j], "areas": {a: v[j] for a, v in areas.items()}} for j, k in enumerate(("H1b", "H1br"))}
    res["height_mae"] = hmae
    for k in ("H1b", "H1br"):
        print(k, split, {a: {x: round(y, 3) for x, y in v.items() if isinstance(y, float)} for a, v in res["test"][k]["areas"].items()}, flush=True)
    if not args.limit:
        b = res["test"]["H1b"]["areas"]
        if set(b) != set(AREAS):
            raise SystemExit(f"test areas {sorted(b)} != {AREAS}")
        dn = {a: b[a]["normal_err_deg"] - h0["areas"][a]["normal_err_deg"] for a in AREAS}
        dr = {a: b[a]["h_rmse"] - h0["areas"][a]["h_rmse"] for a in AREAS}
        res["decision"] = {"h1b_minus_h0_normal_deg": dn, "h1b_minus_h0_rmse": dr,
                           "slope_head_worth_it": all(dn[a] <= -1.0 and dr[a] <= 0.05 for a in AREAS),
                           "second_seed_needed": min(abs(dn[a] + 1.0) for a in AREAS) < 0.25 or min(abs(dr[a] - 0.05) for a in AREAS) < 0.05}
        led[sha]["completed"] = time.strftime("%Y-%m-%d %H:%M")
        ledger.write_text(json.dumps(led, indent=1))
        print(json.dumps(res["decision"], indent=1))
    res["wall_s"] = time.time() - t0
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
