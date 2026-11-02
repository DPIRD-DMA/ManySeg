"""E011c: per-seed table and the pre-registered decision (README, Decision rule)."""
import glob
import json
import statistics as st
from pathlib import Path

runs = Path(__file__).parent / "runs"
by = {}
for f in sorted(glob.glob(str(runs / "*.json"))):
    r = json.load(open(f))
    by.setdefault(r["tag"], {})[r["seed"]] = r

print("| arm | seed | val MAE | val RMSE | val bias | test MAE | test RMSE | test bias | "
      "best-epoch val MAE | train time (min) |")
print("|---|---|---|---|---|---|---|---|---|---|")
agg = {}
for arm in ("density", "density_softplus"):
    for s, r in sorted(by.get(arm, {}).items()):
        v, t = r["final_val"], r["test"]
        t = t.get("test", t) if isinstance(t, dict) and "mae" not in t else t
        nonfinite = sum(h.get("nonfinite_steps", 0) for h in r["history"])
        print(f"| {arm} | {s} | {v['mae']:.2f} | {v['rmse']:.2f} | {v['bias']:+.2f} | "
              f"{t['mae']:.2f} | {t['rmse']:.2f} | {t['bias']:+.2f} | {r['best_val_mae']:.2f} | "
              f"{r['train_time_s'] / 60:.1f} |" + (f" nonfinite {nonfinite}" if nonfinite else ""))
        agg.setdefault(arm, {"val": [], "test": []})
        agg[arm]["val"].append(v["mae"])
        agg[arm]["test"].append(t["mae"])
for arm, d in agg.items():
    print(f"{arm}: val {st.mean(d['val']):.2f} ± {st.stdev(d['val']):.2f}, "
          f"test {st.mean(d['test']):.2f} ± {st.stdev(d['test']):.2f} (n={len(d['val'])})")


def verdict(split):
    a, b = agg["density"][split], agg["density_softplus"][split]
    delta = st.mean(b) - st.mean(a)
    pooled = ((st.stdev(a) ** 2 + st.stdev(b) ** 2) / 2) ** 0.5
    m = max(2 * pooled, 0.025 * st.mean(a))
    print(f"{split}: delta {delta:+.2f}, pooled SD {pooled:.2f}, margin {m:.2f} "
          f"(2 SD {2 * pooled:.2f}, 2.5% {0.025 * st.mean(a):.2f}) -> "
          f"{'within' if delta <= m else 'WORSE'}")
    return delta <= m


if len(agg.get("density", {}).get("val", [])) == 3 == len(agg.get("density_softplus", {})
                                                             .get("val", [])):
    ok = verdict("val") & verdict("test")
    print("DECISION:", "PASS (softplus becomes the default)" if ok else "FAIL (default stays raw)")
