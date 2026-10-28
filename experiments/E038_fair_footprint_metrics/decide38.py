"""E038 decision (README): seed-0 screen, then the two-seed rule.  python decide38.py runs/e038_s0.json [runs/e038_s1.json]"""
import json
import sys

res = [json.load(open(p)) for p in sys.argv[1:]]
adv = []
for r in res:
    t = {a: r[a]["test"] for a in ("B0", "B0reg", "B3")}
    base = max(("B0", "B0reg"), key=lambda a: t[a]["boundary_f"])
    d = t["B3"]["boundary_f"] - t[base]["boundary_f"]
    adv.append(d)
    print(f"seed {r['seed']}: B3 {t['B3']['boundary_f']:.4f} vs better baseline {base} {t[base]['boundary_f']:.4f} "
          f"-> {d:+.4f}; det F1 B3 - B0 {t['B3']['det_f1'] - t['B0']['det_f1']:+.4f}")

r0 = next(r for r in res if r["seed"] == 0)
t0 = {a: r0[a]["test"] for a in ("B0", "B0reg", "B3")}
screen = adv[0] >= 0.01 and t0["B3"]["det_f1"] >= t0["B0"]["det_f1"] - 0.03
print("seed-0 screen:", "ADVANCE" if screen else "STOP (B0 + reg stays the default)")
if screen and len(res) >= 2:
    ok = all(a >= 0.01 for a in adv)
    print(f"two-seed rule: advantages {[round(a, 4) for a in adv]}, mean {sum(adv) / len(adv):+.4f} ->",
          "B3 PREFERRED" if ok else "B0 + reg stays the default")
