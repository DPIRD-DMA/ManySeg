"""E050b (README): tile size / overlap for roof faces; blend and midline vs the single pass.

    uv run python e050b.py [--procs 6]
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool

import numpy as np

import e050 as E5

SETTINGS = [(512, 128), (512, 256), (1024, 256), (1024, 512)]


def work(item):
    E5.torch.set_num_threads(1)
    key, f, ref_dec, zone = item
    return key, E5.agree("roofs", ref_dec, E5.decode("roofs", f), zone)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    args = ap.parse_args()
    out = E5.HERE / "runs" / "e050b.json"
    if out.exists():
        raise SystemExit(f"{out} exists")
    T = E5.Task("roofs")
    acc = {}
    with Pool(args.procs, maxtasksperchild=8) as pool:
        for scene in T.scenes("test"):
            x = T.x(scene)
            ref = T.fields(x)
            ref_dec = E5.decode("roofs", ref)
            items = []
            for p, o in SETTINGS:
                wins = list(dict.fromkeys(E5.make_patch_indexes(E5.S, E5.S, p, o)))
                cov = np.zeros((E5.S, E5.S), np.int32)
                for t, b, l, r in wins:
                    cov[t:b, l:r] += 1
                patches = [T.fields(x[:, :, t:b, l:r]) for t, b, l, r in wins]
                for name in ("blend", "midline"):
                    f, _ = (E5.blend if name == "blend" else E5.midline)(patches, wins, o)
                    items.append(((p, o, name, len(wins)), {k: v.astype(np.float16) for k, v in f.items()}, ref_dec, cov >= 2))
                del patches
            for (p, o, name, n), a in pool.imap_unordered(work, items):
                s = acc.setdefault(f"{p}|{o}|{name}", {"patches": n})
                for z, d in a.items():
                    s[z] = [u + v for u, v in zip(s.get(z, [0, 0, 0, 0]), d["face"])]
                print(scene, p, o, name, {z: round(E5.f1(d["face"]), 3) for z, d in a.items()}, flush=True)
    res = {k: {"patches_per_scene": v["patches"], **{z: {"f1": E5.f1(c), "counts": c} for z, c in v.items() if z != "patches"}}
           for k, v in acc.items()}
    base = {m: res[f"512|128|{m}"]["all"]["f1"] for m in ("blend", "midline")}
    dec = {}
    for m in ("blend", "midline"):
        gains = {f"{p}|{o}": res[f"{p}|{o}|{m}"]["all"]["f1"] - base[m] for p, o in SETTINGS[1:]}
        ok = [k for k, g in gains.items() if g >= 0.01]
        dec[m] = {"all_face_gain_vs_512_128": gains, "helps": bool(ok),
                  "smallest_helping": min(ok, key=lambda k: res[f"{k}|{m}"]["patches_per_scene"]) if ok else None}
    res["decision"] = dec
    out.write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        if k != "decision":
            print(k, v["patches_per_scene"], {z: round(v[z]["f1"], 4) for z in ("all", "ov", "single")})
    print(json.dumps(dec, indent=1))


if __name__ == "__main__":
    main()
