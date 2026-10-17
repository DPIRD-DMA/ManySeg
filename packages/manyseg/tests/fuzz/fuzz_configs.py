"""Random ``create_model`` configurations and the invariants every built model must satisfy.

``sample_config(rng)`` draws a plausible configuration (backbones, single / multi input with
every fusion type and op, alignments, time series, vector inputs in every conditioning mode,
decoders and out_strides, pixel / image / per-input / shared / task-decoder heads, scales).
``check(cfg, seed)`` builds it and checks:

* the build succeeds or is rejected with ``ValueError`` / ``NotImplementedError`` (a clear
  message, never an internal crash);
* training forward at odd sizes gives the expected keys and shapes, finite values and finite
  gradients;
* eval is deterministic and per sample (a batch of two equals each sample alone);
* ``describe()``, a state-dict round trip into a freshly built twin, ``select_heads``,
  ``features`` / ``embed``, ``channels_last``;
* optionally ``torch.export`` with a dynamic shape, evaluated at another size.

Used by ``test_fuzz.py`` (a fixed handful of seeds) and runnable as a script for thousands.
"""

from __future__ import annotations

import copy
import math
import random
import traceback

import torch
from manyseg import create_model

BACKBONES = ["resnet18", "mobilenetv3_small_100", "convnext_atto", "pvt_v2_b0",
             "efficientnet_b0", "vit_tiny_patch16_224"]
DECODERS = ["unet", "fpn", "deeplabv3plus", "segformer", "upernet", "unetplusplus",
            "fastai_unet"]
OPS = ["concat", "add", "gated", "attention", "cross_attention", "diff", "concat_diff"]
ADAPTERS = ["repeat", "projection", "mean", "rgb_plus_random", "rgb_plus_zero"]
FIELDS = {"cont": {"type": "continuous", "mean": 50.0, "std": 20.0},
          "doy": {"type": "cyclic", "period": 365}, "cat": {"type": "categorical", "num": 4},
          "ll": {"type": "coords"}}


class Rejected(Exception):
    """The configuration was refused with a clear error (allowed)."""


def _maybe(rng, p):
    return rng.random() < p


def _indices(rng):
    """Feature-level indices, now and then malformed (floats, bools, empty, out of range)."""
    if _maybe(rng, 0.15):
        return rng.choice([[1.5, 2], [True, 3], [], [0, 9], [-1, 4]])
    return sorted(rng.sample(range(5), rng.randint(1, 4)))


def sample_config(rng: random.Random) -> dict:
    cfg: dict = {"backbone": rng.choice(BACKBONES), "pretrained": False}
    multi = _maybe(rng, 0.55)
    if multi:
        inputs: dict = {}
        n_raster = rng.choice([1, 2, 2, 3])
        for i in range(n_raster):
            name = f"r{i}"
            c = {"channels": rng.choice([1, 2, 3, 4, 13])}
            if i and _maybe(rng, 0.35):
                c["alignment"] = rng.choice(["same_extent", "same_grid"])
            if _maybe(rng, 0.2):
                c["kind"] = "timeseries"
                if _maybe(rng, 0.7):
                    c["temporal"] = rng.choice(["mean", "max", "attention", "concat", "diff"])
                    if c["temporal"] == "concat" and _maybe(rng, 0.8):
                        c["temporal"] = {"type": "concat", "dates": rng.choice([2, 3])}
            if _maybe(rng, 0.15):
                c["channel_adapter"] = rng.choice(ADAPTERS)
            if _maybe(rng, 0.1):
                c["backbone"] = rng.choice(BACKBONES)
            elif i and _maybe(rng, 0.1):
                c["backbone"] = "shared:r0"
            if _maybe(rng, 0.08):
                c["out_indices"] = _indices(rng)
            if c.get("backbone") == "shared:r0" and _maybe(rng, 0.5):
                c["share_norm"] = rng.choice([True, False])
            inputs[name] = c
        if _maybe(rng, 0.35):
            mode = rng.choice(["film", "head", "channels", {"mode": "head", "pixel": True},
                               {"mode": "film", "on": rng.choice([["features"], ["output"],
                                                                  ["features", "output"]])},
                               {"mode": "channels", "target": rng.choice(sorted(inputs))}])
            v = {"kind": "vector", "conditioning": mode}
            if _maybe(rng, 0.3):
                v["channels"] = rng.choice([1, 3])
            else:
                v["fields"] = dict(rng.sample(sorted(FIELDS.items()), rng.randint(1, 4)))
            if _maybe(rng, 0.2):
                v["dim"] = rng.choice([8, 64])
            inputs["meta"] = v
        cfg["inputs"] = inputs
        if n_raster > 1 or _maybe(rng, 0.2):
            ftype = rng.choice(["early", "intermediate", "intermediate", "late"])
            f: dict = {"type": ftype}
            if ftype != "early" and _maybe(rng, 0.7):
                f["op"] = rng.choice(OPS)
                if f["op"] == "cross_attention" and _maybe(rng, 0.3):
                    f["op_kwargs"] = {"heads": rng.choice([1, 2, 8, 0, 1.5, True]),
                                      "dim": rng.choice([4, 32, 0, -8])}
            cfg["fusion"] = f
    else:
        cfg["in_channels"] = rng.choice([1, 2, 3, 4, 13])
        if _maybe(rng, 0.08):
            cfg["out_indices"] = _indices(rng)
        if _maybe(rng, 0.3):
            cfg["channel_adapter"] = rng.choice(ADAPTERS)
    dec = rng.choice(DECODERS)
    cfg["decoder"] = dec
    if _maybe(rng, 0.3):
        dk: dict = {}
        if _maybe(rng, 0.6):
            dk["out_stride"] = rng.choice([1, 2, 4])
        if dec == "unet" and _maybe(rng, 0.4):
            dk["input_cross"] = True
        if dec in ("unet", "unetplusplus") and _maybe(rng, 0.2):
            dk["norm"] = rng.choice(["batch", "group", None])
        cfg["decoder_kwargs"] = dk
    if _maybe(rng, 0.25):
        cfg["pad_to_stride"] = False
    if _maybe(rng, 0.15):
        cfg["num_classes"] = rng.choice([1, 2, 5])
        return cfg
    heads: dict = {}
    for j in range(rng.randint(1, 4)):
        name = f"h{j}"
        scope = "image" if _maybe(rng, 0.25) else "pixel"
        t = rng.choice(["segmentation", "segmentation", "regression", "classification"])
        h: dict = {"type": t, "channels": rng.choice([1, 2, 5])}
        if scope == "image":
            h["scope"] = "image"
            if _maybe(rng, 0.2):
                h["source"] = "decoder"
            if _maybe(rng, 0.2):
                h["dropout"] = 0.1
        else:
            if _maybe(rng, 0.2):
                h["scale"] = rng.choice([1, 0.5, 0.25])
            if _maybe(rng, 0.15):
                h["source"] = "encoder"
            if _maybe(rng, 0.2):
                h["decoder"] = rng.choice([rng.choice(DECODERS),
                                           {"type": rng.choice(DECODERS),
                                            "out_stride": rng.choice([1, 2, 4])}])
        if t == "regression" and _maybe(rng, 0.3):
            h["uncertainty"] = True
        heads[name] = h
    rasters = [n for n, c in cfg.get("inputs", {}).items() if c.get("kind") != "vector"]
    if len(rasters) > 1 and _maybe(rng, 0.4):  # per-input (semantic change style) heads
        k = rng.choice([1, 2])
        base = {"type": "segmentation", "channels": rng.choice([2, 4])}
        for i, r in enumerate(rng.sample(rasters, k)):
            hc = dict(base, input=r)
            if i and _maybe(rng, 0.6):
                hc["share"] = "p0"
            heads[f"p{i}"] = hc
    cfg["heads"] = heads
    return cfg


# ---------------------------------------------------------------------------------- inputs

def make_inputs(cfg: dict, rng: random.Random, batch: int, hw=None,
                like=None) -> dict | torch.Tensor:
    """Random inputs; ``like``: another input dict whose keys (masks / dates) and date counts
    to reuse, with new values and sizes."""
    g = torch.Generator().manual_seed(rng.randrange(2**31))
    fusion = cfg.get("fusion") or {}
    # late fusion fuses full-resolution decoder outputs: cross-attention there is quadratic in
    # pixels (documented), so keep those inputs small
    top = 48 if fusion.get("type") == "late" and fusion.get("op") == "cross_attention" else 97
    h, w = hw or (rng.randint(33, top), rng.randint(33, top))
    if "inputs" not in cfg:
        return torch.randn(batch, cfg.get("in_channels", 3), h, w, generator=g)
    x = {}
    for n, c in cfg["inputs"].items():
        if c.get("kind") == "vector":
            if "channels" in c:
                x[n] = torch.randn(batch, c["channels"], generator=g)
            else:
                cols = []
                for spec in c["fields"].values():
                    if spec["type"] == "categorical":
                        cols.append(torch.randint(0, spec["num"], (batch, 1), generator=g))
                    elif spec["type"] == "coords":
                        cols.append(torch.rand(batch, 2, generator=g) * 90)
                    else:
                        cols.append(torch.rand(batch, 1, generator=g) * 100)
                x[n] = torch.cat([col.float() for col in cols], 1)
            continue
        hh, ww = (h, w)
        if c.get("alignment") == "same_extent":
            hh, ww = max(17, h // 2 + 1), max(17, w // 2 - 1)
        if c.get("kind") == "timeseries":
            agg = c.get("temporal", "mean")
            t = (2 if agg == "diff" else agg["dates"] if isinstance(agg, dict)
                 else rng.randint(2, 4))
            if like is not None and (agg == "diff" or isinstance(agg, dict)):
                t = like[n].shape[1]  # fixed date count; others get a new T
            x[n] = torch.randn(batch, t, c["channels"], hh, ww, generator=g)
            if like is not None:
                if f"{n}_mask" in like:
                    m = torch.rand(batch, t, generator=g) > 0.3
                    m[:, 0] = True
                    x[f"{n}_mask"] = m
                if f"{n}_dates" in like:
                    x[f"{n}_dates"] = torch.rand(batch, t, generator=g) * 365
                continue
            if _maybe(rng, 0.5):
                m = torch.rand(batch, t, generator=g) > 0.3
                m[:, 0] = True
                x[f"{n}_mask"] = m
            if c.get("temporal") == "attention" or _maybe(rng, 0.3):
                x[f"{n}_dates"] = torch.rand(batch, t, generator=g) * 365
        else:
            x[n] = torch.randn(batch, c["channels"], hh, ww, generator=g)
    return x


def _primary_hw(cfg, x):
    if isinstance(x, torch.Tensor):
        return x.shape[-2:]
    first = next(n for n, c in cfg["inputs"].items() if c.get("kind") != "vector")
    return x[first].shape[-2:]


def _index(x, i):
    if isinstance(x, torch.Tensor):
        return x[i:i + 1]
    return {k: _index(v, i) for k, v in x.items()}


def _cast(x, fn):
    if isinstance(x, torch.Tensor):
        return fn(x)
    return {k: _cast(v, fn) for k, v in x.items()}


def _outs(y) -> dict:
    return y if isinstance(y, dict) else {"logits": y}


# ------------------------------------------------------------------------------- the checks

def build(cfg: dict, seed: int):
    torch.manual_seed(seed)
    try:
        return create_model(**copy.deepcopy(cfg))
    except (ValueError, NotImplementedError) as e:
        raise Rejected(f"{type(e).__name__}: {e}") from e


def expected_shapes(cfg, model, batch, hw) -> dict:
    heads = cfg.get("heads") or {"logits": {"type": "segmentation",
                                            "channels": cfg["num_classes"]}}
    out = {}
    for n, h in heads.items():
        if n not in model.heads:
            continue
        ch = h["channels"] * (2 if h.get("uncertainty") else 1)
        if h.get("scope") == "image":
            out[n] = (batch, ch)
        else:
            f = round(1 / h.get("scale", 1))
            out[n] = (batch, ch, math.ceil(hw[0] / f), math.ceil(hw[1] / f))
    return out


def close(a, b, tol=1e-4):
    a, b = _outs(a), _outs(b)
    assert a.keys() == b.keys(), (a.keys(), b.keys())
    for k in a:
        torch.testing.assert_close(a[k], b[k], atol=tol, rtol=tol, msg=lambda m, k=k: f"{k}: {m}")


def check(cfg: dict, seed: int, export: bool = False) -> str:
    """Raises ``Rejected`` for a refused config, ``AssertionError`` / anything else for a bug;
    returns "ok"."""
    rng = random.Random(seed)
    model = build(cfg, seed)
    model.train()
    x = make_inputs(cfg, rng, batch=2)
    hw = tuple(_primary_hw(cfg, x))
    y = _outs(model(x))
    want = expected_shapes(cfg, model, 2, hw)
    assert {k: tuple(v.shape) for k, v in y.items()} == want, ("train shapes", hw)
    for k, v in y.items():
        assert torch.isfinite(v).all(), f"non-finite train output {k}"
    sum(v.float().square().mean() for v in y.values()).backward()
    unused = []
    for n, p in model.named_parameters():
        if p.grad is None:
            if p.requires_grad:
                unused.append(n)
        else:
            assert torch.isfinite(p.grad).all(), f"non-finite grad {n}"
    assert not unused, f"parameters outside the graph: {unused[:6]}"

    model.eval()
    with torch.no_grad():
        e1 = model(x)
        close(e1, model(x), 0)
        per = [_outs(model(_index(x, i))) for i in range(2)]
        for k, v in _outs(e1).items():
            torch.testing.assert_close(v, torch.cat([p[k] for p in per]), atol=1e-4,
                                       rtol=1e-4, msg=lambda m, k=k: f"batch vs single {k}: {m}")
        assert isinstance(model.describe(), str)

        twin = build(cfg, seed + 1).eval()
        twin.load_state_dict(model.state_dict(), strict=True)
        close(e1, twin(x))

        feats = model.features(x)
        assert all(torch.isfinite(f).all() for f in feats if f is not None)  # None: unread
        emb = model.embed(x, pool="mean")
        assert emb.ndim == 2 and emb.shape[0] == 2
        if model.decoder is not None:
            st = model.decoder.out_spec.stride
            assert model.embed(x, source="decoder").shape[-2:] == \
                (math.ceil(hw[0] / st), math.ceil(hw[1] / st)), "decoder embed grid"

        cl = copy.deepcopy(model).to(memory_format=torch.channels_last)
        xcl = _cast(x, lambda t: t.contiguous(memory_format=torch.channels_last)
                    if t.ndim == 4 else t)
        close(e1, cl(xcl), 1e-3)

        names = list(_outs(e1))
        if len(names) > 1:
            keep = names[: max(1, len(names) // 2)]
            sub = copy.deepcopy(model).select_heads(keep)
            ys = _outs(sub(x))
            assert sorted(ys) == sorted(keep)
            for k in keep:
                torch.testing.assert_close(ys[k], _outs(e1)[k], atol=1e-5, rtol=1e-5)

    if export:
        from manyseg.export import _min_size, export_model

        # sizes inside the exported range, also for half-size same_extent inputs
        lo = _min_size(model)
        halves = any(c.get("alignment") == "same_extent" for c in cfg.get("inputs", {}).values())
        lo2 = 2 * lo + 4 if halves else lo  # half-size same_extent inputs stay in range
        r2 = random.Random(seed + 7)
        x = make_inputs(cfg, r2, batch=2, hw=(r2.randint(lo2, lo2 + 40),) * 2)
        ep = export_model(model, x)
        # the second sample sits at the bottom of the exported range half the time
        hw2 = (lo2, lo2 + 1) if r2.random() < 0.5 else (r2.randint(lo2, lo2 + 60),
                                                         r2.randint(lo2, lo2 + 60))
        x2 = make_inputs(cfg, r2, batch=2, hw=hw2, like=x if isinstance(x, dict) else None)
        with torch.no_grad():
            close(model(x2), ep.module()(x2), 1e-3)
    return "ok"


def run(n: int, start: int = 0, export_every: int = 0, verbose: bool = False) -> dict:
    stats = {"ok": 0, "rejected": 0, "fail": []}
    for s in range(start, start + n):
        cfg = sample_config(random.Random(s))
        try:
            check(cfg, s, export=bool(export_every) and s % export_every == 0)
            stats["ok"] += 1
        except Rejected as e:
            stats["rejected"] += 1
            if verbose:
                print(f"[{s}] rejected: {e}")
        except Exception as e:  # noqa: BLE001 - every other failure is a finding
            stats["fail"].append((s, cfg, f"{type(e).__name__}: {e}",
                                  traceback.format_exc(limit=-4)))
            print(f"[{s}] FAIL {type(e).__name__}: {str(e)[:300]}\n  {cfg}", flush=True)
    return stats


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=200)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--export-every", type=int, default=0)
    p.add_argument("--out")
    p.add_argument("-v", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(2)
    st = run(a.n, a.start, a.export_every, a.v)
    print(f"ok {st['ok']} rejected {st['rejected']} failed {len(st['fail'])}")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump([{"seed": s, "cfg": c, "error": e, "tb": tb} for s, c, e, tb in
                       st["fail"]], fh, indent=1, default=str)
