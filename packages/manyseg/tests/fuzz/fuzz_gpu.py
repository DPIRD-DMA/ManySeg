"""The random configurations of ``fuzz_configs`` on CUDA, where most training happens:

* CUDA fp32 (TF32 off) agrees with CPU fp32;
* a training step under bf16 autocast: finite outputs and gradients, every parameter in the
  graph;
* bf16 autocast inference stays close to fp32 (relative L2 error), also with channels_last;
* sampled: ``torch.compile(dynamic=True)`` at two sizes agrees with eager.

``check_gpu(cfg, seed)`` raises ``Rejected`` for a refused config, anything else for a bug.
Runnable as a script for campaigns; ``test_fuzz.py`` runs a few seeds when CUDA is present.
"""

from __future__ import annotations

import copy
import random
import traceback

import torch
import torch._dynamo
from fuzz_configs import (
    Rejected,
    _cast,
    _outs,
    _primary_hw,
    build,
    expected_shapes,
    make_inputs,
    sample_config,
)

BF16_REL = 0.1  # bf16 has an 8-bit mantissa: ~1% typical; 10% means something is wrong


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().cpu(), b.float().cpu()
    return float((a - b).norm() / (b.norm() + 1e-6))


def _cuda(x):
    return _cast(x, lambda t: t.cuda())


def check_gpu(cfg: dict, seed: int, compile_: bool = False) -> dict:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rng = random.Random(seed)
    model = build(cfg, seed)  # Rejected propagates
    x = make_inputs(cfg, rng, batch=2)
    hw = tuple(_primary_hw(cfg, x))
    stats = {}

    model.eval()
    with torch.no_grad():
        cpu = _outs(model(x))
    gpu_model = copy.deepcopy(model).cuda()
    xg = _cuda(x)
    with torch.no_grad():
        g32 = _outs(gpu_model(xg))
    assert g32.keys() == cpu.keys()
    for k in cpu:
        e = rel_err(g32[k], cpu[k])
        stats.setdefault("cuda_vs_cpu", []).append(e)
        assert e < 1e-3, f"{k}: CUDA fp32 vs CPU rel err {e:.2e}"

    # bf16 autocast inference vs fp32 (also channels_last)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        b16 = _outs(gpu_model(xg))
    cl = copy.deepcopy(gpu_model).to(memory_format=torch.channels_last)
    xcl = _cast(xg, lambda t: t.contiguous(memory_format=torch.channels_last)
                if t.ndim == 4 else t)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        b16cl = _outs(cl(xcl))
    for k in g32:
        assert torch.isfinite(b16[k]).all(), f"{k}: non-finite bf16 output"
        e, ecl = rel_err(b16[k], g32[k]), rel_err(b16cl[k], g32[k])
        stats.setdefault("bf16_vs_fp32", []).append(e)
        assert e < BF16_REL, f"{k}: bf16 vs fp32 rel err {e:.3f}"
        assert ecl < BF16_REL, f"{k}: bf16 channels_last vs fp32 rel err {ecl:.3f}"

    # a bf16 training step
    gpu_model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = _outs(gpu_model(xg))
    want = expected_shapes(cfg, gpu_model, 2, hw)
    assert {k: tuple(v.shape) for k, v in y.items()} == want, "bf16 train shapes"
    for k, v in y.items():
        assert torch.isfinite(v).all(), f"{k}: non-finite bf16 train output"
    sum(v.float().square().mean() for v in y.values()).backward()
    unused = [n for n, p in gpu_model.named_parameters() if p.requires_grad and p.grad is None]
    assert not unused, f"parameters outside the graph (bf16): {unused[:6]}"
    bad = [n for n, p in gpu_model.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite bf16 grads: {bad[:6]}"

    if compile_:
        torch._dynamo.reset()
        gpu_model.eval()
        comp = torch.compile(gpu_model, dynamic=True)
        for i, hw2 in enumerate([None, (rng.randint(65, 97), rng.randint(65, 97))]):
            xi = xg if i == 0 else _cuda(make_inputs(cfg, random.Random(seed + 11), batch=2,
                                                     hw=hw2, like=x if isinstance(x, dict)
                                                     else None))
            with torch.no_grad():
                ref, got = _outs(gpu_model(xi)), _outs(comp(xi))
            for k in ref:
                e = rel_err(got[k], ref[k])
                stats.setdefault("compile_vs_eager", []).append(e)
                assert e < 1e-3, f"{k}: compiled vs eager rel err {e:.2e} (size {i})"
        torch._dynamo.reset()
    return stats


def run(n: int, start: int = 0, compile_every: int = 0) -> dict:
    out = {"ok": 0, "rejected": 0, "fail": [], "stats": {}}
    for s in range(start, start + n):
        cfg = sample_config(random.Random(s))
        try:
            st = check_gpu(cfg, s, compile_=bool(compile_every) and s % compile_every == 0)
            out["ok"] += 1
            for k, v in st.items():
                out["stats"].setdefault(k, []).extend(v)
        except Rejected:
            out["rejected"] += 1
        except Exception as e:  # noqa: BLE001 - every other failure is a finding
            out["fail"].append((s, cfg, f"{type(e).__name__}: {e}",
                                traceback.format_exc(limit=-4)))
            print(f"[{s}] FAIL {type(e).__name__}: {str(e)[:300]}\n  {cfg}", flush=True)
        finally:
            torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=200)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--compile-every", type=int, default=0)
    p.add_argument("--out")
    a = p.parse_args()
    torch.set_num_threads(4)
    res = run(a.n, a.start, a.compile_every)
    for k, v in res["stats"].items():
        v = sorted(v)
        print(f"{k}: n {len(v)} median {v[len(v) // 2]:.2e} p99 {v[int(len(v) * .99)]:.2e} "
              f"max {v[-1]:.2e}")
    print(f"ok {res['ok']} rejected {res['rejected']} failed {len(res['fail'])}")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump([{"seed": s, "cfg": c, "error": e, "tb": tb} for s, c, e, tb in
                       res["fail"]], fh, indent=1, default=str)
