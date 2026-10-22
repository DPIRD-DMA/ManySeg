"""E026: check that each patch gives the values the current code gives.

For each (patch, model): build the model, run forward + backward with the current code, apply
the patch, rebuild with the same weights, run again on the same input, compare.

* ``cpu64``: float64 on CPU, odd input size. The patch must agree to ~1e-9 (relative L2): identical up to float rounding.
* ``cuda_bf16``: bf16 autocast on CUDA. The patch's difference from the current code is
  reported next to the current code's own bf16-vs-fp32 error; it must be no larger.

One patch per process (patches are global), one JSON line per check:

    uv run python experiments/E026_perf_audit/check.py --patch segformer_fuse
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import models  # noqa: E402
import patches  # noqa: E402

TARGETS = {
    "tail_dtype": ["fastai_r34", "unet_r34_cross"],
    "fastai_layout": ["fastai_r34"],
    "cross_exact": ["unet_r34_cross"],
    "segformer_fuse": ["segformer_r34", "segformer_pvt"],
    "interp_dtype": ["upernet_r34", "deeplab_r34", "fpn_r34", "segformer_r34"],
    "gn_dtype": ["unet_r34_gn", "upernet_r34"],
    "split_conv": ["unet_r34", "unetpp_r34"],
    "pad_fast": ["unet_r34"],
    "mean_agg": ["unet_r34_ts"],
    "xattn_contig": ["xattn_r18"],
    "ln_dtype": ["unet_convnext", "unet_pvt", "unet_edgenext"],
}


def run(model, x, y, autocast_dtype=None, channels_last=False):
    model.zero_grad(set_to_none=True)
    dev = next(model.parameters()).device.type
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
        x = x.contiguous(memory_format=torch.channels_last) if isinstance(x, torch.Tensor) else x
    torch.manual_seed(123)  # same dropout masks in every run
    ctx = (torch.autocast(dev, dtype=autocast_dtype) if autocast_dtype
           else torch.autocast(dev, enabled=False))
    with ctx:
        out = model(x)
    loss = F.cross_entropy(out.float(), y)
    loss.backward()
    grads = torch.cat([p.grad.flatten().double() for p in model.parameters()
                       if p.grad is not None])
    return out.detach().double(), grads


def rel(a, b):
    """Relative L2 error (robust to the run-to-run noise of atomic-add backward kernels)."""
    return float((a - b).norm() / b.norm().clamp(min=1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", required=True, choices=sorted(TARGETS))
    ap.add_argument("--cpu-size", type=int, default=67)
    ap.add_argument("--cuda-size", type=int, default=509)
    ap.add_argument("--no-cuda", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(0)
    cases = []
    for name in TARGETS[a.patch]:
        cl = a.patch == "fastai_layout"
        for kind in ("cpu64",) + (() if a.no_cuda else ("cuda_bf16",)):
            dev = "cpu" if kind == "cpu64" else "cuda"
            size = a.cpu_size if kind == "cpu64" else a.cuda_size
            batch = 2
            torch.manual_seed(1)
            m = models.build(name, 3, 4).to(dev).train()
            if kind == "cpu64":
                m = m.double()
            g = torch.Generator().manual_seed(2)
            x = models.make_input(name, batch, size, 3, dev,
                                  torch.float64 if kind == "cpu64" else torch.float32, 3, g)
            y = torch.randint(0, 4, (batch, size, size), generator=g).to(dev)
            cases.append((name, kind, m, x, y, cl))
    # current code
    ref, state = {}, {}
    for name, kind, m, x, y, cl in cases:
        state[(name, kind)] = {k: v.clone() for k, v in m.state_dict().items()}
        if kind == "cpu64":
            ref[(name, kind)] = run(m, x, y)
        else:
            ref[(name, kind)] = run(m, x, y, torch.bfloat16, cl)
            m.load_state_dict(state[(name, kind)])  # undo BN running-stat updates
            ref[(name, kind, "fp32")] = run(m, x, y, None, cl)
    patches.apply([a.patch])
    for name, kind, m, x, y, cl in cases:
        torch.manual_seed(1)
        m2 = models.build(name, 3, 4).to(x.device if isinstance(x, torch.Tensor)
                                         else next(iter(x.values())).device).train()
        if kind == "cpu64":
            m2 = m2.double()
        m2.load_state_dict(state[(name, kind)])
        out_r, g_r = ref[(name, kind)]
        if kind == "cpu64":
            out_p, g_p = run(m2, x, y)
            rec = {"out_rel": rel(out_p, out_r), "grad_rel": rel(g_p, g_r)}
            rec["pass"] = rec["out_rel"] < 1e-9 and rec["grad_rel"] < 1e-9
        else:
            out_p, g_p = run(m2, x, y, torch.bfloat16, cl)
            out_32, g_32 = ref[(name, kind, "fp32")]
            rec = {"out_rel_vs_current": rel(out_p, out_r), "grad_rel_vs_current": rel(g_p, g_r),
                   "current_bf16_err": rel(out_r, out_32), "patched_bf16_err": rel(out_p, out_32),
                   "current_bf16_grad_err": rel(g_r, g_32),
                   "patched_bf16_grad_err": rel(g_p, g_32)}
            # the patched model must be no further from fp32 than the current one (10% slack)
            rec["pass"] = (rec["patched_bf16_err"] <= 1.1 * rec["current_bf16_err"] + 1e-6 and
                           rec["patched_bf16_grad_err"] <= 1.1 * rec["current_bf16_grad_err"]
                           + 1e-6)
        print(json.dumps({"patch": a.patch, "model": name, "kind": kind, **rec}), flush=True)


if __name__ == "__main__":
    main()
