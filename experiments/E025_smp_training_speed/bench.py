"""E025: time one training step (forward + backward) of ManySeg's default U-Net against SMP's.

One configuration per process (fresh CUDA allocator and compile caches), one JSON line out:

    uv run python experiments/E025_smp_training_speed/bench.py --lib manyseg \
        --backbone edgenext_small --size 509 [--compile] [--profile trace_dir]

bf16 autocast, fp32 weights, batch 40, 3 channels, 4 classes, cross-entropy on random labels
(the step's cost does not depend on the values). Random init: weights do not change the cost.
Per-step time from CUDA events around forward + backward + zero_grad; median over ``--steps``
after ``--warmup`` steps.
"""

import argparse
import json
import statistics
import subprocess

import torch
import torch.nn.functional as F

SMP_ARCH = {"unet": "Unet", "unetplusplus": "UnetPlusPlus", "fpn": "FPN"}


def build(lib: str, backbone: str, decoder: str = "unet"):
    if lib == "manyseg":
        from manyseg import create_model

        return create_model(backbone=backbone, pretrained=False, in_channels=3, num_classes=4,
                            decoder=decoder)
    import segmentation_models_pytorch as smp

    return getattr(smp, SMP_ARCH[decoder])(encoder_name=f"tu-{backbone}", encoder_weights=None,
                                           in_channels=3, classes=4)


def gpu_state() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                               "--format=csv,noheader"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", choices=["manyseg", "smp"], required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--decoder", default="unet", help="unet | unetplusplus | fpn | fastai_unet")
    ap.add_argument("--size", type=int, default=509)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--profile", default=None, help="write a profiler table (and trace) here")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    dev = "cuda"
    model = build(a.lib, a.backbone, a.decoder).to(dev).train()
    mf = torch.channels_last if a.channels_last else torch.contiguous_format
    model = model.to(memory_format=mf)
    params = sum(p.numel() for p in model.parameters())
    step_model = torch.compile(model) if a.compile else model
    x = torch.randn(a.batch, 3, a.size, a.size, device=dev).to(memory_format=mf)
    y = torch.randint(0, 4, (a.batch, a.size, a.size), device=dev)

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = step_model(x)
            loss = F.cross_entropy(out.float(), y)
        loss.backward()
        model.zero_grad(set_to_none=True)

    warmup = a.warmup + (10 if a.compile else 0)
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(a.steps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        step()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    peak = torch.cuda.max_memory_allocated() / 2**30
    rec = {"lib": a.lib, "backbone": a.backbone, "decoder": a.decoder, "size": a.size,
           "batch": a.batch,
           "compile": a.compile, "channels_last": a.channels_last, "params_m": params / 1e6,
           "median_ms": statistics.median(times), "p10_ms": sorted(times)[len(times) // 10],
           "p90_ms": sorted(times)[len(times) * 9 // 10], "peak_gib": peak,
           "steps": a.steps, "gpu_before": a.tag, "gpu_after": gpu_state(),
           "torch": torch.__version__}
    if a.profile:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                step()
            torch.cuda.synchronize()
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=40)
        name = (f"{a.profile}/{a.lib}_{a.backbone}_{a.decoder}_{a.size}"
                f"{'_c' if a.compile else ''}.txt")
        with open(name, "w") as f:
            f.write(table)
        rec["profile"] = name
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
