"""torch.compile(dynamic=True) on CUDA miscomputes nearest-neighbour upsampling.

Prints, per input / output size, whether the compiled result equals eager. Expected: all
True. Observed (torch 2.14.1+cu130, RTX 4090): False for 49x66 -> 98x132 and 41x33 -> 81x66
after compiling at 35x38 -> 70x76. CPU is correct, as are dynamic=False, mode="nearest-exact",
scale_factor=2, and integer index arithmetic.
"""
import torch
import torch.nn.functional as F


def interp(x, ref):
    return F.interpolate(x, size=ref.shape[-2:], mode="nearest")


def float_index(x, ref):  # the same rule written out: also miscompiled
    h, w = x.shape[-2:]
    H, W = ref.shape[-2:]
    rows = torch.floor(torch.arange(H, device=x.device, dtype=torch.float32) * (h / H)).long()
    cols = torch.floor(torch.arange(W, device=x.device, dtype=torch.float32) * (w / W)).long()
    return x.index_select(-2, rows.clamp(max=h - 1)).index_select(-1, cols.clamp(max=w - 1))


for fn in (interp, float_index):
    torch._dynamo.reset()
    compiled = torch.compile(fn, dynamic=True)
    for (h, w), (H, W) in [((35, 38), (70, 76)), ((49, 66), (98, 132)), ((41, 33), (81, 66))]:
        x = torch.randn(2, 3, h, w, device="cuda")
        ref = torch.empty(1, 1, H, W, device="cuda")
        print(fn.__name__, (h, w), "->", (H, W), torch.equal(compiled(x, ref), fn(x, ref)))
print(torch.__version__)
