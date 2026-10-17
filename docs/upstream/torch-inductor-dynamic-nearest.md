# Draft upstream issue (PyTorch) — not posted

**Title:** Inductor (CUDA, `dynamic=True`) silently miscomputes nearest-neighbour upsampling
for some output sizes

**Environment:** torch 2.14.1+cu130, Linux, RTX 4090, Python 3.12.

**Reproduction:** [`repro_inductor_dynamic_nearest.py`](repro_inductor_dynamic_nearest.py)

```
interp (35, 38) -> (70, 76) True
interp (49, 66) -> (98, 132) False
interp (41, 33) -> (81, 66) False
float_index (35, 38) -> (70, 76) True
float_index (49, 66) -> (98, 132) False
float_index (41, 33) -> (81, 66) False
```

**What happens**

`torch.compile(fn, dynamic=True)` of `F.interpolate(x, size=ref.shape[-2:], mode="nearest")`
returns wrong values (relative L2 error about 1.0, so not rounding) for some input / output
sizes after compiling at another size, even for an exact 2x upsample (49x66 -> 98x132). The
same rule written out with float indices (`floor(arange(H_out) * (H_in / H_out))` then
`index_select`) is miscompiled the same way, so the problem seems to be in how Inductor
lowers that float index computation with symbolic sizes, not in the upsample op itself.

Correct in the same setting: CPU; `dynamic=False`; `mode="nearest-exact"`;
`scale_factor=2`; bilinear; and the integer form `arange(H_out) * H_in // H_out`.

**Impact:** silent wrong outputs in compiled segmentation decoders (U-Net / FPN style top-down
upsampling to the skip connection's size) when inputs are not padded to the network stride.

**Workaround (ManySeg):** `manyseg.utils.spatial.resize_nearest` gathers with the integer
index (`floor(i * in / out)`), which compiles correctly. It equals `mode="nearest"` whenever
the output size is a multiple of the input size; at other sizes PyTorch's float32 rounding
can select the previous row.
