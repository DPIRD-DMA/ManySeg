## Patches (train step, bf16, current -> patched; Δ time = median of paired A/B runs)

| model | patch | mode | ms current | ms patched | Δ time | GiB current | GiB patched | Δ peak | pairs |
|---|---|---|---|---|---|---|---|---|---|
| fastai_r34 | tail_dtype | eager b16 512² | 265.3 | 237.9 | -10.3% | 11.08 | 8.86 | -20.0% | 2 |
| fastai_edgenext | tail_dtype | eager b16 512² | 187.8 | 163.8 | -12.8% | 9.79 | 7.81 | -20.3% | 2 |
| unet_r34_cross | tail_dtype | eager b16 512² | 105.8 | 99.9 | -5.6% | 4.48 | 4.18 | -6.6% | 2 |
| fastai_r34 | tail_dtype | compile b16 512² | 210.7 | 205.6 | -2.4% | 9.56 | 8.43 | -11.8% | 1 |
| unet_r34_cross | tail_dtype | compile b16 512² | 78.6 | 79.1 | +0.6% | 3.92 | 3.73 | -4.9% | 1 |
| unet_r34_cross | cross_exact | compile b16 512² | 85.3 | 84.9 | -0.4% | 3.92 | 3.92 | -0.0% | 1 |
| fastai_r34 | fastai_layout | eager CL b16 512² | 282.6 | 278.9 | -1.3% | 11.07 | 10.39 | -6.2% | 2 |
| segformer_r34 | segformer_fuse | eager b16 512² | 62.5 | 56.3 | -9.8% | 4.20 | 2.62 | -37.7% | 2 |
| segformer_r34 | segformer_fuse | compile b16 512² | 52.3 | 44.3 | -15.3% | 3.42 | 2.32 | -32.2% | 1 |
| segformer_pvt | segformer_fuse | eager b16 512² | 58.2 | 52.3 | -10.1% | 4.93 | 3.35 | -32.1% | 2 |
| segformer_pvt | segformer_fuse | compile b16 512² | 54.9 | 47.7 | -13.2% | 4.19 | 3.10 | -26.0% | 1 |
| upernet_r34 | interp_dtype | eager b16 512² | 111.9 | 106.2 | -5.1% | 5.79 | 4.24 | -26.8% | 2 |
| deeplab_r34 | interp_dtype | eager b16 512² | 50.4 | 50.0 | -0.9% | 2.77 | 2.49 | -10.0% | 2 |
| fpn_r34 | interp_dtype | eager b16 512² | 56.6 | 57.0 | +0.7% | 2.76 | 2.76 | +0.0% | 2 |
| unet_r34_gn | gn_dtype | eager b16 512² | 94.7 | 90.3 | -4.6% | 5.06 | 4.08 | -19.3% | 2 |
| unet_r34 | split_conv | eager b16 512² | 91.0 | 94.8 | +4.1% | 3.57 | 3.39 | -5.0% | 2 |
| unet_r34 | pad_fast | eager b16 512² | 85.2 | 84.2 | -1.1% | 3.57 | 3.55 | -0.3% | 2 |
| unet_r34 | split_conv | compile b16 512² | 69.1 | 66.6 | -3.7% | 3.20 | 3.15 | -1.7% | 1 |
| unetpp_r34 | split_conv | eager b8 512² | 76.5 | 86.1 | +12.5% | 3.13 | 2.57 | -17.9% | 2 |
| unetpp_r34 | split_conv | compile b8 512² | 50.4 | 56.3 | +11.6% | 2.82 | 2.46 | -12.9% | 1 |
| unet_r34_ts | mean_agg | eager b8 256² | 31.6 | 30.6 | -3.0% | 1.87 | 1.62 | -13.3% | 2 |
| xattn_r18 | xattn_contig | eager b8 512² | 98.5 | 49.4 | -49.9% | 9.44 | 2.22 | -76.5% | 2 |

## Gradient checkpointing (train step)

| model | checkpointing | ms | peak GiB |
|---|---|---|---|
| unet_convnext b16 512² | enc | 110.5 | 2.65 |
| unet_convnext b16 512² | none | 93.5 | 6.82 |
| unet_r34 b16 512² | dec | 113.1 | 3.02 |
| unet_r34 b16 512² | dec,enc | 116.8 | 2.22 |
| unet_r34 b16 512² | enc | 96.6 | 2.78 |
| unet_r34_ts b8 256² | enc | 37.2 | 1.28 |

## Inference (bf16, inference_mode)

| model | batch | mode | ms | peak GiB |
|---|---|---|---|---|
| segformer_pvt | 1 | default | 2.49 | 0.16 |
| segformer_pvt | 1 | eager | 4.90 | 0.19 |
| segformer_pvt | 1 | reduce-overhead | 1.39 | 0.02 |
| segformer_pvt | 8 | default | 5.40 | 1.14 |
| segformer_pvt | 8 | eager | 8.87 | 1.37 |
| segformer_pvt | 8 | reduce-overhead | 5.31 | 0.05 |
| unet_pvt | 1 | default | 2.64 | 0.06 |
| unet_pvt | 1 | eager | 5.63 | 0.11 |
| unet_pvt | 1 | reduce-overhead | 1.45 | 0.03 |
| unet_pvt | 8 | default | 5.11 | 0.26 |
| unet_pvt | 8 | eager | 10.29 | 0.53 |
| unet_pvt | 8 | reduce-overhead | 5.00 | 0.06 |
| unet_r34 | 1 | default | 1.55 | 0.13 |
| unet_r34 | 1 | eager | 4.56 | 0.18 |
| unet_r34 | 1 | reduce-overhead | 1.35 | 0.10 |
| unet_r34 | 8 | default | 6.13 | 0.33 |
| unet_r34 | 8 | eager | 10.61 | 0.65 |
| unet_r34 | 8 | reduce-overhead | 5.96 | 0.13 |

## Training loop (ms per step incl. loader, median of reps)

| model | variant | ms/step | Δ vs base | peak GiB |
|---|---|---|---|---|
| unet_r34 512² b16 3ch | base | 92.3 | +0.0% | 4.45 |
| unet_r34 512² b16 3ch | nosync | 93.6 | +1.4% | 4.45 |
| unet_r34 512² b16 3ch | foreach | 94.0 | +1.8% | 4.45 |
| unet_r34 512² b16 3ch | fused | 92.9 | +0.6% | 4.45 |
| unet_r34 512² b16 3ch | loader | 94.3 | +2.1% | 4.45 |
| unet_r34 512² b16 3ch | foreach,fused,loader,nosync | 91.2 | -1.2% | 4.45 |
| segformer_pvt 256² b16 3ch | base | 23.4 | +0.0% | 1.46 |
| segformer_pvt 256² b16 3ch | nosync | 23.0 | -1.8% | 1.46 |
| segformer_pvt 256² b16 3ch | foreach | 23.7 | +0.9% | 1.46 |
| segformer_pvt 256² b16 3ch | fused | 23.3 | -0.4% | 1.46 |
| segformer_pvt 256² b16 3ch | loader | 23.2 | -0.8% | 1.46 |
| segformer_pvt 256² b16 3ch | foreach,fused,loader,nosync | 17.8 | -24.1% | 1.46 |
| unet_pvt 256² b16 3ch | base | 28.3 | +0.0% | 1.28 |
| unet_pvt 256² b16 3ch | nosync | 26.3 | -6.8% | 1.28 |
| unet_pvt 256² b16 3ch | foreach | 29.6 | +4.5% | 1.28 |
| unet_pvt 256² b16 3ch | fused | 26.1 | -7.9% | 1.28 |
| unet_pvt 256² b16 3ch | loader | 27.6 | -2.4% | 1.28 |
| unet_pvt 256² b16 3ch | foreach,fused,loader,nosync | 21.1 | -25.5% | 1.28 |
| unet_r34 256² b16 13ch | base | 28.0 | +0.0% | 1.90 |
| unet_r34 256² b16 13ch | loader | 26.4 | -5.7% | 1.90 |

## CloudSEN12+ cold reads (batch 16, U-Net R34 13-band 512²)

| variant | ms/step | waiting for data ms/step | MB/s | images/s |
|---|---|---|---|---|
| fetch_only | 299 | 299 | 361 | 53.5 |
| fetch_only | 272 | 272 | 397 | 58.8 |
| prefetch | 268 | 243 | 403 | 59.7 |
| seq | 302 | 282 | 358 | 53.0 |
| prefetch | 269 | 248 | 402 | 59.5 |
| seq | 283 | 264 | 381 | 56.5 |
| prefetch | 264 | 236 | 409 | 60.5 |

## Post-processing

| bench | device | input | current s | new s | speedup | identical / max diff |
|---|---|---|---|---|---|---|
| CC | cpu | snake b2 128² | 0.920 | 0.042 | 21.7x | True |
| CC | cpu | snake b2 256² | 4.015 | 0.093 | 43.2x | True |
| CC | cpu | blobs b8 256² | 0.140 | 0.260 | 0.5x | True |
| CC | cpu | blobs b8 512² | 8.640 | 3.877 | 2.2x | True |
| CC | cpu | fields b8 256² | 0.115 | 0.025 | 4.6x | True |
| CC | cpu | fields b8 512² | 10.623 | 0.526 | 20.2x | True |
| EDT | cpu (scipy: cpu) | b4 256² | 0.126 | 0.011 | 11.4x | 9.5367431640625e-07 |
| EDT | cpu (scipy: cpu) | b4 512² | 1.049 | 0.048 | 21.7x | 9.5367431640625e-07 |
| CC | cuda | snake b2 128² | 0.410 | 0.020 | 20.7x | True |
| CC | cuda | snake b2 256² | 1.628 | 0.037 | 44.2x | True |
| CC | cuda | blobs b8 256² | 0.015 | 0.013 | 1.2x | True |
| CC | cuda | blobs b8 512² | 0.069 | 0.032 | 2.2x | True |
| CC | cuda | fields b8 256² | 0.014 | 0.002 | 6.9x | True |
| CC | cuda | fields b8 512² | 0.082 | 0.005 | 18.0x | True |
| EDT | cuda (scipy: cpu) | b4 256² | 0.001 | 0.012 | 0.1x | 0.0 |
| EDT | cuda (scipy: cpu) | b4 512² | 0.010 | 0.049 | 0.2x | 0.0 |
