# E055: Does FiLM help when the metadata changes the answer? (NYUv2 metric depth under zoom)

Status: **done, positive** (2026-11-09, RTX 3090, torch 2.14.1+cu130, seeds 0–2). Seed 0
ran first as a screen; the decision rule below was written after seed 0 was seen and before
seeds 1–2 ran. FiLM passes: the first experiment here where metadata conditioning helps.

## Question
FiLM showed no clear gain in E009, E013, E016 and E020. In E016 a shuffled-metadata arm matched
the real one, so the model never used the vector: on those tasks the metadata (date, location)
was either redundant with the pixels or irrelevant. This experiment picks a task where the
vector *should* change the correct output for the same pixels.

Metric depth from one image depends on the focal length. Zooming in (cropping a 1/z window and
resizing it to the output size) multiplies the effective focal length by z and makes everything
look closer, while the true depth is unchanged. From pixels alone the zoom is ambiguous apart
from object-size priors, so a model told z should recover absolute scale better
(CAM-Convs, Fácil et al., CVPR 2019, made this point for cameras with different intrinsics).

## Protocol
* Data: NYU Depth v2 labelled set via `jagennath-hari/nyuv2` on Hugging Face (MIT; 1,014 train /
  290 val / 145 test; hole-filled depth in mm, 0.7–10 m). `prep.py` packs it into npz.
* Each sample: crop a 3:4 window of side 1/z of the usable 464×618 frame (8 px border removed),
  resize to 192×256. z is log-uniform in [1, 2.4]; z = 2.4 is native resolution, so every crop is
  *downscaled* (antialiased), so there's no upsampling blur to give z away. Random crop position and
  flip in training; centre crops at z ∈ {1, 1.4, 1.8, 2.4} for val/test, metrics averaged over z.
* Model: ManySeg U-Net, ResNet-34 ImageNet weights, a pixel regression head predicting log depth,
  L1 loss on log depth. AdamW 3e-4, one-cycle, 100 epochs, batch 16, bf16. 9–22 min per run (mean 15; up to four jobs shared the GPU).
* Arms (vector input = log z, a `continuous` field):
  * `none`: no vector (**baseline**).
  * `film`: FiLM (the default conditioning).
  * `channels`: log z broadcast as an extra input channel.
  * `film_shuffled`: FiLM fed a z drawn independently of the image's zoom (same distribution, no
    information). A control: should match `none`.
  * `nozoom`: reference only, trained at z = 1 (the usual NYU task), tested at every zoom.
  * `probe`: diagnostic. An image-scope head predicts log z from pixels. If it can, the baseline
    can also infer zoom, and that limits how much the vector can add.
* Metrics: AbsRel (primary), RMSE, δ1, scale-invariant log error (`si_log`, shape only) and
  |median log ratio| (`scale_err`, absolute scale only).
* Code reviewed by Codex before launch. Fixes taken: the shuffled control was a within-batch
  permutation, which leaks z because eval batches share one zoom; `nozoom` docs; the `probe` arm
  added for the resampling-cue concern.

## Decision rule
* FiLM helps if `none` − `film` test AbsRel > 2 pooled SD over seeds 0–2, `film_shuffled` is
  within 2 pooled SD of `none` (the gain needs the true z), and val agrees in sign.
* If it passes, FiLM on a zoom/focal-length vector goes into "What the evidence says" as the first
  task where FiLM helped, with the condition that the vector must change the target for fixed
  pixels.

## Results (mean ± seed SD, seeds 0–2; metrics averaged over the four test zooms)

| arm | test AbsRel | val AbsRel | per seed (test) |
|---|---|---|---|
| none (baseline) | 0.1823 ± 0.0034 | 0.1858 ± 0.0026 | 0.1805 / 0.1801 / 0.1862 |
| **film** | **0.1742 ± 0.0018** | **0.1790 ± 0.0014** | 0.1727 / 0.1762 / 0.1738 |
| channels | 0.1799 ± 0.0024 | 0.1822 ± 0.0035 | 0.1772 / 0.1808 / 0.1816 |
| film_shuffled | 0.1811 ± 0.0034 | 0.1823 ± 0.0019 | 0.1850 / 0.1797 / 0.1786 |

Means over seeds: δ1 0.699 (none) → 0.726 (film); scale_err 0.126 → 0.114; si_log
0.157 → 0.154.

* **Decision: passes.** `none` − `film` = 0.0080 test AbsRel against 2 pooled SD = 0.0054; FiLM
  wins on every seed (paired 0.0078 / 0.0039 / 0.0123); val agrees (0.0068 against 0.0043).
  `film_shuffled` − `none` = −0.0012, inside 2 pooled SD (0.0068): the gain needs the true z.
* **What FiLM fixes is absolute scale.** scale_err falls 9%, si_log (shape) barely moves.
* **FiLM beats `channels`.** For this dense task FiLM modulates every encoder level entering the
  decoder and the decoder output; `channels` only adds z at the input.

### Diagnostics (seed 0)

* `nozoom` (trained at z = 1 only): z1.0 AbsRel 0.168, δ1 0.767; at z2.4 AbsRel 0.340,
  δ1 0.225, scale_err 0.414. Zoom does change metric depth, and a model that never saw it
  breaks.
* `probe` (predict log z from pixels): log z RMSE 0.053 against a prior SD of 0.253,
  R² = 0.96. A probe trained with random blur (σ 0–1.5 px) and noise (SD 0–0.03) drawn
  independently of z (`--degrade`) still gets R² = 0.90. Zoom stays highly predictable from
  pixels under these degradations. Scene content (object and room sizes) is the likely source,
  but these probes don't fully exclude resampling cues. A likely reason the gain is small: the
  vector can only add what the pixels don't already carry.

### Takeaway
FiLM helped on a task where the vector changes the correct output for fixed pixels. A
plausible reading of the earlier nulls (E009, E013, E016, E020) is that their metadata was
implied by the pixels or irrelevant (E016's shuffled arm matched the real one), but these
experiments don't test that directly. Larger gains might be expected where the vector is less
recoverable from the image; that's a hypothesis, not a result.
