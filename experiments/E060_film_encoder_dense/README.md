# E060: Does FiLM between encoder stages help a dense task? (NYUv2 metric depth under zoom)

Status: **done, negative** (2026-11-09, RTX 3090, torch 2.14.1+cu130, seeds 0–2). FiLM
between encoder stages shows no demonstrated improvement over today's FiLM on a dense task. **Pre-registered:** this
README, the arms and the decision rule were written before any E060 run.

## Question
E058 found that for an image-level head, FiLM after every encoder stage beat ManySeg's FiLM,
which there reaches only the pooled feature. For pixel heads, ManySeg's FiLM already modulates
every encoder level entering the decoder plus the decoder output, and it won in E055. Before
adding an `on: ["encoder"]` option to ManySeg, this checks whether FiLM between stages also helps
a dense task. Explainer with diagrams: <(private link removed)>
(private).

## Protocol
* Data, model, recipe and metrics exactly as E055 (`train.py` is E055's plus two arms): NYUv2,
  zoom z log-uniform in [1, 2.4], 192×256 crops, ResNet-34 U-Net predicting log depth, L1 loss,
  100 epochs, test AbsRel averaged over z ∈ {1, 1.4, 1.8, 2.4}. Seeds 0–2.
* Arms, all re-run on this code:
  * `none`: no vector (baseline).
  * `film`: ManySeg FiLM, `on: ["features", "output"]` (today's default for pixel heads).
  * `film_encoder`: FiLM after each ResNet stage via forward hooks, no ManySeg conditioning.
    Same MLP shape as ManySeg's metadata encoder, zero-initialised projection.
  * `film_both`: ManySeg FiLM plus FiLM after each stage.
* Checked before launch: `film_encoder` gives exactly the `none` output at init (max diff 0.0)
  and changes it once its projection is non-zero; parameters: none 24.44 M, film 24.58 M,
  film_both 24.70 M. Codex review found no bugs.

## Decision rule
* Build `on: ["encoder"]` into ManySeg for dense models if `film_encoder` or `film_both` beats
  `film` on test AbsRel by more than 2 pooled SD over seeds 0–2, and val agrees in sign.
* If neither does, keep today's FiLM for pixel heads. The E058 result then only supports an
  image-head option, or documenting `channels` for image-level tasks.
* Secondary: the improvement of `film` over `none` should repeat E055 (0.0080 test AbsRel); a large miss would mean
  the re-run differs from E055 and needs explaining first.

## Results (mean ± seed SD, seeds 0–2)

| arm | test AbsRel | per seed | val AbsRel | δ1 | scale_err | si_log |
|---|---|---|---|---|---|---|
| none | 0.1823 ± 0.0034 | 0.1805 / 0.1801 / 0.1862 | 0.1858 ± 0.0026 | 0.699 | 0.1261 | 0.1570 |
| film (today) | 0.1742 ± 0.0018 | 0.1727 / 0.1762 / 0.1738 | 0.1790 ± 0.0014 | 0.726 | 0.1144 | 0.1541 |
| film_encoder | 0.1739 ± 0.0004 | 0.1736 / 0.1738 / 0.1744 | 0.1788 ± 0.0025 | 0.722 | 0.1172 | 0.1540 |
| film_both | 0.1737 ± 0.0036 | 0.1727 / 0.1777 / 0.1708 | 0.1771 ± 0.0027 | 0.729 | 0.1138 | 0.1537 |

About 9 min per run (8.5–9.0; two at a time, GPU otherwise idle).

| comparison (AbsRel improvement) | test | 2 pooled SD | val | 2 pooled SD | verdict |
|---|---|---|---|---|---|
| film_encoder over film | +0.0003 | 0.0026 | +0.0002 | 0.0041 | no difference |
| film_both over film | +0.0005 | 0.0056 | +0.0018 | 0.0044 | no difference |
| film over none | +0.0080 | 0.0054 | +0.0068 | 0.0043 | helps (as E055) |
| film_encoder over none | +0.0084 | 0.0048 | +0.0070 | 0.0052 | helps |
| film_both over none | +0.0086 | 0.0069 | +0.0086 | 0.0054 | helps |

* **Decision: does not pass.** Neither `film_encoder` nor `film_both` demonstrates an improvement over
  today's FiLM: both are slightly lower on mean AbsRel, by about a tenth of the 2-SD bar.
  Equivalence was not tested, only the absence of a demonstrated gain. Today's FiLM stays for pixel heads, and
  `on: ["encoder"]` is not built as a general option.
* **Secondary check holds exactly.** `none` and `film` reproduce E055 to four decimals on every
  seed, so the re-run matches E055.
* All three placements give similar gains over `none` (0.008–0.009 test AbsRel). A plausible
  reading: for pixel heads, ManySeg's FiLM already reaches enough of the network: every encoder level entering the
  decoder and the decoder output.
* **With E058:** FiLM between encoder stages helped only where ManySeg's FiLM is weakest, image
  heads (pooled feature only). What the evidence supports is an image-head change: either
  document `channels` for image-level heads (within noise of encoder FiLM in E058, and it already
  exists) or make encoder FiLM the image-head placement.
