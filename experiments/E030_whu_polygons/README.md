# E030: Trained footprint polygons on WHU aerial: instance masks vs graph tensor (pre-registered)

Status: **complete, one seed** (2026-10-27; pre-registered 2026-10-26): B2 (graph tensor) well
below B0 (mask instances) on seed 0, remaining seeds cancelled (amendment 3); the graph tensor
was not shipped. Local RTX 4090.

Background: in the E028 oracle (perfect OSM footprints with synthetic corruption), the
Sat2Graph-style graph tensor decoded to faces (B2) gave the best polygons through corruption
level 2 (C-IoU 0.72–0.75 vs 0.53 for instance-mask contours) and failed all-or-nothing. An
oracle cannot show how a *trained* network's errors look per representation; this experiment
does. It decides whether `manyseg.vectors.graph_tensor` stays (opt-in, experimental) or is
removed.

## Question
Trained on the same data with the same backbone, decoder, recipe and budget, do graph-tensor
heads (B2) give better building polygons than boundary + distance instance heads with
polygonised instance masks (B0)?

## Data
WHU Building, aerial subset (Christchurch; LINZ imagery, CC BY 4.0; WHU labels, licence not
stated, used for research only and not redistributed): 0.3 m, 512² tiles, official split 4,736
train / 1,036 val / 2,416 test. Ground-truth polygons come from the official vector shapefile
clipped to each tile if the tiles are georeferenced; otherwise from the official raster labels
via `manyseg.vectors.polygons_from_labels` (exact pixel edges). This choice is made from the
files, before any training, and recorded here. In both cases a GT polygon's vertex count is taken
after a 0.5 px simplification (as in E028), and tiles are clipped at their border.

## Arms (3 seeds each, seeds 0–2)
Both: ManySeg U-Net, resnet34 `tv_in1k`, one shared decoder (no per-head decoders), recipe
`fastai_like`, 40 epochs, batch 16, bf16 autocast, full 512² tiles with a random dihedral
transform (8 elements, applied to image, labels and vertex coordinates before targets are
encoded), same training loop (`train.py`).

| id | heads | targets | loss | decoding |
|---|---|---|---|---|
| B0 | `instance_heads()`: extent, boundary, distance | `instance_targets(boundary_width=2)` (E010) | dice-BCE extent + dice-BCE boundary + masked L1 distance (bench `InstanceTask`) | `instances_from_outputs` → `polygons_from_labels` |
| B2 | `graph_tensor_heads()`: vertex, edge (6), vector (12), extent | `vertex_graph(spacing=20, border=tile)` → `encode_graph_tensor` | BCE vertex (soft Gaussian target, logits) + BCE edge on support pixels + smooth-L1 vector / 20 on pixels whose edge target is 1 + dice-BCE extent | `polygons_from_tensor` |

## Decoder tuning
Each arm's decoder parameters are tuned on **val** (one grid of at most 24 settings per arm,
listed in `tune.py`), separately for each seed, by C-IoU × detection F1, and frozen. Test is
decoded once per run.

## Metrics
On test, pooled over tiles (`manyseg_bench.evaluation.vectors.polygon_metrics` per tile; counts
summed, matched-pair metrics averaged over all matched pairs): **C-IoU** (primary), detection F1 @
IoU 0.5, matched IoU, PoLiS (m), N-ratio, C-IoU × detection F1. Also pixel IoU of the extent head
and decode time per tile.

## Decision rule (fixed before any run)
B2 **wins** if its test C-IoU exceeds B0's by more than 2 pooled seed-SDs and by at least 0.02,
**and** its test detection F1 is no more than 0.02 below B0's.
* B2 wins → `graph_tensor` stays as an opt-in, experimental polygon head, documented with E030.
  It becomes the recommended footprint head only after a second dataset (P^3 or SpaceNet 2)
  replicates this (the repository's two-dataset rule).
* Otherwise → `graph_tensor` is removed from the public API (the code stays in `experiments/`),
  and the recommended footprint pipeline is instance heads + `polygons_from_labels`.

A **training failure** check: if after training an arm's val extent IoU is below 0.5, or B2's
vertex head has no peak above 0.05 on val, the loss setup is broken. It is fixed and *both* arms
are rerun, as a dated amendment.

## Budget
6 runs, about 3 h of GPU on the 4090 (CPU workers encode B2 targets). Stop and report if it
exceeds 6 h.

## Amendment 1 (2026-10-26 16:50, before any training)
The WHU tiles are not georeferenced, so, as the Data section allows, ground truth comes from the
raster labels: buildings are the label's False pixels (checked visually; empty tiles are all
True), instances are 4-connected components (buildings that share a wall are one instance, for
both arms alike). Raster outlines of rotated buildings are staircases that a 0.5 px
simplification keeps, which would inflate GT vertex counts and favour staircase-like
predictions in C-IoU. GT polygons are therefore simplified by **1 px (0.3 m)** when they are made
(`prepare.py`), checked visually to keep real corners; the metric's 0.5 px tolerance then
applies on top as registered. Nothing else changed.

## Amendment 2 (2026-10-27 08:20, before any full training run)
The first chain (26 Oct, 17:00) ran the workstation out of memory during the B2 *preflight*
evaluation (a 1-epoch model trained on 64 tiles): its vertex map is noise, so at the lowest
tuning threshold every tile gave tens of thousands of peaks, and polygonizing their edges took
over 10 GB per decoding worker (about 110 GB in total). Changes, before any full run: the B2
decoder keeps at most the 5,000 strongest vertex peaks per tile (`max_vertices`; a WHU tile has
at most about 2,000 GT vertices, so a trained model is unaffected) and drops decoded edges longer
than 60 px (`max_edge_length`; true edges are at most 20 px, bridged gaps 25 px), since a first
fix with the vertex cap alone still reached the 12 GB cap on the noise model; evaluation uses 6 decoding
workers, counts a tile whose decoding runs out of memory as an empty prediction
(`failed_tiles`, reported); and the chain runs in a systemd scope capped at 12 GB, so a fault
kills only this job. Nothing else changed.

## Amendment 3 (2026-10-27 12:45, agreed with the maintainer after a Codex review of the options, `E032_graph_completeness/OPTIONS.md`; no result of the affected runs exists): remaining seeds cancelled
Seed 0 (test): B0 C-IoU 0.709, detection F1 0.850; B2 C-IoU 0.613, F1 0.750. Both gaps (about 0.1)
are far larger than plausible seed variation, so B2's seeds 1–2 and B0's seed 2 are cancelled to
free the GPU for redesigns of the graph method; B0 seed 1 (already running) finishes, for baseline
variance. The registered 3-seed rule therefore cannot be applied: E030's result is reported as
**one seed, B2 well below B0**, not as a full decision. E030b runs on seed 0 only. If a later arm
advances to seeds 1–2, B2 control seeds 1–2 are trained then. Also added (post hoc, labelled as
such): C-IoU on GT outlines cleaned of raster staircase vertices, and detection F1, matched IoU
and PoLiS reported alongside, because the registered GT inflates vertex counts.

## Results
Seed 0, test (amendment 3): B0 C-IoU 0.709, detection F1 0.850; B2 C-IoU 0.613, detection
F1 0.750. The per-building fallback is E030b below. Registered 3-seed rule not applicable
(seeds cancelled); reported as one seed.

Outcome for the library (2026-10-29, user decision before 0.1.0): with B2 well below B0
on seed 0 (see E030b's table) and the remaining seeds cancelled, the graph tensor was removed
from `manyseg.vectors` as the decision rule says; the code is now `graph_tensor.py` in this
directory, its tests `test_graph_tensor.py`.

# E030b: per-building fallback on the trained models (pre-registered 2026-10-27 11:00)

After E030 finishes: for each seed, B2F as in E028b, with that seed's B2 model for the graph and
its extent, and that seed's B0 model for the fallback (`instances_from_outputs` →
`polygons_from_labels`). `cov` tuned on val (C-IoU × detection F1), test decoded once. This is a
**two-model ensemble** (about twice B0's or B2's inference cost) and is reported as such; a single
model with both head sets is a later training experiment. Rule as E030 (C-IoU primary; detection
F1 not more than 0.02 lower), against B2 (primary) and B0, paired by seed.

## E030b results

Status: **complete for seed 0 only** (2026-10-27; E030 amendment 3 cancelled the other seeds, so
the registered paired rule cannot be applied). Tuned coverage threshold 0.95: the graph was kept
for 36% of buildings, the B0 model's polygons used for the rest.

| seed 0, test | C-IoU | detection F1 | C-IoU × F1 | matched IoU | PoLiS (m) | N-ratio |
|---|---|---|---|---|---|---|
| B0 | 0.709 | 0.850 | 0.603 | 0.861 | 0.39 | 1.18 |
| B2 | 0.613 | 0.750 | 0.460 | 0.854 | 0.39 | 0.49 |
| B2F (B2 + B0 fallback, two models) | 0.643 | 0.814 | 0.524 | 0.857 | 0.39 | 0.70 |

The fallback recovers part of B2's missing buildings (detection F1 +0.06) but stays below B0 on every
column except PoLiS. On the trained models the graph method's closure failures are frequent enough
that the best per-building gate hands most buildings to the mask model; one seed.
