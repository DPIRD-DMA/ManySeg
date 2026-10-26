# E030: Trained footprint polygons on WHU aerial: instance masks vs graph tensor (pre-registered)

Status: **pre-registered** (2026-10-26). Local RTX 4090.

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

## Results
(pending)
