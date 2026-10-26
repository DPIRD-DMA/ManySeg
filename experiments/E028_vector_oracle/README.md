# E028: Oracle decodability of vector representations (roads, footprints) (pre-registered)

Status: **pre-registered** (2026-10-26). CPU only; no model is trained.

Background: the vector-output proposal (roads and lines as graphs, building footprints as
polygons, later LoD1/LoD2 buildings) and its Codex review. The review's main point was that the
planar graph works as a shared *output format*, but probably not as one learned representation
and decoder for every task. The cheapest test is an oracle: rasterize perfect ground-truth
vectors into each candidate set of dense targets, corrupt them in controlled ways, decode, and
score. If a representation cannot be decoded well from perfect or mildly corrupted fields, there
is no point training a model to predict it.

## Questions
1. **Roads.** Under the same corruptions, does a Sat2Graph-style graph tensor (vertexness plus,
   per angular sector, edge probability and a vector to the neighbouring vertex) decode to better
   road graphs than a road mask → skeleton, or a mask plus junction heatmap?
2. **Footprints.** Does a corner heatmap improve polygons over instance masks (ManySeg's
   boundary + distance heads) → contour → simplify?
3. **Unification.** Can the *same* graph tensor and decoder that is used for roads also produce
   footprints (closed rings → faces, labelled by the building mask) about as well as the
   dedicated footprint pipelines?

## Data
OpenStreetMap (© OpenStreetMap contributors, ODbL), fetched 2026-10-26 by `fetch_osm.py`, stored
outside the repo and not redistributed. Three ~3 × 3 km areas in Western Australia, in GDA2020 /
MGA 50:

| area | role | roads (ways) | buildings |
|---|---|---|---|
| innaloo | tune (decoder parameters) | 886 | 179 |
| murdoch | test | 1032 | 1068 |
| northam | test (regional town) | 235 | 1160 |

* Roads: drivable public classes (motorway … residential, links, living_street; no service roads
  or tracks; tunnels and covered ways dropped). Junctions exist only where OSM ways share a node,
  so bridges cross other roads without a junction, and are kept as a separate subset.
  Rasterized at **1 m** GSD in 1024² tiles. Mask widths by class (full width, m): motorway 10,
  trunk / primary 9, secondary 8, tertiary 7, residential 6, others 5.
* Buildings: closed OSM ways (multipolygon relations are skipped). Rasterized at **0.5 m** in
  1024² tiles.
* Only tiles fully inside an area are used. Vectors are clipped to the tile; a clipped edge ends at
  a boundary vertex.

## Representations and decoders
Roads:

| id | targets | decoder |
|---|---|---|
| R0 | road mask | threshold → skeletonize → pixel graph → contract degree-2 chains → prune spurs → simplify |
| R1 | road mask + junction/end heatmap (σ 2 px) | R0, then merge graph nodes within a radius of each heatmap peak and keep only spurs that end at a peak |
| R2 | Sat2Graph-style graph tensor: vertices every ≤ 20 px along roads; within 3 px of each vertex, vertexness and, for each of 6 sectors, edge probability + (dx, dy) to the neighbour | vertex NMS → for each sector above threshold, link to the nearest vertex to the predicted endpoint, within a radius |

Footprints:

| id | targets | decoder |
|---|---|---|
| B0 | `manyseg.instances` extent + boundary + distance | `instances_from_outputs` → contour per instance → Douglas–Peucker |
| B1 | B0 + corner heatmap (σ 2 px) | B0's contour, with vertices at the contour points nearest each corner peak (B0's polygon if < 3 corners) |
| B2 | the R2 graph tensor on building rings (corners are vertices; long walls densified to ≤ 20 px) + building mask | the R2 decoder → `polygonize` the graph → faces kept if mostly inside the mask → drop near-collinear vertices |

The R2/B2 encoder records how many ground-truth edges it cannot represent, i.e. two neighbours
in the same sector with no free sector within 60°.

## Corruptions
They are applied to the dense targets of every representation, with the same random draw (same
disks, same noise seed) for every representation in a tile. Severity levels:

| level | occlusion disks (radius 4–12 px) per 100 m of GT line or ring | blur σ (px) | additive noise std | missed peaks | vector noise std (px) |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0.5 | 1 | 0.1 | 5% | 1 |
| 2 | 1 | 1.5 | 0.2 | 10% | 2 |
| 3 | 2 | 2 | 0.3 | 20% | 3 |

Occlusion zeroes every probability channel inside the disk. Blur and noise apply to probability
channels, which are then clipped to [0, 1]. "Missed peaks" zero a disk of radius 3 px around a
random fraction of the vertices, junctions or corners, in the heatmap channels only. Vector noise
applies to the R2/B2 offsets. Five noise seeds per tile and level.

## Amendment 1 (2026-10-26 13:00, before any area result)
In the synthetic sanity figures (default, untuned decoder parameters), per-pixel independent
noise at level 2 (std 0.2) turned every representation into speckle: it pushes about 0.6% of
background pixels over 0.5 and punches holes inside roads, giving skeleton loops, spurs and
stray vertices everywhere (APLS ≈ 0.05 for R0, R1 and R2 alike). A network's errors are smooth,
not salt-and-pepper, so this mostly measured speckle filtering. The additive noise is now white
noise smoothed with a Gaussian (σ = 2 px, per channel), then rescaled to the table's std.
The same figures showed that a single occlusion gap disconnects a road (APLS charges every
broken path) and opens a building ring (B2 then loses the whole face), and that none of the
graph decoders could bridge a gap. That would measure decoder naivety rather than the
representation, so every graph decoder (R0, R1, R2 and B2) gets the same end-bridging step:
mutually-nearest dangling ends (not at the tile border) within `gap` px are joined, unless a
path shorter than 3 × `gap` already connects them. `gap` ∈ {0, 25, 50} px joins each
grid; to keep grids at ≤ 24, other values were dropped (grids in `run.py`). B0/B1 need no such step: a mask
contour stays closed around a gap.
When noise was changed, no real-area tile had been decoded. The gap step and its grid were fixed
after viewing one Murdoch (test) road tile decoded with default, untuned parameters (visual
check of encoders and decoders); no tuning or test metrics existed.

## Amendment 2 (2026-10-26 13:45, after a Codex review of the metric code; no tuning or test output read)
* Metric bugs fixed, with regression tests: APLS summed parallel chains between the same two
  nodes (SciPy adds duplicate sparse entries) instead of taking the shorter; an isolated loop
  shorter than 50 m got one control point and APLS 0 against itself; GEO sampled per edge,
  not per metre of road; sub-pixel peak refinement was biased at the tile border.
* Crossings: the registered metric (no predicted junction within 8 m, of all GT crossings) is
  reported, plus the same among crossings with a predicted road within 5 m, because a missing
  road otherwise counts as correct.
* Footprint tuning selects by C-IoU × detection F1 (like PQ = SQ × RQ), not C-IoU alone:
  C-IoU averages over matched pairs only, so it rewards predicting a few confident polygons.
  The reported metrics and decision rules are unchanged.
* The first tuning run (`runs/aborted_v1/`, buggy APLS) was stopped and not analysed; everything
  is rerun from scratch.

Secondary analysis: each corruption family alone at its level-2 setting (the primary analysis is
the combined levels).

## Tuning
Each decoder has a small grid of parameters: thresholds, spur length, simplify tolerance, merge or
link radius; at most 24 combinations per decoder. The grid is searched on **innaloo** at each
level with seeds 0–1, separately for each representation, and the best setting by the primary
metric is then frozen for the test areas (seeds 0–4). This is the equivalent of calibrating on
validation predictions. The chosen settings are recorded in `runs/`.

## Metrics
Per tile, in metres, then averaged over the tiles of an area (one value per area × seed).

* Roads, primary: **APLS** (control nodes every 50 m plus all junctions and ends; snapping
  buffer 5 m; path penalty min(1, |ΔL|/L), with 1 for a missing path; harmonic mean of the GT→pred
  and pred→GT scores). Secondary: GEO precision / recall / F1 (1 m samples within 5 m), junction
  F1 (degree ≥ 3, matched within 5 m), and on bridges, the fraction of GT crossings without a
  predicted junction within 8 m.
* Footprints, primary: **C-IoU** (IoU × (1 − |N_pred − N_gt| / (N_pred + N_gt)) per matched
  pair), with polygons matched one-to-one at IoU > 0.5, greedy by IoU. Secondary: detection
  F1 @ IoU 0.5, mean matched IoU, N-ratio (Σ predicted vertices / Σ GT vertices over matched
  pairs), PoLiS (m).

## Decision rules (fixed before any result)
Representation A **beats** B at a level if, on **both** test areas, A's mean over seeds exceeds
B's by more than 2 pooled seed-SDs **and** by at least 0.02 absolute (seed noise is small here, so
there is a floor). It **matches** B if it is never more than 0.02 below it on either area.

1. **Roads:** stage 1 (trained) includes R2 only if R2 beats R0 at level 1 or 2 *and* matches
   it at level 0. Likewise R1. If neither does, stage 1 trains only the mask baseline and the
   graph-tensor heads are dropped.
2. **Corner heatmap:** stage 2 includes B1 only if B1 beats B0 (C-IoU) at level 1 or 2 and
   matches it on detection F1.
3. **Unification:** B2 counts as viable only if it matches the better of B0 and B1 on C-IoU
   *and* detection F1 at levels 0–2. If it does not, one shared learned representation is
   dropped: roads and footprints keep separate dense encodings and decoders and share only the
   output schema and geometry utilities.

## Limits (stated in advance)
The corruptions are synthetic. A trained model's errors are spatially correlated and differ by
representation, which the oracle cannot show. A positive result here only licenses the trained
comparison (stage 1/2); it does not replace it. OSM completeness and alignment vary, which affects
the oracle only through the geometry it is given. The areas are all in Western Australia.

## Results
(pending)
