# E031: Oracle for LoD2 roofs: roof-edge graph vs height-model plane segmentation (pre-registered)

Status: **pre-registered** (2026-10-27). CPU only, no model is trained (local, under a systemd
memory cap; Pawsey was considered and is not needed at this size).

Background: E028 found the Sat2Graph-style graph tensor decodes building footprints well (closed,
small planar graphs), and E030 is testing it trained. An LoD2 roof is the same kind of object: in
plan view a planar graph whose outer ring is the eaves and whose inner edges are ridges, hips,
valleys and step edges, with one plane per face. This oracle asks, before any training, whether
delineating roof faces with that graph beats the classic approach, which segments a height
model (DSM / LiDAR) into planar patches.

## Question
With perfect targets corrupted in controlled ways, does decoding roof faces from a roof-edge
graph tensor (A1) give better roof structure and heights than plane segmentation of the height
model (A0), when both get the same building mask and the same height model?

## Data
3DBAG v20250903 LoD2.2 ("© 3DBAG by tudelft3d and 3DGI", CC BY 4.0), CityJSON tiles fetched and
SHA-256-verified by `fetch_3dbag.py`, stored outside the repo. The LoD2.2 roofs are themselves
reconstructed automatically from AHN LiDAR (roofer), so they are model output, not survey truth;
here they only provide realistic roof geometry.

| area | role | 3DBAG tiles | buildings |
|---|---|---|---|
| delft | tune | 4 | 4,448 |
| leidscherijn | test (new-build suburb) | 4 | 6,632 |
| staphorst | test (village, farmhouses) | 1 | 1,145 |

Rasterized at **0.25 m** (Dutch aerial imagery) in 512² tiles (128 m) on a grid over each area;
tiles with fewer than 3 buildings are skipped; at most 40 tiles per area, chosen at random with
seed 0. Ground truth per tile: every RoofSurface projected to plan view and clipped to the tile;
faces under 1 m² are dropped; each face gets the least-squares plane of its 3D vertices. The
height model is the roof planes rasterized as height above the building's ground (GroundSurface
height), 0 outside buildings. The building mask is the union of the faces.

## Arms
Both get the same (corrupted) **building mask** and **height model**.

| id | roof faces from | then |
|---|---|---|
| A0 | the height model: smooth, per-pixel gradient, join 4-neighbours inside the mask whose gradients and heights agree, drop small regions, merge regions with near-identical planes, polygonize (`polygons_from_labels`) | robust (trimmed) least-squares plane per face |
| A1 | a graph tensor of the plan-view roof graph (`manyseg.vectors.graph_tensor`, planar, 20 px spacing): `polygons_from_tensor` with faces kept inside the mask | same plane fit per face |
| LoD1 | the building mask (one face per connected component) | one flat height (median) | reference only |

## Corruptions
Graph tensor and mask: E028's levels (occlusion disks along roof edges, blur, correlated noise,
missed peaks, vector noise). Height model, per level: L0 none; L1 white noise 0.05 m; L2 0.15 m
plus blur σ 1 px; L3 0.3 m plus blur σ 2 px plus 2% of pixels lost (filled from the nearest
valid pixel). Five noise seeds per tile and level; the same draw for both arms.

## Tuning
Each arm's parameters (A0: smoothing, gradient and height tolerances, minimum region, merge
tolerance, simplify; A1: E030's B2 decoder grid) are tuned on **delft** at each level with seeds
0–1, at most 24 settings per arm, by face F1 × (1 − min(1, height RMSE / 1 m)); frozen for test.

## Metrics
Per tile, pooled per area: **face F1** (plan-view faces matched one-to-one at IoU > 0.5,
`polygon_metrics`), **height RMSE** (m; reconstructed roof surface vs ground truth over all
building pixels; a missing roof counts as height 0), corner F1 (face vertices matched within
0.5 m), C-IoU of matched faces, and normal angle error of matched faces (degrees).

## Decision rule (fixed before any result)
A1 **beats** A0 at a level if on both test areas its face F1 is higher by more than 2 pooled
seed-SDs and by at least 0.02, and its height RMSE is no more than 0.05 m worse. A1 is **viable**
if it is never more than 0.02 below A0 in face F1, nor more than 0.05 m worse in height RMSE, at
levels 0–2.
* A1 beats A0 at level 1 or 2 and is viable → a trained LoD2 experiment with roof-graph heads
  (imagery + height model in, roof graph + mask out) is worth running.
* A1 viable but never beats A0 → roof graphs are not worth a trained experiment by themselves;
  LoD2 uses plane segmentation of a height model, and ManySeg's part is the footprint.
* A1 not viable → as above, and the graph tensor is not used for roofs.

## Limits (stated in advance)
The corruptions are synthetic and independent between the graph and the height model, whereas
a trained network's graph errors would correlate with what the height model shows. The labels
come from an automatic LiDAR reconstruction. Dutch roofs only.

## Amendment 1 (2026-10-27 09:12, before tuning)
A 2-tile smoke run on delft (tuning data) showed the registered tuning objective,
face F1 × (1 − min(1, RMSE / 1 m)), is 0 for almost every setting: height RMSE is above 1 m even
for good decodes, because a pixel assigned to the wrong side of a step edge (often 10–20 m) and
uncovered pixels (height 0) dominate it. Tuning would then pick the first grid entry. The
objective is now **face F1 / (1 + RMSE in m)**, which never saturates. Metrics and decision rules
are unchanged. The aborted run was discarded.

## Results
(pending)
