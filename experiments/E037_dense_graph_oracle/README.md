# E037: dense pixel graph (P8) and mask-supported local links (R3), oracle (pre-registered 2026-10-28 10:00)

Why: the maintainer proposed a dense pixel-graph output ([B, 11, H, W]: node confidence, 8-neighbour
links N..NW, sub-pixel XY offset; no vertex cap), asking that its limits be investigated rather
than assumed away. Codex's options review (ManySeg-e028/notes/codex_options.md) recommends a
cheap representation-loss and chain-survival gate before any corruption sweep or training, and a
mask-supported variant (links only where the mask skeleton is ambiguous) as the preferred unified
candidate. Plan: ManySeg-e028/notes/unified_graph_plan.md (Phase A1-A2; A3 corruption sweeps
need an amendment naming the survivors).

## Representations
**P8 (pure dense graph)**: GT lines (roads: graph edges; footprints: building rings without the
tile-border parts, as E028's B2; roofs: E031 roof-face edges, shared edges once) are split into
straight segments; each segment is rasterised as the 8-connected DDA path from the pixel holding
its start to the pixel holding its end. Every path pixel is a node (confidence 1); consecutive
path pixels are linked both ways; offsets: the GT vertex for pixels holding one, otherwise the
closest point of the segment to the pixel centre. A pixel claimed by two segments that do not
share a GT vertex there, or holding two GT vertices, is a *collision*: it is still one node, so
topology changes (recorded).
**P8r (displaced)**: P8, but each GT vertex (highest degree first) may move its pixel within
Chebyshev radius 1 (9 options) to minimise new collisions on its incident segments (ties: no
move); the offset still points to the true position (up to 1.5 px). Tests "a vertex may sit on a
nearby pixel when that preserves topology".
**R3 (mask-supported local links, roads)**: E028's R0 road mask + P8r links only inside zones:
disks of radius 10 px around keypoints (GT ends, junctions of degree >= 3, crossings) plus
*merge zones* (pixels where the road masks of two different chains overlap or come within 2 px,
dilated by 12 px: where a mask skeleton cannot separate roads). Decoder: R0's decoded graph
clipped outside the zones + the P8 graph decoded inside them, joined in vector space (each line
end on a zone boundary to the nearest end of the other kind, one-to-one, within 4 px).
*Design note (before any full A1 result):* the zone and join rules above were fixed while
debugging on 4 Murdoch smoke tiles with rendered overlays: a pixel-level join made zig-zags and
lost lines that cross the zone boundary repeatedly; keypoint zones alone left dual
carriageways to the mask (ladder skeletons); 3 px merge zones were too thin (lines exited where
the lanes still touched). With these rules R3 matched P8r on the 4 smoke tiles. (R3 therefore had development-set
tuning the fixed P8 decoder did not; Codex review #10.) *Codex review fix:* R3's encoded
node/link/offset targets are now zeroed outside the zones (they were full-tile; decoding was
already zone-restricted, so results are unchanged).
Footprints and roofs: P8 faces are kept only if mostly inside the building mask, which is part
of the P8 tensor here as it is for B2 (Codex: a pure 11-channel P8 is too risky for areas).
P8 decoder: nodes with confidence > 0.5; an edge where both reciprocal links > 0.5 ("AND");
positions = pixel centre + offset; degree-2 chains contracted and Douglas-Peucker simplified
(1 px roads / 0.5 px buildings and roofs); footprints/roofs: polygonize with the tile border,
faces kept if > 50% inside the mask (E028/E031 rule), roof planes from the height model (E031).

## Data (as E028 / E031; public OSM and 3DBAG)
Roads: E028 Murdoch + Northam (32 tiles, 1 m/px, 32 overpasses) and City-Scale (public; 20 val
tiles at 1 m/px). Footprints: E028 Murdoch + Northam (0.5 m/px) and 100 WHU val tiles (0.3 m/px).
Roofs: E031's 3DBAG tiles.

## A1: L0 inventory (perfect targets, no corruption, fixed decoders, no tuning)
Arms: roads R0, R2, P8, P8r, R3; footprints B0, B2, B3r (E035b decoder), P8, P8r; roofs A1 (E031),
P8, P8r. Metrics: roads APLS, junction F1, crossings preserved (E028 `crossing_score`),
collision rate (per GT junction / per km); footprints C-IoU, detection F1, PoLiS, N-ratio, ring
closure (GT rings decoded as a face with IoU > 0.5); roofs face F1, height RMSE (E031 `metrics`).
Stratified: junction minimum angle (< 15, 15-30, 30-45, > 45 deg), crossing vs not, building
perimeter (terciles), minimum separation to another building (< 2 px vs >= 2 px).
**Kill at L0** (applies to P8, P8r, R3), any of:
1. > 0.5% of GT junctions lost or wrongly connected (junction F1 < 0.995) or > 1% of GT rings /
   roof faces not closed, aggregate;
2. primary metric (APLS / C-IoU x det F1 / face F1) more than 0.02 below the best existing oracle
   arm at L0 (R2 / B2 / A1);
3. a hard-case stratum (angle < 15 deg, crossings, separation < 2 px) with > 5% topology failure
   *and* that stratum making up > 1% of its features (else reported, not killing).

## A2: target stability and chain survival (survivors of A1)
Stability: encode the 8 dihedral transforms of each tile (inverse-transformed back) and +-0.25 px
translations; positive-link disagreement = |symmetric difference| / |union| of link sets.
Chain survival: delete each positive link independently with probability q in {0.001, 0.005,
0.01, 0.02, 0.05} (one-sided: each direction separately; decoders AND and OR), plus correlated
gaps (runs of 3 links, same expected rate); with and without E028's end bridging (gap 10 px).
Reported vs chain length / perimeter alongside p^L. Seeds 0-2.
**Kill at A2**, any of:
1. positive-link disagreement > 5% under dihedral or +-0.25 px transforms (aggregate);
2. at q = 0.01 (best decoder, bridging allowed), the primary metric falls more than 0.05 below
   its L0 value while the mask-supported arm on the same task (R3 / B3r / E031b) falls less.
Survivors go to A3 (amendment before any A3 result). If all P8 variants die, the link head is kept
only inside R3-style zones.

## Amendment 1 (2026-10-28 10:10, while A1 ran; no A1 result read)
Writing the A2 stability check exposed a target-entropy problem in the registered rasteriser:
the DDA path interpolates between the two *end cells*, so when an end vertex sits on a pixel
boundary a flip moves that cell by one and the whole path shifts (toy lines: mean dihedral
link disagreement 0.45, max 0.81). Added arms with a **geometric rasteriser** (`geo_path`): the
end cells, and in between, per column (or row, along the major axis) the cell holding the true
line at the column centre; gaps filled by DDA. Toy lines: disagreement 0.02 (max 0.11). New arms
**P8g, P8rg** (all tasks) and **R3g** (roads), same decoders, same A1/A2 rules; the registered
P8/P8r/R3 keep their rules (A2 stability will be reported for them as registered). A1 for the new
arms runs after the registered A1.

## Compute
CPU only, systemd MemoryMax <= 12 GB, OMP_NUM_THREADS=1, <= 6 processes (E036 shares the box).

## Results

### A1 (L0, perfect targets; 270 tiles: 68 road, 122 building, 80 roof)
Roads (APLS / junction F1 / junctions lost or wrong-degree / crossings kept, of 285):

| arm | APLS | junc F1 | lost/wrong | < 15 deg fail | crossings |
|---|---|---|---|---|---|
| R0 mask | 0.927 | 0.799 | 22.2% | 96% | 19 |
| R2 graph (20 px) | **0.982** | 0.976 | 3.9% | 43% | **245** |
| P8 / P8g | 0.946 | 0.967 | 2.1% | 33% / 29% | 23 |
| P8r | 0.958 | **0.983** | **1.0%** | **9%** | 96 |
| P8rg | 0.949 | 0.972 | 1.5% | 28% | 43 |
| R3 / R3g | 0.948 / 0.939 | 0.983 / 0.972 | 1.0% / 1.5% | 9% / 28% | 96 / 43 |

Footprints (C-IoU x det F1; OSM 0.5 m vector GT / WHU 0.3 m raster-traced GT): B0 0.900 / 0.652,
B3r 0.949 / 0.921, B2 0.976 / 0.954, P8 0.985 / 0.996, P8r 0.987 / 0.996, P8g 0.985 / 0.996,
P8rg 0.985 / 0.997. Rings not closed: 0.15-0.33% for every arm. WHU outlines are traced from
rasters (vertices on pixel corners); P8 reproduces them vertex for vertex, which C-IoU's
vertex-count term rewards, so the OSM column is the fair comparison (checked visually).
*Regularised WHU GT (post hoc, user's suggestion):* the 100 WHU tiles' outlines regularised with
buildingregulariser 0.2.5 (parallel_threshold 3 px, simplify_tolerance 1.5 px, no circles; a sweep
of 0.5-6 px gave ~0.54x vertices at median IoU 0.95 vs the raw outline; it staircases walls not
square to the building). C-IoU x det F1 / PoLiS m: B0 0.870 / 0.146, B3r 0.939 / 0.065, B2 0.947
/ 0.013, P8r 0.982 / 0.010, P8rg 0.982 / 0.010. The raw GT's staircases cost B0 most (0.652 ->
0.870); the ranking is unchanged.
Roofs (face F1 / height RMSE m): A1 0.992 / 0.320, P8 0.994 / 0.268, P8r 0.995 / 0.253,
P8g 0.994 / 0.267, P8rg 0.995 / 0.246.

**L0 decision:** roads: every P8 arm and R3 **killed** (rule 1 junction F1 0.967-0.983 < 0.995, rule 2
APLS 0.024-0.043 below R2, rule 3 crossings: one node per pixel turns a crossing on a shared
pixel into a false junction; R2 itself would fail rule 1 with junction F1 0.976). *Codex review fix:*
`analyze_a1.py` first implemented rule 1 as GT-junction recall with correct degree (> 0.5% lost);
it now uses the registered junction F1; no verdict changed. Footprints and roofs: all
P8 arms survive L0, and are the best arms at L0.

### A2 stability (positive-link disagreement, mean; rule: kill if > 0.05)
Dihedral: DDA arms 0.06-0.14 (roads/roofs), geometric arms 0.002-0.02 (roads/roofs); buildings
OSM 0.004 (geo) but WHU 0.39-0.77 (outlines on pixel edges: every floor() is a tie).
+-0.25 px shifts: **0.34-0.46 for every arm and task** -> by the registered rule **all P8 arms
are killed**. Diagnosis (rendered): a quarter-pixel shift moves the steps of each rasterised
staircase by one pixel, and every moved step swaps two links; with a 1-px tolerance the
disagreement is 0.000 on every task. This is aliasing of any hard one-pixel line target, not a
P8-specific defect: a trained P8 would need a shift-tolerant (e.g. distance-weighted) link loss,
which an oracle cannot test.
*Calibration check (post hoc, 20 building tiles, +-0.25 px):* B0 extent pixels 0.014, **B0
boundary pixels 0.281**, B2 edge pixels 0.105, B2 vertex pixels 0.170. The registered 5% rule
would also have rejected B0's boundary head (which trains well; B0 is the best trained arm) and
B2's targets. This is suggestive, not a calibration: it was computed ad hoc (not a committed
script), and pixel-set disagreement of a heatmap is not structurally the same quantity as
disagreement of a link set (one moved staircase step swaps two links but one boundary pixel)
(Codex review #7). The registered verdict stands; a trained P8 would need its own
pre-registration with a shift-tolerant link loss.

### A2 chain survival (diagnostic only, after the kill; user's direction-aware healing)
P8rg (buildings 40 tiles, roofs 30, roads 20) and R3g (roads), 3 seeds; damage: independent
one-sided link loss ("link"), runs of 3 links lost both ways ("link3"), missing node pixels whose
neighbours still point at them ("node"). Decoders: AND/OR reciprocal rule, E028 end bridging
(10 px), and the maintainer's healing (`heal`: only ends with a dangling link heal, searching along the
line's own direction, end-to-end or end-to-line). Primary metric (L0: buildings 0.990, roads APLS
0.925, roofs face F1 0.995):

| task | damage | q | OR | OR + bridge | OR + heal | OR + bridge + heal |
|---|---|---|---|---|---|---|
| buildings | node | 0.01 / 0.05 | 0.42 / 0.04 | 0.93 / 0.58 | 0.96 / 0.80 | **0.97 / 0.91** |
| buildings | link3 | 0.01 / 0.05 | 0.73 / 0.25 | 0.98 / 0.86 | 0.73 / 0.25 | **0.98 / 0.86** |
| roads | node | 0.01 / 0.05 | 0.12 / 0.01 | 0.86 / 0.27 | 0.92 / 0.88 | **0.92 / 0.90** |
| roads | link3 | 0.01 / 0.05 | 0.46 / 0.05 | 0.90 / 0.68 | 0.46 / 0.05 | **0.90 / 0.68** |
| roofs | node | 0.01 / 0.05 | 0.66 / 0.13 | 0.95 / 0.65 | 0.98 / 0.91 | **0.98 / 0.93** |
| roofs | link3 | 0.01 / 0.05 | 0.86 / 0.47 | 0.97 / 0.88 | 0.86 / 0.47 | **0.97 / 0.88** |

One-sided link loss is absorbed by the OR rule (buildings 0.98 at q 0.01). Healing is what saves
missing pixels (it uses the neighbours' dangling links); links lost both ways leave no evidence,
so bridging covers those. With bridge + heal every task stays within 0.02 of L0 at q = 0.01 (the
registered rule-2 tolerance is 0.05; with the registered decoders alone, bridging only, node
loss would fail it for buildings 0.062 and roads 0.070). *Scope (Codex review #8):* the healing
decoder was added after the kill and selected from six decoders; the node-loss damage model keeps
the neighbours' links pointing at the missing pixel, which is exactly the cue `heal()` uses. A
trained model may drop a pixel together with its neighbours' links, leaving no cue. Supported:
the OR rule absorbs one-sided link loss, healing repairs missing pixels when neighbour links
survive, bridging repairs short two-sided gaps. Whether a trained model's errors look like this
is untested.
