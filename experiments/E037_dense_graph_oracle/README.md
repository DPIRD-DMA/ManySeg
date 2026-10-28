# E037: dense pixel graph (P8) and mask-supported local links (R3), oracle (pre-registered 2026-10-28 10:00)

Why: the user proposed a dense pixel-graph output ([B, 11, H, W]: node confidence, 8-neighbour
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
the lanes still touched). With these rules R3 matched P8r on the 4 smoke tiles.
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

## Compute
CPU only, systemd MemoryMax <= 12 GB, OMP_NUM_THREADS=1, <= 6 processes (E036 shares the box).

## Results
(pending)
