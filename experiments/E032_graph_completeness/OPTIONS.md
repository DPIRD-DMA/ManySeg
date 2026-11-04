# Options for vector footprints / roofs (for review, 2026-10-27 12:30)

## Evidence so far
* E028 oracle (OSM, synthetic corruptions): Sat2Graph-style graph tensor (B2: vertex map, 6
  sector edge probs, 12 vector channels, extent; outline densified to ≤ 20 px) gives the best
  footprint polygons up to moderate corruption but fails all-or-nothing (a missed vertex breaks
  two links, the ring opens, the building is lost). For roads it fails (missed vertices break
  routes).
* E031 oracle (3DBAG LoD2): roof graph finds faces far better than plane segmentation of a height
  model, but buildings whose graph does not close get no roof. E031b: per-building fallback to
  plane segmentation passes.
* E030 trained WHU (0.3 m), seed 0, test: B0 (instance heads → pixel-edge polygons) C-IoU 0.709,
  detection F1 0.850, mIoU 0.861, PoLiS 0.39 m, N-ratio 1.18; B2 C-IoU 0.613, F1 0.750, mIoU
  0.854, PoLiS 0.39 m, N-ratio 0.49. (GT polygons come from raster labels simplified by 1 px and
  keep staircase jogs on rotated buildings, which inflates GT vertex counts and C-IoU's penalty
  for clean predictions.) B2's tuned vertex threshold is at the grid edge (0.1).
* Trained B2 seed 0 on 200 val tiles (threshold 0.2): recall within 2 px 0.50 at "corners"
  (53,674, most are raster staircase jogs) vs 0.40 at densified wall points (10,348); of the
  spurious peaks (> 2 px from any target vertex), 15,783 lie on a true wall (< 1.5 px from the
  outline) and 6,192 elsewhere. The densified points are arbitrary positions along a wall
  (they depend on the wall's whole length), and the loss penalises a geometrically equivalent
  vertex placed elsewhere on the same wall.

## Options discussed with the maintainer
1. **B2t, wall-tolerant vertex loss:** keep B2 (densified points for the vectors) but do not
   penalise vertex-map predictions on a true wall; real corners stay exact targets.
2. **B2c, corners only:** no densification; vectors reach the next real corner (long vectors).
3. **Every-pixel outline + direction (Frame Field Learning-like):** outline map + per-pixel tangent
   direction (and mask); decode by tracing the outline along the direction, then simplify
   (Douglas–Peucker) to recover corners.
4. **B3, corners + lines (user's proposal):** outline map (every outline pixel is "part of a
   line"), a corner heatmap ("I am a corner", true corners only), a per-outline-pixel direction
   (encoded orientation-invariantly, e.g. (cos 2θ, sin 2θ)), and the extent mask. Decode: corner
   peaks → from each corner trace along outline pixels following the direction to the next corner
   → straight edges between true corners → polygonize, keep faces inside the mask. Claimed
   benefits: no arbitrary targets; a missed corner merges two walls (simpler shape) instead of
   losing the building; small outline gaps bridged along the direction; corners are the polygon.
   Junctions (shared walls, roof ridges) are degree-3 corners.
5. Already queued and running: E032 completeness losses (B2w: decode-cost weighting; B2e: + a
   target-sampled endpoint loss), E033 (B2d: distance-weighted vertex BCE; B2s: snap offset head
   to the nearest true vertex), each with seed 0 first and seeds 1–2 only if seed-0 val
   C-IoU × F1 beats B2 by > 0.005. Also E030b (per-building fallback to B0 on the trained models).

## Queue (one RTX 4090, shared; CPU-bound: load ~45 on 32 cores)
Running now: E030 seeds 1–2 (B0_s1 → B2_s1 → B0_s2 → B2_s2, ~6 h) in parallel with the screening
queue (B2e_s0 [running] → B2w_s0 → B2d_s0 → B2s_s0 → seeds 1–2 of advanced arms). Each B2-type run
currently takes ~2–2.3 h under contention. After E030: E030b + control re-evaluation (~2 h).

## Questions for the reviewer
1. Rank options 1–4 (and the queued E032/E033 arms) by expected benefit for WHU footprints and
   for LoD2 roofs; which are redundant with each other; what is missing (e.g. HiSup, FFL details,
   PolyWorld, a decoder fix like E030b's fallback).
2. Is B3 sound? Failure modes (corner over/under-detection, tracing across adjacent buildings or
   along shared walls, curved buildings, T-junctions, tracing loops, raster staircase GT), how to
   encode direction, and what the decoder must do. Is it materially different from FFL / HiSup?
3. Given the E030 seed-0 result (B2 clearly below B0) and the vertex analysis, should the queue be
   reordered: e.g. stop or deprioritise E030 seeds 1–2, drop some queued B2 variants, and spend the
   GPU on an oracle-validated B3 instead? What is the most informative next GPU-hour?
4. Is the C-IoU comparison against staircase raster GT fair to graph methods; should the GT be
   regularised differently (or evaluated with a vertex-count-insensitive metric as co-primary)?
