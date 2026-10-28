# E038: B3 vs masks on WHU with a staircase-free score (pre-registered 2026-10-28 12:00)

Why: E036's screen used C-IoU against WHU's raster-traced GT, whose staircase vertices reward
copying staircases. Post hoc (E036 README) B3 had the most accurate outlines on every
vertex-free measure but lost on C-IoU's vertex count. 42% of B3's mask failures touch the tile
edge, where outlines are artificial cuts. This experiment fixes the score *before* looking at
any new result: it re-scores existing models (no retraining) and adds seeds by the rule below.
Approved by the user 2026-10-28 11:54.

## Arms (decoders tuned on val, test once)
* **B0**: E030 B0 instance thresholds (0.6 / 0.5 / 0.15, as registered for B3), Douglas-Peucker
  simplify tuned over {1, 1.5, 2, 3} px.
* **B0 + reg**: B0 (its tuned simplify) then buildingregulariser 0.2.5 (parallel_threshold 3 px,
  simplify_tolerance 1.5 px, no circles), tuned over parallel_threshold {2, 3} x simplify_tolerance {1, 1.5, 2}.
* **B3**: E036's decoder, tuned over its 16 registered settings.
Models: B0_s0 (E030), B3_s0 (E036). Seeds 1 (B3_s1 trained now; B0_s1 exists) follow the rule below.

## Ground truth and scores
GT: WHU outlines simplified with Douglas-Peucker at 1.5 px (straight lines, no forced angles).
**Primary: boundary F** per split, pooled over tiles: points every 0.5 px along every GT and
predicted outline, a point matched if the other set's outlines lie within 1 px (0.3 m); points
within 1 px of the tile edge (artificial cuts) are excluded from both sets. Misses lower recall,
false buildings lower precision.
Secondary: detection F1 (IoU > 0.5), C-IoU x F1 and matched IoU against the DP GT, PoLiS,
N-ratio; all also for interior buildings only (not touching the tile edge).

## Decision rule (fixed now)
Seed 0, test: B3 advances if its boundary F beats both B0 and B0 + reg by >= 0.01 and its
detection F1 is no more than 0.03 below B0's. Then seed 1 for B3 (and B0_s1, already trained):
B3 is preferred if the mean paired boundary-F advantage over the better baseline is >= 0.01 on
both seeds. Otherwise B0 + reg stays the footprint default.

## Compute
CPU decoding (6 processes, MemoryMax 14 GB) + one GPU pass per model; B3_s1 training ~1.5 h.

## Results
(pending)
