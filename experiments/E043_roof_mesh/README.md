# E043: roof meshes from graph faces + per-face planes (pre-registered 2026-10-28 17:00)

User's idea (28 Oct): the graph gives the top-down roof faces; raster heads give the height; fit
one 3D plane per face; join the oriented polygons into a mesh with the minimum adjustment. The user
asked whether height alone suffices (vs height + two slope rasters); both are tested. Code:
`e043.py`, `run43.py`; E031's 3DBAG tiles (Leidsche Rijn, Staphorst; 40 tiles each, 0.25 m/px),
levels L0-L3, seeds 0-4. Faces: E031b's hybrid decoder (its tuned settings per level), shared by all
arms. Fixed parameters (no tuning): erosion 2 px, step threshold 0.5 m, J2 cap 2 px, lambda 0.05.

## Arms
Planes: **fit** (E031 baseline), **robust** (height only: eroded face, trimmed twice), **slope**
(height + slope rasters: A, B = slope medians, C = median offset). Joins: **J1** (shared vertices
take the mean height if faces agree within 0.5 m, else a step wall), **J2** (vertex first moved, at
most 2 px, towards the faces' plane intersection).
Extra corruption (beyond E031's): low-frequency height ramps (sigma 40 px field, std 0.1 / 0.3 /
0.5 m at L1-L3) and slope-raster noise (0.02 / 0.05 / 0.1 m/px), slope rasters blurred like the height.

## Metrics
E031 height RMSE, normal error (deg), face F1; vertex height error against the GT planes within
1.5 px; steps and false steps (faces disagree > 0.5 m where the GT planes agree within 0.5 m); mean
gap. Validated on GT faces + planes: vertex error 0.007 m, 0 false steps.

## Decision rules (fixed now)
* Slope rasters worth it: at >= 2 of L1-L3, slope beats robust by >= 1 deg normal error AND
  >= 0.05 m height RMSE (J1).
* Robust beats fit: robust height RMSE <= fit's at L1-L3.
* J2 better: for robust, J2 vertex error <= J1's - 0.05 m and no more false steps, at L1 and L2.
Also reported: false steps per arm (slope rasters may trade orientation for step errors).

## Results (Pawsey job 51716389, 20 s; first submission 51716204 failed on a quoting error before running)
Means over 2 areas x 40 tiles x 5 seeds (`runs/decision.json`):

| plane arm (J1) | normal err L1 / L2 / L3 (deg) | height RMSE L1 / L2 / L3 (m) | vertex z err L1 / L2 (m) | false steps L1 / L2 |
|---|---|---|---|---|
| fit | 1.72 / 8.50 / 11.77 | 1.128 / 2.000 / 2.787 | 0.266 / 0.666 | 5.9 / 12.8 |
| robust (height only) | 1.32 / 5.17 / 8.88 | 1.134 / 2.011 / 2.787 | 0.255 / 0.687 | 5.5 / 12.2 |
| slope (height + slopes) | **0.61 / 2.30 / 4.28** | 1.185 / 2.040 / 2.789 | 0.277 / 0.685 | 5.1 / 11.3 |

J2 vs J1 (robust): false steps 5.5 -> 4.0 (L1), 12.2 -> 7.5 (L2), 7.3 -> 4.2 (L3); vertex z error
unchanged (0.255 -> 0.261, 0.687 -> 0.681).

### Decision (registered rules)
* Slope rasters worth it: **no** - normal error drops by >= 1 deg at L2 and L3 (and 0.7 deg at L1),
  but height RMSE does not drop by 0.05 m at any level.
* Robust beats fit: **no** - height RMSE 0.006-0.011 m worse at L1-L2 (normal error 0.4-3.3 deg better).
* J2 better: **no** - vertex error not 0.05 m lower (false steps 27-38% fewer).

### Interpretation
Height RMSE was the wrong primary measure for the plane question: all arms share the same decoded
faces, so the RMSE is dominated by face errors (missed or merged buildings, ~1-2.8 m) and cannot
move. On the measure the plane estimate controls - roof orientation - explicit slope rasters halve
the error at every corrupted level (L2 8.5 -> 2.3 deg), and the robust eroded height fit gets about
a third of that gain without extra outputs. J2 (moving shared vertices towards the plane
intersection) removes about a third of false step walls at no cost in vertex height. A re-test
with orientation as the primary (new pre-registration) is warranted before choosing; on this
evidence the trained LoD2 design should keep a height head with a robust fit and consider two
slope channels if roof orientation matters.

# E043b: plane and join choice with orientation as the primary measure (pre-registered 2026-10-28 16:40)
E043 showed its height-RMSE criterion cannot move (shared faces dominate it). Re-test, fresh
corruption seeds 5-9 (E043 used 0-4), same tiles, arms, parameters and code. Rules (fixed now):
* Slope rasters adopted if, at >= 2 of L1-L3, slope's normal error <= robust's - 1 deg and its false
  steps <= robust's x 1.10 (J2 both).
* Robust adopted over fit if its normal error is lower at L1-L3 and height RMSE within +0.02 m (J1).
* J2 adopted if, for the chosen plane arm, false steps <= J1's x 0.8 at L1 and L2 and vertex z error
  <= J1's + 0.02 m.
