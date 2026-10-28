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

## Results
(pending)
