# E054: keeping courtyards in corner-decoded footprints (pre-registered 2026-11-02 12:00)

Why: the Codex release review (2026-11-02) found that the B3 decoder fills holes:
`polygons_from_corners` straightens only an instance's exterior ring, so with perfect targets a
3,200 px courtyard building decodes as 3,600 px with no interior, although `footprint_targets`
keeps courtyards. The user asked for this to be fixed, not documented. New option `holes`: an
instance's holes of at least `min_area` (16 px) are cut out of the straightened polygon as their
traced outline simplified by `simplify` px. `corner_targets` puts no corners or directions on
hole walls, so holes are not straightened. Risk on trained models: a predicted instance mask can
contain spurious holes, which would add false outline and lower precision.

## Design
No training. E036 `B3_s0`, `B3_s1` (WHU; E038's metrics), `holes` False (today) vs True at the
shipped settings: 0.6 / 0.5 / 0.15, `min_seed` 8, corners 0.5, big_dev 4, agree 0.4, merge 8,
fixed (no tuning), val and test. These test sets were used before for these checkpoints (E038,
E053); this is a new comparison of a decoder option at fixed settings, not a re-selection. Roof
faces stay `holes=False`: they join and mesh on their outer walls.

## Decision rule (fixed now)
On test, mean over the two seeds, holes True - False: `holes=True` becomes the footprint default
if boundary F >= -0.005 and detection F1 >= -0.005. Otherwise footprints keep filled holes and
the limitation is documented.

## Compute
One GPU pass per model and split, about 20 min in all (GPU shared, timings not reported); CPU
decoding under MemoryMax 10 GB.

## Results (12:04; `runs/e054.json`; GPU shared, timings not reported)
Test, `holes` False -> True at the shipped settings. False reproduces E053's `min_seed=8` run.

| checkpoint | boundary F | boundary P | boundary R | det F1 | holes (GT 692) |
|---|---|---|---|---|---|
| B3_s0 | 0.6580 -> 0.6577 | 0.6625 -> 0.6614 | 0.6536 -> 0.6539 | 0.8544 -> 0.8544 | 4 -> 264 |
| B3_s1 | 0.6623 -> 0.6619 | 0.6657 -> 0.6644 | 0.6590 -> 0.6593 | 0.8517 -> 0.8517 | 4 -> 317 |

Mean True - False: boundary F -0.0004, detection F1 -0.00002; val agrees (-0.0001, -0.0004).
**Decision (rule above): `holes=True` is the footprint default.** Holes are about 0.0002 of the
outline here (692 GT holes among about 50,000 buildings), so the score barely moves either way:
this keeps the geometry the targets promise, it does not improve the score. Rendered tiles with
GT courtyards show large courtyards kept and some thin slivers between wings missed; the
models find about 40% as many holes as the GT has, and how many of those match a GT hole was not
measured. (The 4 holes with `holes=False` come from `buffer(0)` repairing an invalid straightened
polygon, as before.)
