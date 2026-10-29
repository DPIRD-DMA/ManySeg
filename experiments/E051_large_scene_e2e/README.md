# E051: large-scene vectors end to end through ManyPred (2026-10-29)

Validation, not a comparison of methods: does the shipped pipeline (ManyPred `merge="cut"`
stitching with `BlockPostprocess`, then `manyseg.vectors.block_decoder`) give the same vectors as
a whole-scene decode of the same stitched rasters, on real scenes with trained models?

* Roofs: 1 km x 1 km Leidsche Rijn (4096 x 4096 px, 0.25 m, PDOK 2023 ortho + AHN4 nDSM, RD New),
  E047b model; the wrapper also returns the input nDSM as the `height` field, so the roof decoder
  gets heights through the same stream. Decoder: `block_decoder("roofs", **E047b F3 settings)`.
* Roads: City-Scale region 8 (2048 x 2048 px, 1 m), E044 unified model seed 0,
  `block_decoder("roads", **E044 tuned settings)`.
ManyPred tiles 1024 / overlap 256 (E050b), decoder blocks 1024 / halo 256, `workers=0` (the
ManyPred pool hang with torch-using decoders is being fixed separately), float32 weights with
bf16 autocast in the wrapper. ManyPred branch feature/vector-blocks, ManySeg feature/vector-bench.

## Results (`runs/*_result.json`; GPU shared with two other jobs, timings indicative)
| task | scene | hook (predict + decode) | whole-scene decode | block decode (1024/256) | equal |
|---|---|---|---|---|---|
| roofs | 4096 x 4096 | 5537 faces, 92 s | 5537 faces, 32 s | 5537 faces, 85 s | ids, geometry exact, planes max diff 0.0 |
| roads | 2048 x 2048 | 2398 nodes / 2658 edges, 13 s | same, 5.6 s | same, 5.7 s | node and edge arrays equal |

A CPU smoke run (1536 px roof crop) gave the same equality (837 faces). The hook writes a
GeoPackage in EPSG:28992 (2-D faces and 3-D roof mesh). Visual check (decoded vectors over the
image with ManyPred cut lines and decoder block edges; seams invisible): `vis51.py`, figures in
the status page / session scratchpad.
