# E029: City-Scale road graphs, data preparation

Status: **data preparation only** (2026-10-27); no experiment or decision of its own.

`prepare.py` converts City-Scale (Sat2Graph's 20 cities, 2048 x 2048 px at 1 m, from the
RNGDet++ release) into pickle-free files: one `region_<id>.npz` per region with the graph's node
coordinates `xy` (pixel centres, `(x, y)`), `edges`, and the annotated `overpass`, `complicated`
and `parallel` points, plus `split.json` (144 train / 9 valid / 27 test). The source graphs are
Python pickles; they are read with an unpickler that refuses every global, so no code from the
download runs.

    python -I prepare.py "<raw>/cityscale/data" "<out>"

Used by E044 / E044b (`E044_cityscale_unified/data.py` reads the npz files and `split.json`)
for the roads decoder (`graph_from_road_heads`, R4).
