# Plan: making the graph method robust (draft for review, 2026-10-27)

Context: the graph tensor (Sat2Graph-style: vertex map 1 ch, edge 6 ch per 60° sector, vector
12 ch to the neighbour, extent 1 ch; decoder = peaks → follow vectors → gap bridging →
polygonize with tile border → keep faces inside the extent → simplify; code in
`packages/manyseg/src/manyseg/vectors/graph_tensor.py` at the time, now
`experiments/E030_whu_polygons/graph_tensor.py`; training task in
`experiments/E030_whu_polygons/common.py:GraphTensorTask`). Evidence so far: E028 (oracle)
graph best for footprints but all-or-nothing, fails for roads; E031 (oracle LoD2) best roof
structure, loses buildings when the graph does not close; E031b (oracle) a per-building fallback
to plane segmentation passes. E030 (trained WHU footprints, B0 instance masks vs B2 graph) is
running. The training loss is purely per-pixel; nothing penalises an outline that will not
close, although one missed vertex loses a whole building at decode time.

Three experiments, two decoder-only (CPU), one training (GPU).

## E028b (CPU, oracle): decoder-only fixes for footprints
Reuse E028 (OSM footprints, 0.5 m, corruption levels 0–3, innaloo tune, murdoch + northam
test, 5 seeds), arms B0 and B2 frozen at their E028 settings, plus:
* **B2F, per-building fallback:** connected components of the extent mask; a component takes
  B2's faces whose representative point lies in it if they cover ≥ `cov` of it, else B0's
  polygons for it. Tune `cov` ∈ {0.7, 0.8, 0.9, 0.95}.
* **B2M, close with the mask:** for each component not covered ≥ `cov`, add the component's
  boundary (pixel-edge outline, simplified 1 px) to the graph's edges before polygonizing, so
  open chains close along the mask. Tune `cov` as above.
Decision (E028 rule 3 form): an arm is *viable* if it matches the better of B0/B2 on C-IoU and
detection F1 at levels 0–2 (no more than 0.02 below on either test area); it *beats* B2 if
detection F1 is higher by > 2 pooled seed-SDs and ≥ 0.02 at level 1 or 2 with C-IoU no more than
0.02 lower. The best viable arm that beats B2 becomes the default footprint decoder.

## E030b (GPU inference + CPU decoding): the same fixes on the trained E030 models
After E030: for each seed, B2F / B2M using that seed's B2 model for the graph and that seed's B0
model for the fallback (B0's `instances_from_outputs` → `polygons_from_labels`) on WHU val
(tune `cov`) and test. Same rule as E030 (C-IoU primary, detection F1 not more than 0.02 lower)
against both B0 and B2. Two separate models stand in for one model with both head sets.

## E032 (GPU, trained): completeness losses
Data, model, recipe, augmentation, budget and evaluation as E030 (WHU, resnet34 U-Net,
`fastai_like`, 40 epochs, batch 16, dihedral aug, decoder tuned on val). Control: E030's B2
runs (same code path, seeds 0–2; reused if the loss code for the control is byte-identical).
Arm **B2c** = B2's loss plus:
1. **Decode-cost weighting:** vertex BCE weighted 1 + 10·target per pixel; edge BCE with
   pos_weight 5.
2. **Endpoint consistency:** at support pixels with a true edge in sector k, the predicted vector
   (×20 px) from the pixel centre gives an endpoint; bilinearly sample sigmoid(vertex logits)
   there (`grid_sample`) and add −log(sample + 1e-4), mean over such pixel-sectors, weight 1.
   Gradients reach both the vector head and the vertex map.
3. **Degree:** at support pixels, |Σ_k sigmoid(edge_k) − Σ_k edge_target_k|, mean, weight 0.1
   (each vertex should have exactly its true number of edges; 2 on a footprint ring).
(Reciprocity and topology losses on an edge-line map are deferred: the first needs the
opposite-sector bookkeeping, the second an edge-line target that does not exist yet.)
Decision: B2c **wins** if its test detection F1 beats B2's by > 2 pooled seed-SDs and ≥ 0.02,
with C-IoU no more than 0.02 lower; then the completeness losses become the default in the B2
training recipe. Budget: 3 runs × ~65 min train + ~10 min eval on the 4090, queued after E030,
in the 12 GB capped service.

## Open questions for the reviewer
1. Are the B2F / B2M rules sound, and is a per-building fallback a fair comparison when B0 and
   B2 come from two separately trained models (E030b)?
2. Is the endpoint-consistency loss well-posed (gradients through `grid_sample` coordinates
   while the vertex map is also being learned; degenerate solutions, e.g. a blurry vertex map
   everywhere)? Better formulation?
3. Degree loss on soft probabilities: sensible, or does it fight the per-sector BCE?
4. Anything that would make E032 unfair to the control (reusing E030's B2 runs)?
5. What should be cut or added?
