# E048: joining trained roof faces into a mesh: vertex snapping (pre-registered 2026-10-28 19:20)

Why: E047b's trained faces are decoded per instance, so neighbouring faces almost never share
vertices and E043b's J2 join (which acts on shared vertices) does little; the mesh has cracks.
E047's demo snapped vertices of different faces within 1.5 px (shared vertices 39 -> 154).
This tests snapping before adopting it.

## Setup
E047b's model (`B3_s0_e150`) and its val-tuned F0 / F3 settings; E047's val (59) and test (48)
tiles. Per tile: faces -> [snap] -> robust planes from the input nDSM (E043b) -> J2 mesh.
**Snap**: vertices of different faces within r px are union-found and moved to their cluster
mean; a face whose polygon becomes invalid keeps its original ring. r tuned on val over
{1.0, 1.5, 2.5} by seam closure (F3).

## Metrics
* **Seam closure** (primary): outline points every 0.5 px that lie within 2 px of another face's
  outline; the fraction of them within 0.1 px of it (1 = no cracks between neighbouring faces).
* E043's mesh metrics against 3DBAG planes: vertex z error (m), false step vertices per shared
  vertex; face F1 (E031 metrics) as a guard.

Reference: 3DBAG's own faces score 0.79 seam closure (first 10 test tiles), since faces of
neighbouring buildings can lie 0.1-2 px apart without touching; 1.0 is not the target.

## Decision rule (fixed now)
Snapping adopted (into `manyseg.vectors`) if, for F3 on both test areas: seam closure rises by
>= 0.20, vertex z error worsens by <= 0.05 m and face F1 drops by <= 0.01. F0 reported alongside.

## Results (19:28; 3 min, local)
Val seam closure (F3): r 1.0: 0.349, r 1.5: 0.383, r 2.5: 0.421 -> r = 2.5 px. Test (48 tiles):

| area | faces | join | seam closure | vertex z err m | false steps / shared | shared vertices | face F1 |
|---|---|---|---|---|---|---|---|
| leidscherijn | F0 | J2 | 0.733 | 0.523 | 0.156 | 10690 | 0.574 |
| leidscherijn | F0 | snap+J2 | 0.648 | 0.515 | 0.163 | 3190 | 0.563 |
| leidscherijn | F3 | J2 | 0.384 | 0.683 | 0.150 | 794 | 0.570 |
| leidscherijn | F3 | snap+J2 | 0.490 | 0.657 | 0.181 | 1750 | 0.570 |
| staphorst | F0 | J2 | 0.681 | 0.397 | 0.151 | 13075 | 0.619 |
| staphorst | F0 | snap+J2 | 0.614 | 0.400 | 0.164 | 3250 | 0.611 |
| staphorst | F3 | J2 | 0.385 | 0.510 | 0.167 | 658 | 0.589 |
| staphorst | F3 | snap+J2 | 0.503 | 0.488 | 0.169 | 1797 | 0.589 |

**Decision: snapping not adopted** (F3 seam closure +0.107 / +0.118, below the +0.20 bar; vertex z
error slightly better, face F1 unchanged). Two observations for the next design: (1) F0's
pixel-edge outlines already close most seams (0.68 at staphorst vs 0.79 for 3DBAG's own faces),
because neighbouring instances share exact pixel edges, and snapping makes F0 worse (its staircase
vertices merge); (2) F3's straight walls leave cracks that vertex snapping only partly closes:
closing them needs edge-level joining (a vertex of one face onto the edge of its neighbour, i.e.
T-junctions), or decoding all faces of a building jointly as one planar graph.


## E048b (registered 19:35, after E048's result): planar joining
Edge-level joining for F3: each face is snapped (`shapely.snap`) to the union of its neighbours
within r px; all outlines are noded together (`unary_union`) and re-polygonised; each cell goes
to the original face covering most of it (cells covered < 50% by any face are dropped); a face's
cells are merged. Neighbouring faces then share edges and vertices exactly (a planar partition),
and J2 acts on every shared vertex. r tuned on val over {1.0, 1.5, 2.5} by seam closure.
Rule as E048 against F3 without joining (seam closure +0.20, vertex z error <= +0.05 m, face F1
>= -0.01, both areas); E048's snap shown alongside. Test once (`runs/e048b.json`).

### E048b results
(pending)
