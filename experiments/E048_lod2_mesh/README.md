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

## Results
(pending)
