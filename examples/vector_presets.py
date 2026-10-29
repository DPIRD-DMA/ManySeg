"""Training the vector-output presets (runs on CPU with random weights).

Corner-line instances (footprints, fields, roof faces) and unified roads: head preset -> targets
encoded from geometry (polygons, a road graph) -> a random dihedral transform per step, with the
direction fields transformed too -> a few training steps -> vector decoding.

The losses here are short versions of manyseg-bench's ``CornerLineTask`` and ``RoadGraphTask``
(``manyseg_bench.tasks``), the exact losses of the trained models (E036, E044). With the bench,
a dataset whose cache holds these targets as a label dict trains either preset through
``run_training`` (``TASK = "corner_lines"`` or ``"roads"``), which augments with the same exact
dihedral transforms (``GPULoader(dihedral=True, directions=("direction",))``).
"""

import torch
import torch.nn.functional as F
from manyseg import create_model


def focal(logits, target, min_pos=1):
    """CenterNet focal loss on Gaussian heatmaps; positives are the 3 x 3 peaks above 0.5."""
    pos = (target > 0.5) & (target == F.max_pool2d(target, 3, 1, 1))
    p = logits.sigmoid().clamp(1e-6, 1 - 1e-6)
    loss = -((1 - p) ** 2 * p.log() * pos + (1 - target) ** 4 * p ** 2 * (1 - p).log() * ~pos)
    return loss.sum() / pos.sum().clamp(min=min_pos)


def masked_l1(pred, target, mask):
    m = mask.expand_as(pred).float()
    return ((pred - target).abs() * m).sum() / m.sum().clamp(min=1)


def dice_bce(logits, target):
    p = logits.sigmoid()
    dice = 1 - (2 * (p * target).sum((-2, -1)) + 1) / (p.sum((-2, -1)) + target.sum((-2, -1)) + 1)
    return F.binary_cross_entropy_with_logits(logits, target) + dice.mean()


def augment(d, image, targets, directions=("direction",)):
    """One dihedral transform ``d`` (0..7) of an image and its target dict."""
    from manyseg.vectors import dihedral

    return dihedral(image, d), {k: dihedral(v, d, direction=k in directions)
                                for k, v in targets.items()}


def corner_lines(size=64, steps=15):
    import shapely
    from manyseg.instances import instance_targets, instances_from_outputs
    from manyseg.vectors import corner_line_heads, corner_targets, polygons_from_corners
    from rasterio.features import rasterize
    from rasterio.transform import Affine

    polys = [shapely.affinity.rotate(shapely.box(8, 10, 26, 22), 20, origin="centroid"),
             shapely.Polygon([(36, 8), (56, 8), (56, 30), (46, 30), (46, 18), (36, 18)]),
             shapely.box(12, 38, 30, 56), shapely.box(40, 40, 56, 54)]
    ids = rasterize([(p, k + 1) for k, p in enumerate(polys)], out_shape=(size, size),
                    transform=Affine.identity(), dtype="int32")
    ids = torch.from_numpy(ids).long()
    image = torch.rand(3, size, size) * 0.2 + (ids > 0).float() * (0.5 + 0.1 * (ids % 3))
    ct = corner_targets(polys, (size, size))  # corner heatmap, wall directions, outline mask
    targets = {"instances": ids, **{k: torch.from_numpy(v) for k, v in ct.items()}}
    # Perfect outputs round-trip: the instances and the target maps decode to straight walls
    exact = polygons_from_corners(ids.numpy(), ct["corner"], ct["direction"])
    print("corner lines: vertices per polygon from the targets:",
          [len(q.exterior.coords) - 1 for q in exact])

    model = create_model(backbone="resnet18", pretrained=False, heads=corner_line_heads())
    bias = float(model.heads.corner.conv.bias.detach())
    print(f"corner head bias {bias:.3f}: starts at p = 0.1 (CenterNet prior)")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(0)
    for step in range(steps):
        batch = [augment(int(torch.randint(8, (1,), generator=g)), image, targets)
                 for _ in range(2)]
        x = torch.stack([b[0] for b in batch])
        t = {k: torch.stack([b[1][k] for b in batch]) for k in targets}
        it = instance_targets(t["instances"], boundary_width=2)
        out = model(x)
        loss = (dice_bce(out["extent"], it["extent"]) + dice_bce(out["boundary"], it["boundary"])
                + masked_l1(out["distance"], it["distance"], it["extent"] > 0)
                + focal(out["corner"], t["corner"][:, None])
                + masked_l1(out["direction"], t["direction"], t["direction_mask"][:, None]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 5 == 0:
            print(f"  step {step} loss {loss.item():.2f}")
    model.eval()
    with torch.no_grad():
        out = model(image[None])
    labels = instances_from_outputs(out["extent"], out["boundary"], out["distance"],
                                    logits=True)[0]
    pred = polygons_from_corners(labels, out["corner"][0, 0].sigmoid(), out["direction"][0])
    print(f"  {steps} steps (not converged): {len(pred)} polygons, true {len(polys)}")


def roads(size=64, steps=15):
    import networkx as nx
    import numpy as np
    from manyseg.vectors import graph_from_road_heads, road_heads, road_targets

    g = nx.Graph()
    for n, xy in {"w": (0, 30), "j": (30, 31), "e": (64, 26), "s": (31, 64), "end": (40, 8),
                  "a": (50, 0), "b": (58, 64)}.items():
        g.add_node(n, xy=np.array(xy, float))
    g.add_edges_from([("w", "j"), ("j", "e"), ("j", "s"), ("j", "end")], width=6.0)
    g.add_edge("a", "b", width=8.0)  # an overpass: crosses the other roads without a node
    crossing = [(53.4, 27.6)]
    rt = road_targets(g, (size, size), crossings=crossing)  # mask, line, keypoints, direction
    targets = {k: torch.from_numpy(v) for k, v in rt.items()}
    image = torch.rand(3, size, size) * 0.2 + targets["mask"] * 0.6
    exact = graph_from_road_heads(rt["mask"], rt["keypoints"], rt["direction"], min_length=5)
    at_crossing = np.hypot(*(exact.xy - crossing[0]).T) < 6
    print(f"roads: graph decoded from the targets: {len(exact.edges)} edges, "
          f"{int((exact.degree[at_crossing] >= 3).sum())} junctions at the overpass")

    model = create_model(backbone="resnet18", pretrained=False, heads=road_heads())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(0)
    for step in range(steps):
        batch = [augment(int(torch.randint(8, (1,), generator=gen)), image, targets)
                 for _ in range(2)]
        x = torch.stack([b[0] for b in batch])
        t = {k: torch.stack([b[1][k] for b in batch]) for k in targets}
        out = model(x)
        d = t["direction"]
        loss = (dice_bce(out["mask"], t["mask"][:, None])
                + dice_bce(out["line"], t["line"][:, None])
                + focal(out["keypoints"], t["keypoints"], min_pos=16)  # floor: rare crossings
                + masked_l1(out["direction"][:, :2], d[:, :2], t["mask"][:, None] > 0.5)
                + masked_l1(out["direction"][:, 2:], d[:, 2:], d[:, 2:].abs().sum(1, True) > 0))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 5 == 0:
            print(f"  step {step} loss {loss.item():.2f}")
    model.eval()
    with torch.no_grad():
        out = model(image[None])
    iou = ((out["mask"][0, 0] > 0) & (targets["mask"] > 0.5)).sum() / (
        (out["mask"][0, 0] > 0) | (targets["mask"] > 0.5)).sum().clamp(min=1)
    print(f"  {steps} steps (not converged): road mask IoU {float(iou):.2f}")


def main():
    try:
        import networkx  # noqa: F401
        import rasterio  # noqa: F401
        import shapely  # noqa: F401
    except ImportError:
        print("skipping: needs pip install 'manyseg[vectors]' rasterio")
        return
    torch.manual_seed(0)
    corner_lines()
    roads()


if __name__ == "__main__":
    main()
