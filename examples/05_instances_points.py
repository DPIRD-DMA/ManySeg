"""Instances (boundary + distance heads) and points (centre heatmap + density heads).

Targets come from an instance-ID mask; post-processing runs outside the model (and outside
any exported graph). Longer walkthroughs: `instance_fields.py`, `point_counting.py`.
"""

import torch
import torch.nn.functional as F

from manyseg import create_model
from manyseg.instances import instance_heads, instance_targets, instances_from_outputs
from manyseg.points import (
    centers_from_instances,
    count_from_density,
    density_head,
    density_map,
    find_peaks,
    heatmap_from_instances,
    point_heads,
)


def toy_instances(n=2, size=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.zeros(n, size, size, dtype=torch.long)
    for i in range(n):
        for k in range(1, int(torch.randint(3, 7, (1,), generator=g)) + 1):
            y, x = torch.randint(0, size - 10, (2,), generator=g).tolist()
            ids[i, y:y + 9, x:x + 9] = k  # later objects may overlap earlier ones
    img = torch.rand(n, 3, size, size, generator=g) * 0.2 + (ids > 0).float()[:, None]
    return img, ids


def main():
    img, ids = toy_instances()
    model = create_model(
        backbone="resnet18", pretrained=False,  # True downloads ImageNet weights
        heads={**instance_heads(decoders={"boundary": "unet"}),  # extent, boundary, distance
               **point_heads(), **density_head()})               # heatmap, density
    t = instance_targets(ids, boundary_width=1)  # extent, boundary, distance: (B, 1, H, W)
    pts = [c for c, _ in centers_from_instances(ids)]  # (K, 2) float (y, x) per image
    hm = heatmap_from_instances(ids, sigma=2.0)
    den = density_map(pts, tuple(ids.shape[-2:]), sigma=2.0)  # sums to the count

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(3):
        out = model(img)
        loss = (F.binary_cross_entropy_with_logits(out["extent"], t["extent"])
                + F.binary_cross_entropy_with_logits(out["boundary"], t["boundary"])
                + F.l1_loss(out["distance"] * t["extent"], t["distance"])
                + F.mse_loss(out["heatmap"], hm) + F.mse_loss(out["density"], den))
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        out = model(img)
    labels = instances_from_outputs(out["extent"], out["boundary"], out["distance"],
                                    logits=True, backend="torch")  # (B, H, W) ids
    # Calibrate the peak threshold on validation data: MSE heatmaps rarely reach 1, and a
    # fixed 0.5 cost recall (E011b: test F1 0.40 -> 0.59 with a val-chosen 0.15-0.2).
    peaks = find_peaks(out["heatmap"], threshold=0.2)  # per image: (points (K, 2), scores)
    print("3 steps from random weights, so predictions are not meaningful:")
    print("  true objects:  ", [len(p) for p in pts])
    print("  instances:     ", [int(lab.max()) for lab in labels])
    print("  peaks:         ", [len(p) for p, _ in peaks])
    print("  density counts:", [round(c, 1) for c in count_from_density(out["density"]).tolist()])

    # Perfect predictions (the targets) recover the objects exactly:
    exact = instances_from_outputs(t["extent"], t["boundary"], t["distance"], backend="torch")
    print("round trip from targets: instances", [int(e.max()) for e in exact],
          "| peaks", [len(p) for p, _ in find_peaks(hm, threshold=0.5)],
          "| density", [round(c, 2) for c in count_from_density(den).tolist()])


if __name__ == "__main__":
    main()
