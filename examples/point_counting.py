"""Counting and locating objects with centre-heatmap (+ offset) and density heads (CPU).

Targets come from instance masks (centroids) or point lists; post-processing is outside the
model: peaks via max-pool NMS, count = density integral, instances via offset grouping.
"""

import torch
import torch.nn.functional as F

from manyseg import create_model
from manyseg.points import (
    centers_from_instances,
    count_from_density,
    density_head,
    density_map,
    find_peaks,
    group_by_offsets,
    heatmap_from_instances,
    offset_targets,
    point_heads,
)


def synthetic_objects(n=4, size=96, seed=0):
    g = torch.Generator().manual_seed(seed)
    masks = torch.zeros(n, size, size, dtype=torch.long)
    for i in range(n):
        for k in range(1, int(torch.randint(4, 10, (1,), generator=g)) + 1):
            y, x = torch.randint(4, size - 12, (2,), generator=g).tolist()
            masks[i, y:y + 7, x:x + 7] = k  # small "trees" / "sheds"
    img = torch.rand(n, 3, size, size, generator=g) * 0.2 + (masks > 0).float()[:, None]
    return img, masks


def main():
    img, masks = synthetic_objects()
    model = create_model(backbone="resnet18", pretrained=False,
                         heads={**point_heads(offsets=True), **density_head()})
    pts = [torch.floor(c + 0.5) for c, _ in centers_from_instances(masks)]
    hm_t = heatmap_from_instances(masks, sigma=2.0)
    off_t = offset_targets(masks)
    den_t = density_map(pts, (96, 96), sigma=2.0) * 100  # scaled for optimisation
    fg = (masks > 0)[:, None].float()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(40):
        out = model(img)
        loss = (F.mse_loss(out["heatmap"], hm_t) + F.mse_loss(out["density"], den_t)
                + (F.l1_loss(out["offset"], off_t, reduction="none") * fg).sum() / fg.sum())
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        out = model(img)
    peaks = find_peaks(out["heatmap"], threshold=0.3)
    print("(40 steps from random weights: predictions are not converged)")
    print("true counts:   ", [len(p) for p in pts])
    print("peak counts:   ", [len(p) for p, _ in peaks])
    print("density counts:", [round(float(c) / 100, 1) for c in count_from_density(out["density"])])
    inst = group_by_offsets([p.float() for p, _ in peaks], out["offset"], masks > 0)
    print("instances from offsets (image 0):", int(inst[0].max()))


if __name__ == "__main__":
    main()
