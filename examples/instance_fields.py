"""Field delineation with boundary + distance heads (runs on CPU with random weights).

extent / boundary / distance heads -> targets from an instance-ID mask -> a few training steps ->
instances via seeded growth (or watershed with `pip install manyseg[instances]`).
Losses here are plain torch; manyseg-bench has Dice+BCE etc. (`manyseg_bench.losses`).
"""

import torch
import torch.nn.functional as F
from manyseg import create_model
from manyseg.instances import instance_heads, instance_targets, instances_from_outputs


def synthetic_fields(n=4, size=96, seed=0):
    g = torch.Generator().manual_seed(seed)
    masks = torch.zeros(n, size, size, dtype=torch.long)
    for i in range(n):
        k = 1
        for y in range(0, size, 24):
            for x in range(0, size, 32):  # touching fields on a grid, random cuts
                cut = int(torch.randint(8, 16, (1,), generator=g))
                masks[i, y:y + 24, x:x + cut] = k
                masks[i, y:y + 24, x + cut:x + 32] = k + 1
                k += 2
    img = torch.rand(n, 3, size, size, generator=g) * 0.2
    img += (masks % 5).float()[:, None] / 5  # each field a different brightness
    return img, masks


def main():
    img, masks = synthetic_fields()
    model = create_model(backbone="resnet18", pretrained=False,
                         heads=instance_heads(decoders={"boundary": "unet"}))  # own decoder
    t = instance_targets(masks, boundary_width=1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(30):
        out = model(img)
        loss = (F.binary_cross_entropy_with_logits(out["extent"], t["extent"])
                + F.binary_cross_entropy_with_logits(out["boundary"], t["boundary"],
                                                     pos_weight=torch.tensor(5.0))
                + F.l1_loss(out["distance"] * t["extent"], t["distance"]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0:
            print(f"step {step} loss {loss.item():.3f}")
    model.eval()
    with torch.no_grad():
        out = model(img)
    labels = instances_from_outputs(out["extent"], out["boundary"], out["distance"],
                                    logits=True, backend="torch")
    true = [int((masks[i].unique() > 0).sum()) for i in range(len(masks))]
    print("predicted (30 steps, not converged):", [int(lab.max()) for lab in labels])
    print("true instances per image:           ", true)
    # Perfect outputs (the targets) round-trip to the true instances:
    exact = instances_from_outputs(t["extent"], t["boundary"], t["distance"], backend="torch")
    print("round-trip from targets:", [int(exact[i].max()) for i in range(len(exact))])


if __name__ == "__main__":
    main()
