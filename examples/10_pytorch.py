"""Training with a plain PyTorch loop: AdamW, a one-cycle schedule, bf16 autocast on GPU,
gradient clipping, a validation IoU, then ``manyseg.save`` and ``manyseg.load``.

A ManySeg model is an ordinary ``nn.Module``: nothing here is ManySeg-specific except building
and saving it. With several heads the model returns a dict; index it by head name in the loss.
"""

import tempfile
from pathlib import Path

import manyseg
import torch
import torch.nn.functional as F
from manyseg import create_model
from torch.utils.data import DataLoader, Dataset


class Blobs(Dataset):
    """Synthetic 4-band images with bright square objects and their foreground masks."""

    def __init__(self, n, size=64, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.items = []
        for _ in range(n):
            mask = torch.zeros(size, size, dtype=torch.long)
            for _ in range(int(torch.randint(1, 4, (1,), generator=g))):
                y, x = torch.randint(6, size - 6, (2,), generator=g).tolist()
                mask[y - 4:y + 5, x - 4:x + 5] = 1
            self.items.append((torch.rand(4, size, size, generator=g) * 0.3 + mask, mask))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    inter = union = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu().bool()
        inter += (pred & y.bool()).sum().item()
        union += (pred | y.bool()).sum().item()
    return inter / max(union, 1)


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(backbone="resnet18", pretrained=False, in_channels=4,
                         num_classes=2).to(device)
    train = DataLoader(Blobs(16, seed=0), batch_size=4, shuffle=True)
    valid = DataLoader(Blobs(8, seed=1), batch_size=4)

    epochs = 2
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3,
                                                total_steps=epochs * len(train))
    # bf16 needs no GradScaler; with float16 use torch.amp.GradScaler
    autocast = torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda")

    for epoch in range(epochs):
        model.train()
        for x, y in train:
            x, y = x.to(device), y.to(device)
            with autocast:
                loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        iou = evaluate(model, valid, device)
        print(f"epoch {epoch}: loss {loss.item():.3f}, val IoU {iou:.3f}")

    # manyseg.save, not torch.save(model): it stores the build arguments and weights, and
    # what a user of the file needs to run it (input scaling, how logits become predictions)
    with tempfile.TemporaryDirectory() as tmp:
        path = manyseg.save(model, Path(tmp) / "blobs.safetensors",
                            normalize=[],  # these inputs are used as is
                            activations={"logits": "softmax"}, classes=["background", "blob"])
        loaded = manyseg.load(path, map_location=device)
        x = next(iter(valid))[0].to(device)
        model.eval()
        with torch.no_grad():
            assert torch.allclose(loaded(x), model(x), atol=1e-5)
        print("saved and reloaded:", path.name)


if __name__ == "__main__":
    main()
