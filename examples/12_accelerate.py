"""The plain training loop (10_pytorch.py) under Hugging Face Accelerate: the same loop runs on
CPU, one GPU or several (``accelerate launch 12_accelerate.py``), with bf16 on GPU.

Four changes from a plain loop: ``accelerator.prepare`` the model, optimiser, scheduler and
loaders; ``accelerator.backward(loss)``; ``gather_for_metrics`` for validation across
processes; and ``unwrap_model`` before saving, so ``manyseg.save`` gets the ManySeg model
rather than a DDP or compile wrapper.

Accelerate is optional (``pip install accelerate``); without it this script prints a message and
exits.
"""

import tempfile
from pathlib import Path

import manyseg
import torch
import torch.nn.functional as F
from manyseg import create_model
from torch.utils.data import DataLoader, Dataset

try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
except ImportError:
    Accelerator = None


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


def main():
    if Accelerator is None:
        print("accelerate is not installed (pip install accelerate): skipping")
        return
    set_seed(0)
    accelerator = Accelerator(mixed_precision="bf16" if torch.cuda.is_available() else "no")
    model = create_model(backbone="resnet18", pretrained=False, in_channels=4, num_classes=2)
    train = DataLoader(Blobs(16, seed=0), batch_size=4, shuffle=True)
    valid = DataLoader(Blobs(8, seed=1), batch_size=4)
    epochs = 2
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3,
                                                total_steps=epochs * len(train))
    model, opt, sched, train, valid = accelerator.prepare(model, opt, sched, train, valid)

    for epoch in range(epochs):
        model.train()
        for x, y in train:  # already on the right device
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

        model.eval()
        inter = union = 0
        with torch.no_grad():
            for x, y in valid:
                pred, y = accelerator.gather_for_metrics((model(x).argmax(1), y))
                inter += (pred.bool() & y.bool()).sum().item()
                union += (pred.bool() | y.bool()).sum().item()
        accelerator.print(f"epoch {epoch}: loss {loss.item():.3f}, "
                          f"val IoU {inter / max(union, 1):.3f}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:  # one process writes the file
        with tempfile.TemporaryDirectory() as tmp:
            path = manyseg.save(accelerator.unwrap_model(model),
                                Path(tmp) / "blobs.safetensors", normalize=[],
                                activations={"logits": "softmax"})
            print("saved", path.name)


if __name__ == "__main__":
    main()
