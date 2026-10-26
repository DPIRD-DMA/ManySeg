"""Training a two-head ManySeg model with Lightning: a ``LightningModule`` that builds the model
from its arguments, a loss that reads the heads by name, one-cycle LR, bf16 mixed precision
on GPU, a checkpoint reloaded with ``load_from_checkpoint``, then ``manyseg.save``.

Build the model inside the module from plain arguments (not by passing an ``nn.Module`` in):
``save_hyperparameters`` records them, so a checkpoint is enough to rebuild it.

Lightning is optional (``pip install lightning``); without it this script prints a message and
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
    import lightning as L
except ImportError:
    L = None


class Blobs(Dataset):
    """Synthetic images with bright square objects: a foreground mask and the object count."""

    def __init__(self, n, size=64, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.items = []
        for _ in range(n):
            mask = torch.zeros(size, size, dtype=torch.long)
            count = int(torch.randint(1, 4, (1,), generator=g))
            for _ in range(count):
                y, x = torch.randint(6, size - 6, (2,), generator=g).tolist()
                mask[y - 4:y + 5, x - 4:x + 5] = 1
            img = torch.rand(3, size, size, generator=g) * 0.3 + mask
            self.items.append({"image": img, "mask": mask, "count": torch.tensor([count / 3])})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class SegModule(L.LightningModule if L is not None else object):
    def __init__(self, backbone="resnet18", pretrained=False, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model(
            backbone=backbone, pretrained=pretrained,
            heads={"seg": {"type": "segmentation", "out_channels": 2},
                   "count": {"type": "regression", "out_channels": 1, "scope": "image"}},
        )

    def forward(self, x):
        return self.model(x)

    def step(self, batch, stage):
        out = self(batch["image"])  # {"seg": (B, 2, H, W), "count": (B, 1)}
        loss = F.cross_entropy(out["seg"], batch["mask"]) + F.mse_loss(out["count"],
                                                                        batch["count"])
        pred, gt = out["seg"].argmax(1).bool(), batch["mask"].bool()
        iou = (pred & gt).sum() / (pred | gt).sum().clamp(min=1)  # a mean of per-batch IoUs:
        # torchmetrics.JaccardIndex gives the exact dataset IoU. sync_dist averages over GPUs.
        self.log_dict({f"{stage}_loss": loss, f"{stage}_iou": iou}, prog_bar=True,
                      sync_dist=stage != "train")
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.hparams.lr, total_steps=self.trainer.estimated_stepping_batches)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


def main():
    if L is None:
        print("lightning is not installed (pip install lightning): skipping")
        return
    L.seed_everything(0, verbose=False)
    train = DataLoader(Blobs(16, seed=0), batch_size=4, shuffle=True)
    valid = DataLoader(Blobs(8, seed=1), batch_size=4)

    with tempfile.TemporaryDirectory() as tmp:
        trainer = L.Trainer(
            max_epochs=2,
            accelerator="auto",
            precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
            default_root_dir=tmp, logger=False, enable_progress_bar=False,
            enable_model_summary=False,
        )
        module = SegModule()
        trainer.fit(module, train, valid)
        print({k: round(v.item(), 3) for k, v in trainer.callback_metrics.items()})

        # the checkpoint holds the arguments, so the model is rebuilt without the code above
        ckpt = trainer.checkpoint_callback.best_model_path
        restored = SegModule.load_from_checkpoint(ckpt, map_location="cpu").eval()
        print("restored from", Path(ckpt).name, "heads:", list(restored.model.heads))

        # save the ManySeg model (module.model), not the LightningModule
        path = manyseg.save(restored.model, Path(tmp) / "blobs.safetensors", normalize=[],
                            activations={"seg": "softmax"})
        print("saved", path.name)


if __name__ == "__main__":
    main()
