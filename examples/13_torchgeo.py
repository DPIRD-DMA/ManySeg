"""A ManySeg model inside TorchGeo's semantic segmentation task.

TorchGeo's task builds an SMP model in ``configure_models``; overriding that one method swaps
in a ManySeg model and keeps everything else: its losses (``loss="ce" | "focal" | "dice" |
...``), torchmetrics, AdamW + ReduceLROnPlateau, ``freeze_backbone``, Lightning checkpoints,
and its datamodules, which yield the same ``{"image", "mask"}`` batches as the toy dataset here.
``model`` names a ManySeg decoder (``"unet"``, ``"fpn"``, ``"upernet"``, ``"segformer"``, ...)
and ``backbone`` any timm name.

TorchGeo is optional (``pip install torchgeo``, Python >= 3.12 for current releases); without
it this script prints a message and exits.
"""

import tempfile
from pathlib import Path

import manyseg
import torch
from manyseg import create_model
from torch.utils.data import DataLoader, Dataset

try:
    import lightning as L

    try:
        from torchgeo.tasks import SemanticSegmentation
    except ImportError:  # torchgeo < 0.10
        from torchgeo.trainers import SemanticSegmentationTask as SemanticSegmentation
except ImportError:
    SemanticSegmentation = None


class ManySegSegmentation(SemanticSegmentation if SemanticSegmentation is not None else object):
    """TorchGeo's segmentation task with a ManySeg model. ``weights=True`` loads the backbone's
    timm weights; a ``manyseg.weights`` tag such as ``"ssl4eo_s2_moco"`` loads those. TorchGeo's
    own weight enums and checkpoint paths are for its SMP models and are refused."""

    def configure_models(self):
        weights = self.weights
        if not (weights is None or isinstance(weights, (bool, str))) or (
                isinstance(weights, str) and Path(weights).exists()):
            raise ValueError(f"weights={weights!r}: pass True, False or a manyseg.weights tag")
        decoder = {"deeplabv3+": "deeplabv3plus"}.get(self.hparams["model"],
                                                       self.hparams["model"])
        self.model = create_model(
            backbone=self.hparams["backbone"],
            pretrained=weights if isinstance(weights, str) else bool(weights),
            in_channels=self.hparams["in_channels"],
            decoder=decoder,
            num_classes=self.hparams["num_classes"] or self.hparams["num_labels"] or 1,
        )
        if self.hparams["freeze_backbone"]:
            self.model.backbone.requires_grad_(False)
        if self.hparams["freeze_decoder"]:
            self.model.decoder.requires_grad_(False)


class Blobs(Dataset):
    """Synthetic 4-band images with bright square objects, as TorchGeo samples."""

    def __init__(self, n, size=64, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.items = []
        for _ in range(n):
            mask = torch.zeros(size, size, dtype=torch.long)
            for _ in range(int(torch.randint(1, 4, (1,), generator=g))):
                y, x = torch.randint(6, size - 6, (2,), generator=g).tolist()
                mask[y - 4:y + 5, x - 4:x + 5] = 1
            img = torch.rand(4, size, size, generator=g) * 0.3 + mask
            self.items.append({"image": img, "mask": mask})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def main():
    if SemanticSegmentation is None:
        print("torchgeo is not installed (pip install torchgeo): skipping")
        return
    L.seed_everything(0, verbose=False)
    task = ManySegSegmentation(model="unet", backbone="resnet18", weights=False,
                               in_channels=4, num_classes=2, loss="ce", lr=1e-3)
    train = DataLoader(Blobs(16, seed=0), batch_size=4, shuffle=True)
    valid = DataLoader(Blobs(8, seed=1), batch_size=4)

    with tempfile.TemporaryDirectory() as tmp:
        trainer = L.Trainer(max_epochs=2, accelerator="auto", default_root_dir=tmp,
                            logger=False, enable_progress_bar=False, enable_model_summary=False)
        trainer.fit(task, train, valid)
        print({k: round(v.item(), 3) for k, v in trainer.callback_metrics.items()
               if k in ("train_loss", "val_loss", "val_AverageJaccardIndex")})

        # TorchGeo's predict_step: softmax probabilities per pixel
        probs = trainer.predict(task, valid)[0]["probabilities"]
        print("probabilities:", tuple(probs.shape))

        # the trained network is task.model, a plain ManySeg model
        path = manyseg.save(task.model, Path(tmp) / "blobs.safetensors", normalize=[],
                            activations={"logits": "softmax"})
        print("saved", path.name)


if __name__ == "__main__":
    main()
