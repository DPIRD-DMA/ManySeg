"""Training a dict-output ManySeg model with fastai: a loss and metrics that read the heads by
name, bf16 mixed precision, one-cycle training and an EMA of the weights.

fastai passes the model's output to the loss and metrics unchanged, so a dict works: write
them as ``f(out, *targets)`` and read ``out["name"]``. ``get_preds(reorder=False)`` returns
the dict too (fastai's default ``reorder=True`` fails on dicts). For
code that expects one tensor (metrics slicing ``pred[:, :2]``), wrap the model with
``manyseg.utils.concat_heads`` instead (end of this file).

fastai is optional (``pip install fastai``); without it this script prints a message and exits.
"""

import copy

import torch
import torch.nn.functional as F
from manyseg import create_model
from manyseg.points import find_peaks, gaussian_heatmap
from manyseg.utils import concat_heads
from torch.utils.data import Dataset

try:
    import fastai.callback.fp16  # noqa: F401  (adds Learner.to_bf16)
    import fastai.callback.schedule  # noqa: F401  (adds Learner.fit_one_cycle)
    from fastai.callback.core import Callback
    from fastai.data.core import DataLoaders
    from fastai.learner import Learner
    from fastai.metrics import AvgMetric
except ImportError:
    Callback = None


class Blobs(Dataset):
    """Synthetic images with bright square objects: a foreground mask and a centre heatmap."""

    n_inp = 1  # fastai: items are (input, mask, heatmap); without it two count as inputs

    def __init__(self, n, size=64, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.items = []
        for _ in range(n):
            mask = torch.zeros(size, size)
            pts = []
            for _ in range(int(torch.randint(1, 4, (1,), generator=g))):
                y, x = torch.randint(6, size - 6, (2,), generator=g).tolist()
                mask[y - 4:y + 5, x - 4:x + 5] = 1
                pts.append([y, x])
            img = torch.rand(3, size, size, generator=g) * 0.3 + mask
            heat = gaussian_heatmap([torch.tensor(pts, dtype=torch.float32)], (size, size),
                                    sigma=2.0)[0]
            self.items.append((img, mask.long(), heat))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]  # (input, *targets): fastai splits off the first as xb


def loss_func(out, mask, heat):
    """The loss reads the dict by head name; fastai calls it as loss_func(pred, *yb)."""
    return F.cross_entropy(out["seg"], mask) + 10 * F.mse_loss(out["heatmap"], heat)


def fg_iou(out, mask, heat):
    pred = out["seg"].argmax(1).bool()
    gt = mask.bool()
    return (pred & gt).sum().float() / (pred | gt).sum().clamp(min=1)


def count_mae(out, mask, heat):
    pred = [len(p) for p, _ in find_peaks(out["heatmap"].float(), 0.3, kernel=7)]
    true = [len(p) for p, _ in find_peaks(heat, 0.99, kernel=7)]
    return torch.tensor(pred, dtype=torch.float32).sub(torch.tensor(true)).abs().mean()


class EMA(Callback if Callback is not None else object):
    """Keeps an exponential moving average of the weights (and BatchNorm statistics:
    ``use_buffers=True``). Evaluate or save ``self.ema.module`` after training. Built on the
    unwrapped model in ``before_fit``, so it works with the dict outputs unchanged."""

    order = 65  # after the optimiser step

    def __init__(self, decay=0.99):
        self.decay = decay

    def before_fit(self):
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

        self.ema = AveragedModel(self.learn.model, multi_avg_fn=get_ema_multi_avg_fn(self.decay),
                                 use_buffers=True)

    def after_step(self):
        self.ema.update_parameters(self.learn.model)


def main():
    if Callback is None:
        print("fastai is not installed (pip install fastai): skipping")
        return
    torch.manual_seed(0)
    model = create_model(
        backbone="resnet18",
        pretrained=False,  # True downloads ImageNet weights
        heads={"seg": {"type": "segmentation", "out_channels": 2},
               "heatmap": {"type": "regression", "out_channels": 1}},
    )
    dls = DataLoaders.from_dsets(Blobs(16, seed=0), Blobs(8, seed=1), bs=4, device="cpu")
    ema = EMA()
    learn = Learner(dls, model, loss_func=loss_func, cbs=[ema],
                    metrics=[AvgMetric(fg_iou), AvgMetric(count_mae)])
    if torch.cuda.is_available():
        # fastai's bf16 autocast is CUDA-only; the callback casts outputs (dicts too) to float
        learn = learn.to_bf16()
        learn.dls.cuda()
        learn.model.cuda()
    learn.fit_one_cycle(2, lr_max=1e-3)

    # reorder=False: fastai's reordering can't handle dicts (the validation set isn't shuffled)
    out, (mask, heat) = learn.get_preds(reorder=False)  # out is a dict, targets a tuple
    print({k: tuple(v.shape) for k, v in out.items()})

    # the EMA weights: a copy of the model, still dict-output
    ema_model = copy.deepcopy(ema.ema.module).eval().cpu()
    x = dls.valid.one_batch()[0].cpu()
    with torch.no_grad():
        print("EMA heads:", list(ema_model(x)))

    # one tensor for single-output pipelines: seg logits in channels 0-1, heatmap in 2
    single = concat_heads(ema_model, ["seg", "heatmap"])
    with torch.no_grad():
        print("concatenated:", tuple(single(x).shape), single.channels)
    # Save the model itself, not the wrapper, and with manyseg.save, never torch.save(model):
    # a pickle depends on ManySeg's internal module paths and breaks across versions.
    #   manyseg.save(ema_model, "model.safetensors", activations={"seg": "softmax"},
    #                normalize=...)  # then manyseg.load("model.safetensors")


if __name__ == "__main__":
    main()
