"""Tasks: how targets, losses and evaluation are derived for a model's outputs.

The trainer is task-agnostic: it calls ``task.loss(outputs, task.targets(y))`` and feeds
``task.evaluator()``. ``y`` is the label array delivered by the loader (class ids for semantic
segmentation, instance ids for instance/point tasks); targets are generated on the device.

Reduced-resolution heads (``manyseg`` head ``scale`` < 1) are detected from the output size:
an output of ``ceil(H / f) x ceil(W / f)`` for ``H x W`` labels has head factor ``f``.
Evaluation is always at input resolution (predictions upsampled with
``manyseg.utils.upsample_output``) so that arms with different scales are comparable.
Semantic losses are computed at head scale against mode-downsampled labels; instance and
point tasks upsample the outputs and keep their full-resolution targets.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from . import losses
from .evaluation import ConfusionMatrix, boundary_f1
from .evaluation.objects import CountMetrics, ObjectMetrics, PointMetrics


def head_factor(out, y) -> int:
    """``f`` such that a pixel output ``(..., h, w)`` is the ``ceil(H / f) x ceil(W / f)`` grid of
    labels ``(..., H, W)``; 1 for full-resolution outputs."""
    if isinstance(out, dict):
        fs = {head_factor(v, y) for v in out.values() if v.ndim == 4}
        if len(fs) > 1:
            raise ValueError(f"outputs on different grids (head factors {sorted(fs)}); "
                             "a task needs one grid")
        return fs.pop() if fs else 1
    (h, w), (H, W) = out.shape[-2:], y.shape[-2:]
    for f in (1, 2, 4, 8):
        if (h, w) == (-(-H // f), -(-W // f)):
            return f
    raise ValueError(f"output {(h, w)} is not a head grid of labels {(H, W)} "
                     "(expected ceil(H / f) x ceil(W / f), f in 1, 2, 4, 8)")


def to_label_grid(out, y):
    """Bilinearly upsample (scaled) pixel outputs to the label grid; full-res outputs pass."""
    from manyseg.utils import upsample_output

    f = head_factor(out, y)
    if f == 1:
        return out
    size = tuple(y.shape[-2:])
    if isinstance(out, dict):
        return {k: upsample_output(v.float(), size, 1 / f) if v.ndim == 4 else v
                for k, v in out.items()}
    return upsample_output(out.float(), size, 1 / f)


class Task:
    name = "task"
    primary_metric = ""
    higher_is_better = True

    def heads(self) -> dict | None:
        """Head configs for providers that support named heads (None = single logits output)."""
        return None

    def targets(self, y):
        return y

    def loss(self, out, tgt):  # pragma: no cover - interface
        raise NotImplementedError

    def evaluator(self):  # pragma: no cover - interface
        raise NotImplementedError


class SemanticTask(Task):
    name = "semantic"
    primary_metric = "miou"

    def __init__(self, num_classes: int, ignore_index: int | None = None,
                 label_smoothing: float = 0.0, native_metrics: bool = False) -> None:
        self.num_classes, self.ignore_index = num_classes, ignore_index
        self.label_smoothing = label_smoothing
        self.native_metrics = native_metrics

    def head_targets(self, out, y):
        """Labels on the output's grid (mode downsampling, ignore_index respected)."""
        from manyseg.utils import downsample_labels

        f = head_factor(out, y)
        if f == 1:
            return y
        return downsample_labels(y, 1 / f, self.ignore_index, num_classes=self.num_classes)

    def loss(self, out, y):
        ignore = self.ignore_index if self.ignore_index is not None else -100
        return F.cross_entropy(out.float(), self.head_targets(out, y), ignore_index=ignore,
                               label_smoothing=self.label_smoothing)

    def evaluator(self):
        return _SemanticEval(self)


class _SemanticEval:
    """Metrics at input resolution (scaled logits are upsampled bilinearly first). With
    ``native_metrics``, also mIoU / boundary F1 on the head grid against mode-downsampled
    labels (``"native"``)."""

    def __init__(self, task: SemanticTask) -> None:
        self.t = task
        self.cm = self.native = None
        self.bf1 = self.loss = self.native_bf1 = 0.0
        self.n = self.px = 0
        self.factor = None

    def update(self, out, y) -> None:
        f = head_factor(out, y)
        if self.factor is None:
            self.factor = f
        if self.t.native_metrics:
            if self.native is None:
                self.native = ConfusionMatrix(self.t.num_classes, self.t.ignore_index,
                                              device=y.device)
            ys, ps = self.t.head_targets(out, y), out.float().argmax(1)
            self.native.update(ps, ys)
            self.native_bf1 += float(boundary_f1(ps, ys, ignore_index=self.t.ignore_index).sum())
        logits = to_label_grid(out, y).float()
        if self.cm is None:
            self.cm = ConfusionMatrix(self.t.num_classes, self.t.ignore_index, device=y.device)
        ignore = self.t.ignore_index if self.t.ignore_index is not None else -100
        self.loss += float(F.cross_entropy(logits, y, ignore_index=ignore, reduction="sum"))
        pred = logits.argmax(1)
        self.cm.update(pred, y)
        self.bf1 += float(boundary_f1(pred, y, ignore_index=self.t.ignore_index).sum())
        self.n += y.shape[0]
        self.px += y.numel()

    def compute(self) -> dict:
        out = self.cm.compute()
        out["boundary_f1"] = self.bf1 / max(1, self.n)
        out["loss"] = self.loss / max(1, self.px)
        out["head_factor"] = self.factor or 1
        if self.native is not None:
            nat = self.native.compute()
            out["native"] = {"miou": nat["miou"], "iou": nat["iou"],
                             "boundary_f1": self.native_bf1 / max(1, self.n)}
        return out


class InstanceTask(Task):
    """Boundary + distance instances (``manyseg.instances``). Primary metric: PQ.

    ``mode="extent_cc"`` is the simple baseline: an extent head only, instances = connected
    components of the thresholded extent (touching objects merge).
    """

    name = "instances"
    primary_metric = "pq"

    def __init__(self, num_classes: int = 1, boundary_width: float = 1.0,
                 distance: str = "normalized", boundary_pos_weight: float | None = None,
                 weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
                 iou_threshold: float = 0.5, decoders=None, mode: str = "boundary_distance",
                 ) -> None:
        if mode not in ("boundary_distance", "extent_cc"):
            raise ValueError(f"mode must be boundary_distance|extent_cc, got {mode!r}")
        self.mode, self.decoders = mode, decoders
        if num_classes != 1:
            raise NotImplementedError("multi-class instance training: use semantic + instances")
        self.num_classes, self.boundary_width, self.distance = num_classes, boundary_width, distance
        self.pos_weight, self.weights = boundary_pos_weight, weights
        self.iou_threshold = iou_threshold

    def heads(self):
        from manyseg.instances import instance_heads

        heads = instance_heads(self.num_classes, distance=self.distance, decoders=self.decoders)
        return {"extent": heads["extent"]} if self.mode == "extent_cc" else heads

    def targets(self, y):
        from manyseg.instances import instance_targets

        t = instance_targets(y, self.boundary_width, distance=self.distance)
        t["instances"] = y
        return t

    def loss(self, out, t):
        out = to_label_grid(out, t["instances"])
        we, wb, wd = self.weights
        if self.mode == "extent_cc":
            return we * losses.dice_bce(out["extent"], t["extent"])
        return (we * losses.dice_bce(out["extent"], t["extent"])
                + wb * losses.dice_bce(out["boundary"], t["boundary"], pos_weight=self.pos_weight)
                + wd * losses.masked_regression(out["distance"], t["distance"], t["extent"] > 0))

    def evaluator(self):
        return _InstanceEval(self)


class _InstanceEval:
    def __init__(self, task: InstanceTask) -> None:
        self.t, self.om, self.loss, self.n = task, ObjectMetrics(task.iou_threshold), 0.0, 0

    def update(self, out, y) -> None:
        from manyseg.instances import instances_from_outputs

        t = self.t.targets(y)
        out = to_label_grid(out, y)
        self.loss += float(self.t.loss(out, t)) * y.shape[0]
        self.n += y.shape[0]
        if self.t.mode == "extent_cc":
            from manyseg.instances import connected_components, relabel_sequential

            labels = relabel_sequential(connected_components(out["extent"][:, 0].float() > 0))
        else:
            labels = instances_from_outputs(out["extent"].float(), out["boundary"].float(),
                                            out["distance"].float(), logits=True)
        self.om.update(labels, y)

    def compute(self) -> dict:
        return {**self.om.compute(), "loss": self.loss / max(1, self.n)}


class PointTask(Task):
    """Centre heatmap (+ offsets) or density from instance-id labels (centroids = points).

    Primary metric: point F1 within ``radius`` (heatmap) or count MAE (density, lower=better).
    """

    name = "points"

    def __init__(self, offsets: bool = False, density: bool = False, sigma: float = 2.0,
                 heatmap: str = "regression", radius: float = 3.0, peak_threshold: float = 0.5,
                 density_scale: float = 100.0) -> None:
        self.offsets, self.density, self.sigma, self.kind = offsets, density, sigma, heatmap
        self.radius, self.peak_threshold, self.density_scale = radius, peak_threshold, density_scale
        self.primary_metric = "mae" if density else "f1"
        self.higher_is_better = not density

    def heads(self):
        from manyseg.points import density_head, point_heads

        return density_head() if self.density else point_heads(self.kind, offsets=self.offsets)

    def targets(self, y):
        from manyseg.points import (
            centers_from_instances,
            density_map,
            heatmap_from_instances,
            offset_targets,
        )

        ignore = y < 0  # e.g. OAM-TCD canopy groups: no individual-tree labels
        y = y.clamp(min=0)
        pts = [torch.floor(c + 0.5) for c, _ in centers_from_instances(y)]
        t = {"points": pts, "instances": y, "valid": ~ignore[:, None]}
        if self.density:
            t["density"] = density_map(pts, tuple(y.shape[-2:]), self.sigma)
        else:
            t["heatmap"] = heatmap_from_instances(y, sigma=self.sigma)
            if self.offsets:
                t["offset"] = offset_targets(y)
        return t

    def loss(self, out, t):
        out = to_label_grid(out, t["instances"])
        valid = t["valid"]
        if self.density:
            return losses.masked_regression(out["density"], t["density"] * self.density_scale,
                                            valid, kind="mse")
        if self.kind == "segmentation":
            hm = losses.heatmap_focal(out["heatmap"].masked_fill(~valid, -20.0),
                                      t["heatmap"])
        else:
            hm = losses.masked_regression(out["heatmap"], t["heatmap"], valid, kind="mse")
        if self.offsets:
            hm = hm + losses.masked_regression(out["offset"], t["offset"],
                                               (t["instances"] > 0)[:, None], kind="l1")
        return hm

    def evaluator(self):
        return _PointEval(self)


class _PointEval:
    def __init__(self, task: PointTask) -> None:
        self.t, self.pm, self.cm, self.loss, self.n = task, PointMetrics(task.radius), \
            CountMetrics(), 0.0, 0

    def update(self, out, y) -> None:
        from manyseg.points import count_from_density, find_peaks

        t = self.t.targets(y)
        out = to_label_grid(out, y)
        self.loss += float(self.t.loss(out, t)) * y.shape[0]
        self.n += y.shape[0]
        gt_counts = torch.tensor([float(len(p)) for p in t["points"]])
        if self.t.density:
            self.cm.update(count_from_density(out["density"].float() * t["valid"]).cpu()
                           / self.t.density_scale, gt_counts)
            return
        hm = out["heatmap"].float()
        hm = hm.sigmoid() if self.t.kind == "segmentation" else hm
        hm = hm * t["valid"]  # no detections inside ignore regions
        peaks = find_peaks(hm, self.t.peak_threshold)
        for (p, _), g in zip(peaks, t["points"]):
            self.pm.update(p.float(), g)
        self.cm.update(torch.tensor([float(len(p)) for p, _ in peaks]), gt_counts)

    def compute(self) -> dict:
        return {**self.pm.compute(), **self.cm.compute(), "loss": self.loss / max(1, self.n)}


def build_task(name: str, num_classes: int, ignore_index: int | None = None, **kwargs) -> Task:
    if name == "semantic":
        return SemanticTask(num_classes, ignore_index, **kwargs)
    if name == "instances":
        return InstanceTask(num_classes=num_classes, **kwargs)
    if name == "points":
        return PointTask(**kwargs)
    raise ValueError(f"unknown task {name!r}; choose from semantic, instances, points")
