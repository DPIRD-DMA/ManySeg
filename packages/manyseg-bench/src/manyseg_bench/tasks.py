"""Tasks: how targets, losses and evaluation are derived for a model's outputs.

The trainer is task-agnostic: it calls ``task.loss(outputs, task.targets(y))`` and feeds
``task.evaluator()``. ``y`` is the label array delivered by the loader (class ids for semantic
segmentation, instance ids for instance/point tasks); targets are generated on the device.

Reduced-resolution heads (``manyseg`` head ``scale`` < 1) are detected from the output size:
an output of ``ceil(H / f) x ceil(W / f)`` for ``H x W`` labels has head factor ``f``. Losses
are computed on the head's own grid against targets downsampled by meaning
(``Task.head_targets``):

* semantic labels: block majority (``ignore_index`` respected);
* instance extent: block majority; boundary: block "any" (``downsample_max``; a 1-2 px band
  would vanish under a majority vote); distance: mean over the block's instance pixels, with
  the loss restricted to blocks containing any instance pixel;
* point heatmap: block max (keeps each centre's peak of 1, which the focal loss keys on);
  offsets: mean over the block's instance pixels divided by ``f`` (head-pixel offsets to the
  centroid, exactly); density: block sum (counts preserved);
* validity (ignore regions) for points: the fraction of valid pixels in each block. Losses
  weight blocks by it, counting integrates ``density * fraction``, and peaks are suppressed
  only in blocks without any valid pixel.

Evaluation compares at input resolution so arms with different scales are comparable:
semantic logits and instance outputs are upsampled (``manyseg.utils.upsample_output``) before
metrics; points are picked on the native heatmap and mapped to input-pixel coordinates
(``find_peaks(scale=1/f)``); counts integrate the native density.
"""

from __future__ import annotations

import math

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


def _with_scale(heads: dict, scale) -> dict:
    """Task head configs with a head ``scale`` (validated; 1 leaves the configs unchanged)."""
    from manyseg.utils import scale_factor

    if scale_factor(scale) == 1:
        return heads
    return {k: {**v, "scale": scale} for k, v in heads.items()}


class Task:
    name = "task"
    primary_metric = ""
    higher_is_better = True
    image_level = False  # one label per image (classification / regression)
    # Label entries holding (cos 2t, sin 2t) direction channel pairs. The loader then augments
    # with exact dihedral transforms (``GPULoader(dihedral=True)``) and transforms their values.
    direction_targets: tuple[str, ...] = ()

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
                 head_scale: float = 1) -> None:
        if mode not in ("boundary_distance", "extent_cc"):
            raise ValueError(f"mode must be boundary_distance|extent_cc, got {mode!r}")
        self.mode, self.decoders = mode, decoders
        self.head_scale = head_scale
        _with_scale({}, head_scale)  # validate early
        if num_classes != 1:
            raise NotImplementedError("multi-class instance training: use semantic + instances")
        self.num_classes, self.boundary_width, self.distance = num_classes, boundary_width, distance
        self.pos_weight, self.weights = boundary_pos_weight, weights
        self.iou_threshold = iou_threshold

    def heads(self):
        from manyseg.instances import instance_heads

        heads = instance_heads(self.num_classes, distance=self.distance, decoders=self.decoders)
        heads = {"extent": heads["extent"]} if self.mode == "extent_cc" else heads
        return _with_scale(heads, self.head_scale)

    def targets(self, y):
        from manyseg.instances import instance_targets

        t = instance_targets(y, boundary_width=self.boundary_width, distance=self.distance)
        t["instances"] = y
        return t

    def head_targets(self, out, t):
        """Targets on the outputs' grid (see the module docstring)."""
        from manyseg.utils import downsample_labels, downsample_max, downsample_mean

        f = head_factor(out, t["instances"])
        if f == 1:
            return {"extent": t["extent"], "boundary": t["boundary"],
                    "distance": t["distance"], "distance_mask": t["extent"] > 0}
        s, fg = 1 / f, (t["instances"] > 0)[:, None]
        return {"extent": downsample_labels(t["extent"].long(), s).float(),
                "boundary": downsample_max(t["boundary"], s),
                "distance": downsample_mean(t["distance"], s, valid=fg),
                "distance_mask": downsample_max(fg, s)}

    def loss(self, out, t):
        nt = self.head_targets(out, t)
        we, wb, wd = self.weights
        if self.mode == "extent_cc":
            return we * losses.dice_bce(out["extent"], nt["extent"])
        return (we * losses.dice_bce(out["extent"], nt["extent"])
                + wb * losses.dice_bce(out["boundary"], nt["boundary"], pos_weight=self.pos_weight)
                + wd * losses.masked_regression(out["distance"], nt["distance"],
                                                nt["distance_mask"]))

    def evaluator(self):
        return _InstanceEval(self)


class _InstanceEval:
    def __init__(self, task: InstanceTask) -> None:
        self.t, self.om, self.loss, self.n = task, ObjectMetrics(task.iou_threshold), 0.0, 0

    def update(self, out, y) -> None:
        from manyseg.instances import instances_from_outputs

        t = self.t.targets(y)
        self.loss += float(self.t.loss(out, t)) * y.shape[0]   # on the head's grid
        self.n += y.shape[0]
        out = to_label_grid(out, y)                             # metrics at input resolution
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
    ``head_scale`` sets every head's output scale; ``decoders`` as in
    :func:`manyseg.points.point_heads` (for density: a decoder name/config, or "separate").
    """

    name = "points"

    def __init__(self, offsets: bool = False, density: bool = False, sigma: float = 2.0,
                 heatmap: str = "regression", radius: float = 3.0, peak_threshold: float = 0.5,
                 calibrate: bool = True,
                 thresholds: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6,
                                                  0.7),
                 density_scale: float = 100.0, head_scale: float = 1, decoders=None) -> None:
        self.offsets, self.density, self.sigma, self.kind = offsets, density, sigma, heatmap
        self.head_scale, self.decoders = head_scale, decoders
        _with_scale({}, head_scale)  # validate early
        self.radius, self.peak_threshold, self.density_scale = radius, peak_threshold, density_scale
        # calibrate=True: evaluation also scores a grid of peak thresholds; the runner picks
        # the best on *val* and evaluates test at that single threshold. Default since E011b
        # (test F1 0.40 -> 0.59 vs a fixed 0.5); calibrate=False keeps peak_threshold.
        self.calibrate, self.thresholds = calibrate, tuple(sorted(thresholds))
        self.primary_metric = "mae" if density else "f1"
        self.higher_is_better = not density

    def heads(self):
        from manyseg.points import density_head, point_heads

        if self.density:
            d = self.decoders
            d = d.get("density") if isinstance(d, dict) else ("unet" if d == "separate" else d)
            heads = density_head(d)
        else:
            heads = point_heads(self.kind, offsets=self.offsets, decoders=self.decoders)
        return _with_scale(heads, self.head_scale)

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

    def head_targets(self, out, t):
        """Targets on the outputs' grid (see the module docstring). ``weight`` is the fraction
        of valid pixels per output pixel, ``valid`` = weight > 0, ``fg`` = any instance pixel."""
        from manyseg.utils import downsample_max, downsample_mean, downsample_sum

        f = head_factor(out, t["instances"])
        fg = (t["instances"] > 0)[:, None]
        if f == 1:
            nt = {k: t[k] for k in ("density", "heatmap", "offset") if k in t}
            return {**nt, "weight": t["valid"].float(), "valid": t["valid"], "fg": fg}
        s = 1 / f
        w = downsample_mean(t["valid"].float(), s)
        nt = {"weight": w, "valid": w > 0, "fg": downsample_max(fg, s)}
        if "density" in t:
            nt["density"] = downsample_sum(t["density"], s)
        if "heatmap" in t:
            nt["heatmap"] = downsample_max(t["heatmap"], s)
        if "offset" in t:
            nt["offset"] = downsample_mean(t["offset"], s, valid=fg) / f
        return nt

    def loss(self, out, t, nt=None):
        nt = self.head_targets(out, t) if nt is None else nt
        if self.density:
            return losses.masked_regression(out["density"], nt["density"] * self.density_scale,
                                            nt["weight"], kind="mse")
        if self.kind == "segmentation":
            hm = losses.heatmap_focal(out["heatmap"].masked_fill(~nt["valid"], -20.0),
                                      nt["heatmap"])
        else:
            hm = losses.masked_regression(out["heatmap"], nt["heatmap"], nt["weight"], kind="mse")
        if self.offsets:
            hm = hm + losses.masked_regression(out["offset"], nt["offset"], nt["fg"], kind="l1")
        return hm

    def find_points(self, out, t, nt=None):
        """Peaks of the native heatmap in input-pixel coordinates, none in invalid blocks."""
        from manyseg.points import find_peaks

        f = head_factor(out, t["instances"])
        nt = self.head_targets(out, t) if nt is None else nt
        hm = out["heatmap"].float()
        hm = hm.sigmoid() if self.kind == "segmentation" else hm
        thr = min(self.peak_threshold, *self.thresholds) if self.calibrate else \
            self.peak_threshold
        return find_peaks(hm * nt["valid"], thr, scale=1 / f)

    def evaluator(self):
        return _PointEval(self)


class _PointEval:
    def __init__(self, task: PointTask) -> None:
        self.t, self.pm, self.cm, self.loss, self.n = task, PointMetrics(task.radius), \
            CountMetrics(), 0.0, 0
        self.grid = {thr: PointMetrics(task.radius) for thr in task.thresholds} \
            if task.calibrate and not task.density else {}  # density has no peaks

    def update(self, out, y) -> None:
        from manyseg.points import count_from_density

        t = self.t.targets(y)
        nt = self.t.head_targets(out, t)
        self.loss += float(self.t.loss(out, t, nt)) * y.shape[0]
        self.n += y.shape[0]
        gt_counts = torch.tensor([float(len(p)) for p in t["points"]])
        if self.t.density:  # integrate on the native grid (counts are block sums)
            self.cm.update(count_from_density(out["density"].float() * nt["weight"]).cpu()
                           / self.t.density_scale, gt_counts)
            return
        peaks = self.t.find_points(out, t, nt)  # input-pixel coordinates (lowest grid threshold)
        thr = self.t.peak_threshold
        for (p, s), g in zip(peaks, t["points"]):
            self.pm.update(p[s >= thr].float(), g)
            for gthr, pm in self.grid.items():
                pm.update(p[s >= gthr].float(), g)
        self.cm.update(torch.tensor([float((s >= thr).sum()) for _, s in peaks]), gt_counts)

    def compute(self) -> dict:
        out = {**self.pm.compute(), **self.cm.compute(), "loss": self.loss / max(1, self.n),
               "threshold": self.t.peak_threshold}
        if self.grid:
            f1 = {thr: pm.compute()["f1"] for thr, pm in self.grid.items()}
            out["f1_by_threshold"] = f1
            out["best_threshold"] = max(f1, key=lambda k: (f1[k], -k))
        return out


# --- image-level tasks ------------------------------------------------------------------------
#
# Labels are one per image: class ids ``(B,)`` (classification), multi-hot ``(B, K)``
# (multilabel) or floats ``(B, K)`` (regression). Models output ``(B, K)``: a tensor, or a
# dict whose ``"logits"`` entry is used (ManySeg named heads). Providers build image-level
# models when ``ModelSpec.output == "image"``.

def _image_out(out):
    return out["logits"] if isinstance(out, dict) else out


class ClassificationTask(Task):
    """Single-label (softmax + cross-entropy; primary: accuracy) or multi-label (sigmoid + BCE;
    primary: macro average precision) image classification."""

    image_level = True

    def __init__(self, num_classes: int, multilabel: bool = False,
                 label_smoothing: float = 0.0) -> None:
        self.num_classes, self.multilabel = num_classes, multilabel
        self.label_smoothing = label_smoothing
        self.name = "multilabel" if multilabel else "classification"
        self.primary_metric = "map_macro" if multilabel else "accuracy"

    def loss(self, out, y):
        out = _image_out(out).float()
        if self.multilabel:
            return F.binary_cross_entropy_with_logits(out, y.float())
        return F.cross_entropy(out, y, label_smoothing=self.label_smoothing)

    def evaluator(self):
        return _MultiLabelEval(self) if self.multilabel else _ClassEval(self)


class _ClassEval:
    def __init__(self, task: ClassificationTask) -> None:
        self.t, self.cm, self.loss, self.n = task, None, 0.0, 0

    def update(self, out, y) -> None:
        out = _image_out(out).float()
        k = self.t.num_classes
        if self.cm is None:
            self.cm = torch.zeros(k, k, dtype=torch.long, device=y.device)
        self.cm += torch.bincount(y * k + out.argmax(1), minlength=k * k).view(k, k)
        self.loss += float(F.cross_entropy(out, y, reduction="sum"))
        self.n += y.shape[0]

    def compute(self) -> dict:
        cm = self.cm.double()  # rows: truth, columns: prediction
        tp, support, predicted = cm.diag(), cm.sum(1), cm.sum(0)
        f1 = (2 * tp / (support + predicted).clamp(min=1)).tolist()
        recall = tp / support.clamp(min=1)
        present = support > 0
        return {"accuracy": float(tp.sum() / cm.sum().clamp(min=1)),
                "macro_f1": sum(f1) / len(f1), "f1": f1,
                "balanced_accuracy": float(recall[present].mean()),
                "confusion": self.cm.tolist(), "loss": self.loss / max(1, self.n)}


def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Non-interpolated AP (mean precision at each positive, scores ranked high to low; ties
    keep input order). NaN without positives."""
    order = torch.argsort(scores, descending=True, stable=True)
    hits = labels[order].double()
    if hits.sum() == 0:
        return float("nan")
    prec = hits.cumsum(0) / torch.arange(1, len(hits) + 1, dtype=torch.double, device=hits.device)
    return float((prec * hits).sum() / hits.sum())


class _MultiLabelEval:
    def __init__(self, task: ClassificationTask) -> None:
        self.t, self.scores, self.labels, self.loss, self.n = task, [], [], 0.0, 0

    def update(self, out, y) -> None:
        out = _image_out(out).float()
        self.loss += float(F.binary_cross_entropy_with_logits(out, y.float(), reduction="sum")
                           / out.shape[1])
        self.n += y.shape[0]
        self.scores.append(out.cpu())
        self.labels.append(y.cpu())

    def compute(self) -> dict:
        s, y = torch.cat(self.scores), torch.cat(self.labels)
        ap = [average_precision(s[:, k], y[:, k]) for k in range(s.shape[1])]
        valid = [a for a in ap if not math.isnan(a)]
        pred, pos = s > 0, y > 0
        tp, fp, fn = (pred & pos).sum(0).double(), (pred & ~pos).sum(0).double(), \
            (~pred & pos).sum(0).double()
        f1 = (2 * tp / (2 * tp + fp + fn).clamp(min=1)).tolist()
        return {"map_macro": sum(valid) / max(1, len(valid)),
                "map_micro": average_precision(s.flatten(), y.flatten()), "ap": ap,
                "classes_without_positives": len(ap) - len(valid),
                "f1_micro": float(2 * tp.sum() / (2 * tp + fp + fn).sum().clamp(min=1)),
                "f1_macro": sum(f1) / len(f1), "f1": f1, "loss": self.loss / max(1, self.n)}


class RegressionTask(Task):
    """Image-level regression of ``num_outputs`` targets. Primary: RMSE (lower is better)."""

    name = "regression"
    primary_metric = "rmse"
    higher_is_better = False
    image_level = True

    def __init__(self, num_outputs: int, loss: str = "mse") -> None:
        if loss not in ("mse", "l1", "smooth_l1"):
            raise ValueError(f"loss must be mse|l1|smooth_l1, got {loss!r}")
        self.num_outputs, self.kind = num_outputs, loss

    def loss(self, out, y):
        fn = {"mse": F.mse_loss, "l1": F.l1_loss, "smooth_l1": F.smooth_l1_loss}[self.kind]
        return fn(_image_out(out).float(), y.float())

    def evaluator(self):
        return _RegressionEval(self)


class _RegressionEval:
    def __init__(self, task: RegressionTask) -> None:
        self.t, self.pred, self.y = task, [], []

    def update(self, out, y) -> None:
        self.pred.append(_image_out(out).float().cpu())
        self.y.append(y.float().cpu())

    def compute(self) -> dict:
        p, y = torch.cat(self.pred).double(), torch.cat(self.y).double()
        err = p - y
        ss_tot = ((y - y.mean(0)) ** 2).sum(0)
        r2 = (1 - (err ** 2).sum(0) / ss_tot.clamp(min=1e-12)).tolist()
        return {"mae": float(err.abs().mean()), "rmse": float(err.pow(2).mean().sqrt()),
                "rmse_per_output": err.pow(2).mean(0).sqrt().tolist(), "r2": r2,
                "r2_mean": sum(r2) / len(r2), "loss": float(self.t.loss(p.float(), y.float()))}


class SegMultilabelTask(Task):
    """Segmentation + multi-label classification from one class mask: the image labels are
    the classes present (``ignore_index`` excluded), so pixel supervision is an auxiliary task
    for the image classifier. Heads: ``seg`` (pixel, via the decoder) and ``logits`` (image,
    pooling the encoder). Loss: CE(seg) + ``cls_weight`` * BCE(logits). Primary: macro AP of
    the image head; mIoU of the pixel head is reported under ``"seg"``."""

    name = "seg_multilabel"
    primary_metric = "map_macro"

    def __init__(self, num_classes: int, ignore_index: int | None = None,
                 cls_weight: float = 1.0, label_smoothing: float = 0.0) -> None:
        self.num_classes, self.ignore_index, self.cls_weight = num_classes, ignore_index, \
            cls_weight
        self.seg = SemanticTask(num_classes, ignore_index, label_smoothing)
        self.cls = ClassificationTask(num_classes, multilabel=True)

    def heads(self):
        k = self.num_classes
        return {"seg": {"type": "segmentation", "out_channels": k},
                "logits": {"type": "classification", "out_channels": k, "scope": "image"}}

    def targets(self, y):
        k = self.num_classes
        valid = (y != self.ignore_index if self.ignore_index is not None
                 else torch.ones_like(y, dtype=torch.bool))
        hot = torch.zeros(y.shape[0], k, dtype=torch.long, device=y.device)
        idx = torch.where(valid, y, torch.zeros_like(y)).flatten(1)
        hot.scatter_add_(1, idx, valid.flatten(1).long())  # count valid pixels per class
        return {"mask": y, "labels": hot.clamp(max=1)}

    def loss(self, out, t):
        return self.seg.loss(out["seg"], t["mask"]) + self.cls_weight * self.cls.loss(
            out["logits"], t["labels"])

    def evaluator(self):
        return _SegMultilabelEval(self)


class _SegMultilabelEval:
    def __init__(self, task: SegMultilabelTask) -> None:
        self.t, self.seg, self.cls = task, task.seg.evaluator(), task.cls.evaluator()

    def update(self, out, y) -> None:
        t = self.t.targets(y)
        self.seg.update(out["seg"], t["mask"])
        self.cls.update(out["logits"], t["labels"])

    def compute(self) -> dict:
        seg = self.seg.compute()
        return {**self.cls.compute(), "seg": {k: seg[k] for k in ("miou", "iou", "loss")}}


# --- multi-label segmentation -------------------------------------------------------------------

class MultiLabelSegTask(Task):
    """Overlapping classes per pixel (e.g. cloud and cloud shadow, or several land-use layers).
    Labels ``(B, K, H, W)``: 1 / 0 per class, 255 = ignored for that class. The model's
    segmentation head returns ``K`` logits; the loss is masked BCE (sigmoid per class, not a
    softmax over classes) and predictions are ``sigmoid(logit) > threshold``. Primary: mean
    IoU over classes (F1, precision and recall per class are reported)."""

    name = "multilabel_seg"
    primary_metric = "miou"

    def __init__(self, num_classes: int, ignore_index: int | None = 255,
                 threshold: float = 0.5) -> None:
        self.num_classes, self.ignore = num_classes, 255 if ignore_index is None else ignore_index
        self.threshold = threshold

    def loss(self, out, y):
        out = _image_out(out).float()
        valid = y != self.ignore
        if not bool(valid.any()):
            return out.sum() * 0.0
        bce = F.binary_cross_entropy_with_logits(out, (y == 1).float(), reduction="none")
        return bce[valid].mean()

    def evaluator(self):
        return _MultiLabelSegEval(self)


class _MultiLabelSegEval:
    def __init__(self, task: MultiLabelSegTask) -> None:
        k = task.num_classes
        self.t, self.loss, self.n = task, 0.0, 0  # BCE summed over valid class-pixels
        self.tp, self.fp, self.fn = (torch.zeros(k, dtype=torch.float64) for _ in range(3))

    def update(self, out, y) -> None:
        out = _image_out(out).float()
        valid, pos = y != self.t.ignore, y == 1
        bce = F.binary_cross_entropy_with_logits(out, pos.float(), reduction="none")
        self.loss += float(bce[valid].sum())
        self.n += int(valid.sum())
        pred = torch.sigmoid(out) > self.t.threshold
        tp = (pred & pos & valid).sum((0, 2, 3)).double().cpu()
        fp = (pred & ~pos & valid).sum((0, 2, 3)).double().cpu()
        fn = (~pred & pos & valid).sum((0, 2, 3)).double().cpu()
        self.tp, self.fp, self.fn = self.tp + tp, self.fp + fp, self.fn + fn

    def compute(self) -> dict:
        tp, fp, fn = self.tp, self.fp, self.fn
        iou = (tp / (tp + fp + fn).clamp(min=1)).tolist()
        f1 = (2 * tp / (2 * tp + fp + fn).clamp(min=1)).tolist()
        return {"miou": sum(iou) / len(iou), "iou": iou, "f1": f1,
                "precision": (tp / (tp + fp).clamp(min=1)).tolist(),
                "recall": (tp / (tp + fn).clamp(min=1)).tolist(),
                "loss": self.loss / max(1, self.n), "threshold": self.t.threshold}


# --- change detection --------------------------------------------------------------------------

def _change_scores(mat: torch.Tensor) -> dict:
    """Precision / recall / F1 / IoU of class 1 ("change") from a 2x2 (rows = target)
    confusion matrix."""
    m = mat.double()
    tp, fp, fn = m[1, 1], m[0, 1], m[1, 0]
    return {"precision_change": float(tp / (tp + fp).clamp(min=1)),
            "recall_change": float(tp / (tp + fn).clamp(min=1)),
            "f1_change": float(2 * tp / (2 * tp + fp + fn).clamp(min=1)),
            "iou_change": float(tp / (tp + fp + fn).clamp(min=1))}


class ChangeTask(SemanticTask):
    """Binary change detection: a 2-class semantic task (0 no change, 1 change) whose primary
    metric is the F1 of the change class (precision, recall and IoU of it are reported)."""

    name = "change"
    primary_metric = "f1_change"

    def evaluator(self):
        return _ChangeEval(self)


class _ChangeEval(_SemanticEval):
    def compute(self) -> dict:
        return {**super().compute(), **_change_scores(self.cm.mat)}


class SemanticChangeTask(Task):
    """Semantic change detection (SECOND-style). Labels ``(B, 3, H, W)``: land cover at date 1
    and date 2 (0 = unchanged, 1..K = class, labelled only where something changed) and the
    binary change mask. Heads: ``change`` (2 classes) and ``lc1`` / ``lc2`` (K land-cover
    classes; with ``per_input`` they read each date's own features, ``share`` ties their
    weights). Losses: CE(change) + CE(lc1) + CE(lc2), the land-cover terms on changed pixels.

    Metrics (Yang et al. 2021): predicted maps are ``argmax(lc) + 1`` where change is
    predicted, else 0; ``miou_scd`` = mean of the no-change and change IoUs, ``sek`` = Cohen's
    kappa of the (K+1)-class confusion without its no-change/no-change cell times
    ``exp(IoU_change - 1)``, primary ``score`` = 0.3 ``miou_scd`` + 0.7 ``sek``.
    """

    name = "semantic_change"
    primary_metric = "score"

    def __init__(self, num_classes: int, ignore_index: int | None = None, per_input: bool = True,
                 share: bool = True) -> None:
        self.num_classes, self.per_input, self.share = num_classes, per_input, share

    def heads(self):
        lc = {"type": "segmentation", "out_channels": self.num_classes}
        h = {"change": {"type": "segmentation", "out_channels": 2},
             "lc1": dict(lc), "lc2": dict(lc)}
        if self.per_input:
            h["lc1"]["input"], h["lc2"]["input"] = "t1", "t2"
            if self.share:
                h["lc2"]["share"] = "lc1"
        return h

    def targets(self, y):
        lc = (y[:, :2] - 1).masked_fill(y[:, :2] == 0, -100)  # land cover on changed pixels
        return {"lc1": lc[:, 0], "lc2": lc[:, 1], "change": y[:, 2]}

    def loss(self, out, t):
        loss = F.cross_entropy(out["change"].float(), t["change"])
        for k in ("lc1", "lc2"):
            if bool((t[k] >= 0).any()):  # a batch without change has no land-cover target
                loss = loss + F.cross_entropy(out[k].float(), t[k], ignore_index=-100)
        return loss

    def evaluator(self):
        return _SemanticChangeEval(self)


class _SemanticChangeEval:
    def __init__(self, task: SemanticChangeTask) -> None:
        k = task.num_classes + 1
        self.t, self.k, self.loss, self.n = task, k, 0.0, 0
        self.scd = torch.zeros(k, k, dtype=torch.float64)
        self.bin = torch.zeros(2, 2, dtype=torch.int64)

    def update(self, out, y) -> None:
        self.loss += float(self.t.loss(out, self.t.targets(y))) * y.shape[0]
        self.n += y.shape[0]
        ch = out["change"].argmax(1)
        for i, key in ((0, "lc1"), (1, "lc2")):
            pred = torch.where(ch == 1, out[key].argmax(1) + 1, torch.zeros_like(ch))
            idx = (y[:, i] * self.k + pred).flatten().cpu()
            self.scd += torch.bincount(idx, minlength=self.k * self.k).view(self.k, self.k)
        self.bin += torch.bincount((y[:, 2] * 2 + ch).flatten().cpu(), minlength=4).view(2, 2)

    def compute(self) -> dict:
        cm = self.scd
        total = cm.sum()
        iou_nc = float(cm[0, 0] / (cm[0].sum() + cm[:, 0].sum() - cm[0, 0]).clamp(min=1))
        iou_c = float(cm[1:, 1:].sum() / (total - cm[0, 0]).clamp(min=1))
        q = cm.clone()
        q[0, 0] = 0
        s = q.sum().clamp(min=1)
        po = float(torch.trace(q) / s)
        pe = float((q.sum(0) * q.sum(1)).sum() / s ** 2)
        kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
        sek = kappa * math.exp(iou_c - 1)
        miou = (iou_nc + iou_c) / 2
        return {"score": 0.3 * miou + 0.7 * sek, "miou_scd": miou, "sek": sek,
                "iou_unchanged": iou_nc, "iou_changed": iou_c, **_change_scores(self.bin),
                "loss": self.loss / max(1, self.n)}


# --- vector-output tasks ----------------------------------------------------------------------
#
# Labels are dicts of per-pixel targets encoded from geometry (polygons, road graphs) by the
# dataset: ``manyseg.vectors.corner_targets`` / ``road_targets``. Their line widths and heatmap
# sigmas are in pixels and their directions are angles, so they are augmented only by exact
# dihedral transforms (``direction_targets``), and the heads stay at full resolution.

def _full_resolution(out, y, task: str) -> None:
    if head_factor(out, y) != 1:
        raise ValueError(f"{task}: targets are encoded at full resolution; use head scale 1")


class CornerLineTask(Task):
    """Corner-line instances (B3: footprints, fields, roof faces; E036, E038, E047). Heads:
    :func:`manyseg.vectors.corner_line_heads`. Labels: ``{"instances": ids (B, H, W),
    "corner": (B, H, W), "direction": (B, 2, H, W), "direction_mask": (B, H, W)}`` (the last
    three from :func:`manyseg.vectors.corner_targets`).

    Loss (E030 ``CornerLineTask``): the :class:`InstanceTask` loss (``boundary_width`` 2) +
    ``corner_weight`` * CenterNet focal loss on the corner heatmap (positives: its 3 x 3 local
    maxima above 0.5) + ``direction_weight`` * L1 of the direction on the outline pixels
    (``direction_mask``). Primary metric: PQ of the instance decoding, as for
    :class:`InstanceTask`; ``loss`` is the full loss, ``instance_loss`` its instance part (the
    experiments' validation loss).

    ``distance="absolute"`` (experimental for footprints and roofs; the default for fields,
    E052): unnormalised distance, so an instance cut by the tile edge gets the same values as
    in the whole scene; decode with a ``seed_threshold`` in pixels."""

    name = "corner_lines"
    primary_metric = "pq"
    direction_targets = ("direction",)

    def __init__(self, boundary_width: float = 2.0, corner_weight: float = 1.0,
                 direction_weight: float = 1.0, iou_threshold: float = 0.5,
                 decoders=None, distance: str = "normalized") -> None:
        self.inner = InstanceTask(boundary_width=boundary_width, iou_threshold=iou_threshold,
                                  distance=distance)
        self.corner_weight, self.direction_weight = corner_weight, direction_weight
        self.decoders = decoders

    def heads(self):
        from manyseg.vectors import corner_line_heads

        return corner_line_heads(self.decoders)

    def targets(self, y):
        return {**y, "inner": self.inner.targets(y["instances"])}

    def losses(self, out, t) -> dict:
        """The loss terms (unweighted): ``instances``, ``corner``, ``direction``."""
        _full_resolution(out, t["instances"], self.name)
        inner = t["inner"] if "inner" in t else self.inner.targets(t["instances"])
        tc = t["corner"].float()[:, None]
        corner = losses.heatmap_focal(out["corner"], tc, pos=losses.heatmap_peaks(tc))
        m = t["direction_mask"].bool()[:, None]
        direction = losses.masked_regression(out["direction"], t["direction"], m)
        return {"instances": self.inner.loss(out, inner), "corner": corner,
                "direction": direction}

    def loss(self, out, t):
        parts = self.losses(out, t)
        return (parts["instances"] + self.corner_weight * parts["corner"]
                + self.direction_weight * parts["direction"])

    def evaluator(self):
        return _CornerLineEval(self)


class _CornerLineEval:
    def __init__(self, task: CornerLineTask) -> None:
        self.t, self.inner, self.loss, self.n = task, task.inner.evaluator(), 0.0, 0

    def update(self, out, y) -> None:
        t = self.t.targets(y)
        self.loss += float(self.t.loss(out, t)) * y["instances"].shape[0]
        self.n += y["instances"].shape[0]
        self.inner.update(out, y["instances"])

    def compute(self) -> dict:
        out = self.inner.compute()
        out["instance_loss"] = out.pop("loss")
        return {**out, "loss": self.loss / max(1, self.n)}


class RoadGraphTask(Task):
    """Unified road heads (R4; E039, E044). Heads: :func:`manyseg.vectors.road_heads`.
    Labels: ``{"mask": (B, H, W), "line": (B, H, W), "keypoints": (B, 3, H, W),
    "direction": (B, 4, H, W)}`` from :func:`manyseg.vectors.road_targets`.

    Loss (E044 ``RoadTask``; ``weights`` for mask, line, keypoints, direction): Dice + BCE on
    the mask and on the line; one CenterNet focal loss over the three keypoint heatmaps
    (positives: 3 x 3 local maxima above 0.5), normalised by their total positives with a floor
    of ``min_keypoints`` (with a floor of 1, a batch without crossings let the summed negatives
    swamp the decoder and the mask collapsed, E044); L1 of direction mode 1 on the road surface
    (mask > 0.5) + L1 of mode 2 where its target is set. Primary metric: road-mask IoU (logits
    > 0 vs mask > 0.5), E044's training-time metric; graph metrics (APLS, junction F1) need the
    decoder (``manyseg_bench.evaluation.vectors``)."""

    name = "roads"
    primary_metric = "mask_iou"
    direction_targets = ("direction",)

    def __init__(self, weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
                 min_keypoints: int = 16, decoders=None) -> None:
        self.weights, self.min_keypoints, self.decoders = tuple(weights), min_keypoints, decoders

    def heads(self):
        from manyseg.vectors import road_heads

        return road_heads(self.decoders)

    def losses(self, out, t) -> dict:
        """The loss terms (unweighted): ``mask``, ``line``, ``keypoints``, ``direction``."""
        _full_resolution(out, t["mask"], self.name)
        kp = t["keypoints"].float()
        d, pd = t["direction"].float(), out["direction"]
        on_road = (t["mask"] > 0.5)[:, None]
        crossing = d[:, 2:].abs().sum(1, keepdim=True) > 0
        return {"mask": losses.dice_bce(out["mask"], t["mask"].float()[:, None]),
                "line": losses.dice_bce(out["line"], t["line"].float()[:, None]),
                "keypoints": losses.heatmap_focal(out["keypoints"], kp,
                                                  pos=losses.heatmap_peaks(kp),
                                                  min_pos=self.min_keypoints),
                "direction": (losses.masked_regression(pd[:, :2], d[:, :2], on_road)
                              + losses.masked_regression(pd[:, 2:], d[:, 2:], crossing))}

    def loss(self, out, t):
        parts = self.losses(out, t)
        return sum(w * parts[k] for w, k in zip(self.weights, ("mask", "line", "keypoints",
                                                               "direction")))

    def evaluator(self):
        return _RoadEval(self)


class _RoadEval:
    def __init__(self, task: RoadGraphTask) -> None:
        self.t, self.loss, self.n, self.inter, self.union = task, 0.0, 0, 0, 0

    def update(self, out, y) -> None:
        b = y["mask"].shape[0]
        self.loss += float(self.t.loss(out, y)) * b
        self.n += b
        p, g = out["mask"][:, 0] > 0, y["mask"] > 0.5
        self.inter += int((p & g).sum())
        self.union += int((p | g).sum())

    def compute(self) -> dict:
        return {"mask_iou": self.inter / max(1, self.union), "loss": self.loss / max(1, self.n)}


def build_task(name: str, num_classes: int, ignore_index: int | None = None, **kwargs) -> Task:
    if name == "semantic":
        return SemanticTask(num_classes, ignore_index, **kwargs)
    if name == "instances":
        return InstanceTask(num_classes=num_classes, **kwargs)
    if name == "points":
        return PointTask(**kwargs)
    if name == "classification":
        return ClassificationTask(num_classes, **kwargs)
    if name == "multilabel":
        return ClassificationTask(num_classes, multilabel=True, **kwargs)
    if name == "regression":
        return RegressionTask(num_classes, **kwargs)
    if name == "seg_multilabel":
        return SegMultilabelTask(num_classes, ignore_index, **kwargs)
    if name == "change":
        return ChangeTask(num_classes, ignore_index, **kwargs)
    if name == "multilabel_seg":
        return MultiLabelSegTask(num_classes, ignore_index, **kwargs)
    if name == "semantic_change":
        return SemanticChangeTask(num_classes, ignore_index, **kwargs)
    if name == "corner_lines":
        return CornerLineTask(**kwargs)
    if name == "roads":
        return RoadGraphTask(**kwargs)
    raise ValueError(f"unknown task {name!r}; choose from semantic, instances, points, "
                     "classification, multilabel, regression, seg_multilabel, change, "
                     "semantic_change, multilabel_seg, corner_lines, roads")
