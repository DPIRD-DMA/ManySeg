"""Re-evaluate a saved training run under precision and int8 variants (E008).

The run's JSON record supplies everything: the model spec, dataset, split scheme, the exact
normalisation statistics used in training and the checkpoint (verified by sha256). Every
variant scores the *same* weights on the same images, and each batch is also run through the
fp32 reference on the variant's device, so ``agreement`` is the paired fraction of valid pixels
whose argmax matches fp32.

The record (``"kind": "evaluate"``) holds the dataset's ``DatasetInfo`` as ``dataset``, as a
training record does, and names the task's ``primary_metric`` and the evaluated ``splits``; a
variant that failed holds ``{"device", "error"}`` instead of scores. ``size_bytes`` is the
serialized ``state_dict`` for the floating-point variants and the quantised program's size for
int8 (only the int8 variants need ``manyseg[quant]``).
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path

import torch

from ..datasets import GPULoader, data_root, get_dataset
from ..env import environment
from ..profiling import serialized_size_bytes
from ..providers import ModelSpec, get_provider
from ..tasks import build_task
from ..training.trainer import autocast
from .runner import info_loader_kwargs, split_getter

# name -> (device kind, autocast precision); int8 variants run on CPU in fp32 graphs.
VARIANTS = {
    "fp32": ("gpu", "fp32"),
    "bf16": ("gpu", "bf16"),
    "fp16": ("gpu", "fp16"),
    "int8_pt2e": ("cpu", "fp32"),
    "int8_pt2e_compiled": ("cpu", "fp32"),
    "int8_weight_only": ("cpu", "fp32"),
}


def _quantised(name: str, model, calib, example):
    """(callable model, serialized size in bytes) for an int8 variant."""
    from manyseg.quantize import model_size, quantize_pt2e, quantize_weights

    if name == "int8_weight_only":
        q = quantize_weights(model, inplace=False)
        return q, model_size(q)
    if name == "int8_pt2e":
        ep = quantize_pt2e(model, example, calib, dynamic_batch=True)  # last batch is smaller
        return ep.module(), model_size(ep)
    # Deployed form. Inductor freezing on a program with symbolic sizes crashes in dynamo
    # (torch 2.14: IndexError in _maybe_preserve_original_meta), so export at the fixed eval
    # shape and pad a smaller final batch up to it (extra outputs are dropped).
    bs = _batch_of(example)
    ep = quantize_pt2e(model, example, [c for c in calib if _batch_of(c) == bs], dynamic=False)
    torch._inductor.config.freezing = True  # constant-folds the int8 weights (oneDNN kernels)
    compiled = torch.compile(ep.module())

    def run(x):
        n = _batch_of(x)
        out = compiled(_pad_batch(x, bs))
        return out[:n] if torch.is_tensor(out) else {k: v[:n] for k, v in out.items()}

    return run, model_size(ep)


def _batch_of(x) -> int:
    return int((x if torch.is_tensor(x) else next(iter(x.values()))).shape[0])


def _pad_batch(x, bs: int):
    """Repeat the last sample up to ``bs`` (tensor or dict of tensors, e.g. multi-input)."""
    if not torch.is_tensor(x):
        return {k: _pad_batch(v, bs) for k, v in x.items()}
    n = x.shape[0]
    return x if n >= bs else torch.cat([x, x[-1:].expand(bs - n, *x.shape[1:])])


def _agreement(ref, out, y, ignore_index):
    """(agreeing, total) valid pixels between reference and candidate class predictions, or
    None when the outputs are not single full-resolution logits (multi-head tasks, reduced
    head scale): there is no meaningful pixel argmax to compare then."""
    if not (torch.is_tensor(ref) and torch.is_tensor(out)) or ref.ndim != 4 or \
            ref.shape[-2:] != y.shape[-2:] or ref.shape != out.shape:
        return None
    valid = y != ignore_index if ignore_index is not None else torch.ones_like(y, dtype=torch.bool)
    return int(((ref.argmax(1) == out.argmax(1)) & valid).sum()), int(valid.sum())


def evaluate_run(run: str | Path, variants=("fp32", "bf16", "fp16"), device: str = "cuda",
                 data: str | Path | None = None, out_dir: str | Path = "runs",
                 calib_batches: int = 32, threads: int = 4) -> dict:
    record = json.loads(Path(run).read_text())
    bad = [v for v in variants if v not in VARIANTS]
    if bad:
        raise ValueError(f"unknown variant(s) {bad}; choose from {sorted(VARIANTS)}")
    ck = Path(record["checkpoint"]["path"])
    if hashlib.sha256(ck.read_bytes()).hexdigest() != record["checkpoint"]["sha256"]:
        raise ValueError(f"checkpoint {ck} does not match the sha256 in its run record")
    torch.set_num_threads(threads)

    ds = get_dataset(record["dataset"]["name"])
    info = ds.INFO
    name = getattr(ds, "CACHE_NAME", info.name)
    cache, raw = data_root(data) / "cache" / name, data_root(data) / "raw" / name
    spec = ModelSpec(**record["spec"])
    mmap = not info_loader_kwargs(ds).get("device_resident", True)
    get_split = split_getter(ds, cache, raw, record.get("split_scheme", "official"), mmap=mmap,
                             subset=record.get("subset"))
    task = build_task(getattr(ds, "TASK", "semantic"), info.num_classes, info.ignore_index,
                      **record.get("task_kwargs", {}))
    if record.get("calibrated_threshold") is not None:
        task.peak_threshold = record["calibrated_threshold"]
    norm = record["norm_stats"]
    built = get_provider(record["provider"]).build(spec, img_size=tuple(record["img_size"]))
    state = torch.load(ck, map_location="cpu", weights_only=True)
    eval_bs = getattr(ds, "EVAL_BATCH_SIZE", 32)
    splits = ("val", *info.test_splits)

    out = {"run_id": f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
           "kind": "evaluate", "source_run": record["run_id"], "tag": record.get("tag", ""),
           "spec": record["spec"], "dataset": info.to_dict(), "seed": record.get("seed"),
           "checkpoint": record["checkpoint"], "calib_batches": calib_batches,
           "threads": threads, "primary_metric": task.primary_metric,
           "splits": list(splits), "variants": {}}
    envs = {}
    for v in variants:
        kind, precision = VARIANTS[v]
        dev = torch.device("cpu" if kind == "cpu" else device)
        envs.setdefault(dev.type, environment(dev))
        ref = built.model.to(dev).eval()
        ref.load_state_dict(state)

        def loader(split, dev=dev):
            s = get_split(split)
            return GPULoader(s.images, s.masks, eval_bs, train=False, vectors=s.vectors,
                             mean=norm["mean"], std=norm["std"], device=dev,
                             stack=spec.input_mode != "dict", **info_loader_kwargs(ds))

        try:
            out["variants"][v] = _score(v, ref, loader, splits, task, info, dev, precision,
                                        calib_batches)
        except Exception as e:  # one failing variant must not discard the others
            out["variants"][v] = {"device": dev.type, "error": f"{type(e).__name__}: {e}"}
    out["env"] = envs
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{out['run_id']}_evaluate_{record['run_id']}.json"
    path.write_text(json.dumps(out, indent=1))
    out["path"] = str(path)
    return out


def _score(v, ref, loader, splits, task, info, dev, precision, calib_batches) -> dict:
    model, size = ref, None
    if v.startswith("int8"):
        calib = [x for _, (x, _) in zip(range(calib_batches), loader("train"))]
        model, size = _quantised(v, ref, calib, calib[0])
    # floating-point variants: the state_dict as saved (no ManySeg / quantisation dependency)
    res = {"device": dev.type, "precision": precision,
           "size_bytes": size if size is not None else serialized_size_bytes(ref)}
    for split in splits:
        ev, agree, total, t0 = task.evaluator(), 0, 0, time.perf_counter()
        for x, y in loader(split):
            with torch.no_grad():
                ref_out = ref(x)
                with autocast(dev, precision):
                    o = model(x)
            ev.update(o, y)
            a = _agreement(ref_out, o, y, info.ignore_index)
            if a is not None:
                agree, total = agree + a[0], total + a[1]
        res[split] = {**ev.compute(), "agreement": agree / total if total else None,
                      "seconds": time.perf_counter() - t0}
    return res
