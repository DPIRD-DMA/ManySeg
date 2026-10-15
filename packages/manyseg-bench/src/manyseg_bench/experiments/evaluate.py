"""Re-evaluate a saved training run under precision and int8 variants (E008).

The run's JSON record supplies everything: the model spec, dataset, split scheme, the exact
normalisation statistics used in training and the checkpoint (verified by sha256). Every
variant scores the *same* weights on the same images, and each batch is also run through the
fp32 reference on the variant's device, so ``agreement`` is the paired fraction of valid pixels
whose argmax matches fp32.
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
    ep = quantize_pt2e(model, example, calib, dynamic_batch=True)  # last eval batch is smaller
    if name == "int8_pt2e":
        return ep.module(), model_size(ep)
    torch._inductor.config.freezing = True  # constant-folds the int8 weights (oneDNN kernels)
    return torch.compile(ep.module()), model_size(ep)


def _size_bytes(model) -> int:
    from manyseg.quantize import model_size

    return model_size(model)


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
           "spec": record["spec"], "dataset": info.name, "seed": record.get("seed"),
           "checkpoint": record["checkpoint"], "calib_batches": calib_batches,
           "threads": threads, "variants": {}}
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

        model, size = ref, None
        if v.startswith("int8"):
            calib = [x for _, (x, _) in zip(range(calib_batches), loader("train"))]
            model, size = _quantised(v, ref, calib, calib[0])
        res = {"device": dev.type, "precision": precision, "size_bytes": size or _size_bytes(ref)}
        for split in splits:
            ev, agree, total, t0 = task.evaluator(), 0, 0, time.perf_counter()
            for x, y in loader(split):
                with torch.no_grad():
                    ref_out = ref(x)
                    with autocast(dev, precision):
                        o = model(x)
                ev.update(o, y)
                r = ref_out if torch.is_tensor(ref_out) else next(iter(ref_out.values()))
                p = o if torch.is_tensor(o) else next(iter(o.values()))
                valid = y != info.ignore_index if info.ignore_index is not None else \
                    torch.ones_like(y, dtype=torch.bool)
                agree += int(((r.argmax(1) == p.argmax(1)) & valid).sum())
                total += int(valid.sum())
            res[split] = {**ev.compute(), "agreement": agree / max(1, total),
                          "seconds": time.perf_counter() - t0}
        out["variants"][v] = res
    out["env"] = envs
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{out['run_id']}_evaluate_{record['run_id']}.json"
    path.write_text(json.dumps(out, indent=1))
    out["path"] = str(path)
    return out
