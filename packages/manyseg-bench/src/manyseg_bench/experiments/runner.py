"""One benchmark run = (provider, model spec, dataset, recipe, seed) -> JSON record."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import torch

from ..datasets import GPULoader, data_root, get_dataset, load_split
from ..datasets.base import load_meta
from ..env import environment, gpu_contention
from ..providers import ModelSpec, get_provider
from ..tasks import build_task
from ..training import evaluate, get_recipe, seed_everything, train


def _select(split, info, subset: int | None = None):
    """Keep only the inputs this dataset variant uses (cache may hold more); optionally the
    first ``subset`` samples (debug/smoke runs only; recorded in the result)."""
    if isinstance(split.images, dict) and info.inputs:
        split.images = {k: split.images[k] for k in info.inputs}
    if subset:
        sl = slice(0, subset)
        split.images = ({k: v[sl] for k, v in split.images.items()}
                        if isinstance(split.images, dict) else split.images[sl])
        split.masks, split.names = split.masks[sl], split.names[:subset]
    return split


def info_loader_kwargs(ds) -> dict:
    """Dataset-specific augmentation settings (e.g. vertical flips for overhead imagery)."""
    return dict(getattr(ds, "LOADER_KWARGS", {}))


def run_training(
    provider: str,
    spec: ModelSpec,
    dataset: str,
    recipe: str = "standard",
    seed: int = 0,
    device: str = "cuda",
    out_dir: str | Path = "runs",
    data: str | Path | None = None,
    evaluate_test: bool = False,
    recipe_overrides: dict | None = None,
    tag: str = "",
    save_model: bool = False,
    task_overrides: dict | None = None,
    subset: int | None = None,
) -> dict:
    dev = torch.device(device)
    ds = get_dataset(dataset)
    info = ds.INFO
    cache = data_root(data) / "cache" / getattr(ds, "CACHE_NAME", info.name)
    rec = get_recipe(recipe, **(recipe_overrides or {}))
    meta = load_meta(cache)
    mean, std = meta.get("mean", info.mean), meta.get("std", info.std)
    if info.inputs:
        mean, std = {k: mean[k] for k in info.inputs}, {k: std[k] for k in info.inputs}
    spec = ModelSpec(**{**spec.to_dict(), "num_classes": info.num_classes,
                        "in_channels": info.in_channels, "inputs": info.inputs})

    seed_everything(seed)
    torch.backends.cudnn.benchmark = True
    task_name = getattr(ds, "TASK", "semantic")
    task_kwargs = {**getattr(ds, "TASK_KWARGS", {}), **(task_overrides or {})}
    if task_name == "semantic":
        task_kwargs.setdefault("label_smoothing", rec.label_smoothing)
    task = build_task(task_name, info.num_classes, info.ignore_index, **task_kwargs)
    if task.heads() is not None:
        if provider != "manyseg":
            raise ValueError(f"task {task.name!r} needs named heads; provider {provider!r} "
                             "only produces a single logits tensor")
        spec = ModelSpec(**{**spec.to_dict(), "options": {**spec.options, "heads": task.heads()}})
    prov = get_provider(provider)
    if spec.input_mode == "dict" and not prov.supports_dict_inputs:
        raise ValueError(f"provider {provider!r} does not support dict inputs; "
                         "use input_mode='stack'")
    # Datasets too large for the (shared) GPU stay memory-mapped on the host.
    mmap = not info_loader_kwargs(ds).get("device_resident", True)
    train_split = _select(load_split(cache, "train", mmap=mmap), info, subset)
    img_size = tuple(train_split.masks.shape[1:3])
    built = prov.build(spec, img_size=img_size)

    stack = spec.input_mode != "dict"
    common = dict(mean=mean, std=std, device=dev, stack=stack, **info_loader_kwargs(ds))
    train_loader = GPULoader(train_split.images, train_split.masks, rec.batch_size, train=True,
                             seed=seed, **common)
    val = _select(load_split(cache, "val", mmap=mmap), info, subset)
    eval_bs = getattr(ds, "EVAL_BATCH_SIZE", 32)
    val_loader = GPULoader(val.images, val.masks, eval_bs, train=False, **common)

    env = environment(dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    result = train(built.model, built.encoder_params, built.decoder_params, train_loader,
                   val_loader, num_classes=info.num_classes, recipe=rec, device=dev,
                   ignore_index=info.ignore_index, task=task)
    record = {
        "run_id": f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
        "kind": "train",
        "tag": tag,
        "provider": provider,
        "provider_version": prov.version(),
        "spec": spec.to_dict(),
        "weights_fingerprint": built.weights_fingerprint,
        "params": sum(p.numel() for p in built.model.parameters()),
        "dataset": info.to_dict(),
        "img_size": list(img_size),
        "recipe": rec.to_dict(),
        "task_kwargs": task_kwargs,
        "subset": subset,
        "seed": seed,
        "env": env,
        "gpu_contention_at_end": gpu_contention() if dev.type == "cuda" else None,
        **result,
    }
    if evaluate_test:
        for split in info.test_splits:
            test = _select(load_split(cache, split, mmap=mmap), info, subset)
            test_loader = GPULoader(test.images, test.masks, eval_bs, train=False, **common)
            record[split] = evaluate(built.model, test_loader, info.num_classes, dev,
                                     rec.precision, info.ignore_index, task=task)
            del test, test_loader
    if save_model:
        import hashlib

        ckpt_dir = data_root(data) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt = ckpt_dir / f"{record['run_id']}.pt"
        torch.save(built.model.state_dict(), ckpt)
        record["checkpoint"] = {"path": str(ckpt),
                                "sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest()}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{record['run_id']}_{provider}_{spec.arch}_{spec.encoder}_s{seed}.json"
    path.write_text(json.dumps(record, indent=1))
    record["path"] = str(path)
    return record
