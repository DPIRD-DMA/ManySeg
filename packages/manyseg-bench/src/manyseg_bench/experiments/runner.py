"""One benchmark run = (provider, model spec, dataset, recipe, seed) -> JSON record."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import torch

from ..datasets import GPULoader, data_root, get_dataset, load_split
from ..datasets.base import channel_stats, load_meta
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


def split_getter(ds, cache: Path, raw: Path, scheme: str = "official", mmap: bool = False,
                 subset: int | None = None):
    """``get(split_name) -> Split`` for a split scheme, with vector (metadata) inputs attached.

    ``"official"``: the cached splits. ``"events:K:F"`` (datasets with ``event_folds``): pool
    the official train/val/test chips and re-split them leave-events-out (fold F of K); other
    splits (e.g. an event holdout such as Sen1Floods11's Bolivia) are used as cached.
    ``run_training`` normalises event splits with statistics of the fold's training chips
    (``meta.json`` holds the official train split's, which would include held-out events).
    """
    import numpy as np

    from ..datasets import Split

    info = ds.INFO
    pooled, part = None, None
    if scheme.startswith("events:"):
        if not hasattr(ds, "event_folds"):
            raise ValueError(f"dataset {info.name!r} has no event-level split")
        _, k, fold = scheme.split(":")
        parts = [_select(load_split(cache, n, mmap=False), info) for n in ("train", "val", "test")]
        cat = np.concatenate
        images = ({k_: cat([p.images[k_] for p in parts]) for k_ in parts[0].images}
                  if isinstance(parts[0].images, dict) else cat([p.images for p in parts]))
        pooled = Split(images, cat([p.masks for p in parts]), sum((p.names for p in parts), []))
        del parts
        part = ds.event_folds(pooled.names, int(k), int(fold))
    elif scheme != "official":
        raise ValueError(f"unknown split scheme {scheme!r}; use 'official' or 'events:K:F'")

    def get(name: str):
        if part is not None and name in part:
            idx = part[name]
            images = ({k_: v[idx] for k_, v in pooled.images.items()}
                      if isinstance(pooled.images, dict) else pooled.images[idx])
            split = _select(Split(images, pooled.masks[idx], [pooled.names[i] for i in idx]),
                            info, subset)
        else:
            split = _select(load_split(cache, name, mmap=mmap), info, subset)
        if info.vectors:
            split.vectors = ds.load_vectors(raw, split.names, info.vectors)
        return split

    return get


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
    split: str = "official",
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
                        "in_channels": info.in_channels, "inputs": info.inputs,
                        "vectors": info.vectors})
    raw = data_root(data) / "raw" / getattr(ds, "CACHE_NAME", info.name)
    # Datasets too large for the (shared) GPU stay memory-mapped on the host.
    mmap = not info_loader_kwargs(ds).get("device_resident", True)
    get_split = split_getter(ds, cache, raw, split, mmap=mmap, subset=subset)

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
    train_split = get_split("train")
    norm_source = "meta.json (official train split)"
    if split != "official":  # pooled splits: never let held-out chips into the statistics
        imgs = train_split.images
        stats = ({k: channel_stats(v) for k, v in imgs.items()} if isinstance(imgs, dict)
                 else {None: channel_stats(imgs)})
        mean = {k: m for k, (m, _) in stats.items()}
        std = {k: sd for k, (_, sd) in stats.items()}
        if None in mean:
            mean, std = mean[None], std[None]
        norm_source = f"train split of {split}"
    img_size = tuple(train_split.masks.shape[1:3])
    built = prov.build(spec, img_size=img_size)

    stack = spec.input_mode != "dict"
    common = dict(mean=mean, std=std, device=dev, stack=stack, **info_loader_kwargs(ds))
    train_loader = GPULoader(train_split.images, train_split.masks, rec.batch_size, train=True,
                             seed=seed, vectors=train_split.vectors, **common)
    val = get_split("val")
    eval_bs = getattr(ds, "EVAL_BATCH_SIZE", 32)
    val_loader = GPULoader(val.images, val.masks, eval_bs, train=False, vectors=val.vectors,
                           **common)

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
        "split_scheme": split,
        "split_sizes": {"train": len(train_split.names), "val": len(val.names)},
        "vectors_dropped": bool(info.vectors) and stack,
        "norm_stats": {"source": norm_source, "mean": mean, "std": std},
        "env": env,
        "gpu_contention_at_end": gpu_contention() if dev.type == "cuda" else None,
        **result,
    }
    if evaluate_test:
        for test_split in info.test_splits:
            test = get_split(test_split)
            test_loader = GPULoader(test.images, test.masks, eval_bs, train=False,
                                    vectors=test.vectors, **common)
            record[test_split] = evaluate(built.model, test_loader, info.num_classes, dev,
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
