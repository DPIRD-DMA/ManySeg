"""One benchmark run = (provider, model spec, dataset, recipe, seed) -> JSON record."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import torch

from ..datasets import GPULoader, data_root, get_dataset, load_split
from ..env import environment, gpu_contention
from ..providers import ModelSpec, get_provider
from ..training import evaluate, get_recipe, seed_everything, train


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
) -> dict:
    dev = torch.device(device)
    ds = get_dataset(dataset)
    info = ds.INFO
    cache = data_root(data) / "cache" / info.name
    rec = get_recipe(recipe, **(recipe_overrides or {}))
    spec = ModelSpec(**{**spec.to_dict(), "num_classes": info.num_classes,
                        "in_channels": info.in_channels})

    seed_everything(seed)
    torch.backends.cudnn.benchmark = True
    prov = get_provider(provider)
    train_split = load_split(cache, "train", mmap=False)
    img_size = tuple(train_split.images.shape[1:3])
    built = prov.build(spec, img_size=img_size)

    common = dict(mean=info.mean, std=info.std, device=dev)
    train_loader = GPULoader(train_split.images, train_split.masks, rec.batch_size, train=True,
                             seed=seed, **common)
    val = load_split(cache, "val", mmap=False)
    val_loader = GPULoader(val.images, val.masks, 32, train=False, **common)

    env = environment(dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    result = train(built.model, built.encoder_params, built.decoder_params, train_loader,
                   val_loader, num_classes=info.num_classes, recipe=rec, device=dev,
                   ignore_index=info.ignore_index)
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
        "seed": seed,
        "env": env,
        "gpu_contention_at_end": gpu_contention() if dev.type == "cuda" else None,
        **result,
    }
    if evaluate_test:
        test = load_split(cache, "test", mmap=False)
        test_loader = GPULoader(test.images, test.masks, 32, train=False, **common)
        record["test"] = evaluate(built.model, test_loader, info.num_classes, dev,
                                  rec.precision, info.ignore_index)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{record['run_id']}_{provider}_{spec.arch}_{spec.encoder}_s{seed}.json"
    path.write_text(json.dumps(record, indent=1))
    record["path"] = str(path)
    return record
