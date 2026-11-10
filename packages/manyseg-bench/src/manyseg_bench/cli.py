"""Command-line entry point: ``manyseg-bench {prepare,train,profile,evaluate,report}``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _spec(a):
    from .providers import ModelSpec

    return ModelSpec(arch=a.arch, encoder=a.encoder,
                     weights=None if a.weights == "none" else a.weights,
                     input_mode=getattr(a, "input_mode", "stack"),
                     options=json.loads(a.options))


def _r4(v):
    """A metric rounded for printing; "n/a" when it was not computed (e.g. ``agreement`` with
    no comparable pixels is ``None``)."""
    return "n/a" if v is None else round(v, 4)


def main(argv=None) -> int:
    p = argparse.ArgumentParser("manyseg-bench")
    p.add_argument("--data", default=None, help="data root (default $MANYSEG_BENCH_DATA)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare")
    pp.add_argument("dataset")
    pp.add_argument("--size", type=int, default=None,
                    help="tile / resize size in px for datasets that tile or resize (dataset "
                    "default if omitted); fixed-size datasets (bigearthnet, ai4boundaries_s2, "
                    "pastis_ts, second) reject a different size")

    for name in ("train", "profile"):
        sp = sub.add_parser(name)
        sp.add_argument("--provider", required=True)
        sp.add_argument("--arch", default="unet")
        sp.add_argument("--encoder", default="resnet34")
        sp.add_argument("--weights", default="tv_in1k")
        sp.add_argument("--options", default="{}", help="JSON provider options")
        sp.add_argument("--device", default="cuda")
        sp.add_argument("--out", default="runs")
        sp.add_argument("--tag", default="")
    tp = sub.choices["train"]
    tp.add_argument("--dataset", default="oxford_pet")
    tp.add_argument("--recipe", default="standard")
    tp.add_argument("--seeds", default="0")
    tp.add_argument("--set", default="{}", help="JSON recipe overrides")
    tp.add_argument("--input-mode", default="stack", choices=["stack", "dict"],
                    help="multi-input datasets: concatenate inputs or pass a dict")
    tp.add_argument("--task-kwargs", default="{}",
                    help="JSON overrides for the dataset's task (e.g. instance decoders)")
    tp.add_argument("--subset", type=int, default=None,
                    help="debug: use only the first N samples of each split (recorded)")
    tp.add_argument("--split", default="official",
                    help="'official' or 'events:K:F' (leave-events-out fold F of K)")
    tp.add_argument("--save-model", action="store_true",
                    help="save the trained state_dict under <data root>/checkpoints")
    tp.add_argument("--evaluate-test", action="store_true",
                    help="also evaluate the test split (final reporting only)")
    pr = sub.choices["profile"]
    pr.add_argument("--size", default="512,512")
    pr.add_argument("--in-channels", type=int, default=3)
    pr.add_argument("--num-classes", type=int, default=3)
    pr.add_argument("--batch-sizes", default="1,8,32")
    pr.add_argument("--precision", default="bf16")
    pr.add_argument("--compile", action="store_true")
    pr.add_argument("--channels-last", action="store_true")
    pr.add_argument("--train-batch-size", type=int, default=16,
                    help="training-step batch size (peak training memory); 0 skips it")

    ep = sub.add_parser("evaluate", help="re-evaluate a saved run (--save-model) under "
                        "precision / int8 variants")
    ep.add_argument("run", help="the training run's JSON record")
    ep.add_argument("--variants", default="fp32,bf16,fp16",
                    help="comma-separated: fp32, bf16, fp16 (on --device), int8_pt2e, "
                    "int8_pt2e_compiled, int8_weight_only (CPU; need manyseg[quant])")
    ep.add_argument("--device", default="cuda")
    ep.add_argument("--out", default="runs")
    ep.add_argument("--calib-batches", type=int, default=32)
    ep.add_argument("--threads", type=int, default=4, help="CPU threads for int8 variants")

    rp = sub.add_parser("report")
    rp.add_argument("paths", nargs="+")

    a = p.parse_args(argv)
    if a.cmd == "prepare":
        from .datasets import data_root, get_dataset

        ds = get_dataset(a.dataset)
        root = data_root(a.data)
        name = getattr(ds, "CACHE_NAME", ds.INFO.name)
        kw = {} if a.size is None else {"size": a.size}
        print(json.dumps(ds.prepare(root / "raw" / name, root / "cache" / name, **kw), indent=1))
    elif a.cmd == "train":
        from .experiments import run_training

        for seed in [int(s) for s in a.seeds.split(",")]:
            r = run_training(a.provider, _spec(a), a.dataset, a.recipe, seed, a.device, a.out,
                             a.data, a.evaluate_test, json.loads(a.set), a.tag,
                             save_model=a.save_model,
                             task_overrides=json.loads(a.task_kwargs), subset=a.subset,
                             split=a.split)
            pm = r.get("primary_metric", "miou")
            name = "mIoU" if pm == "miou" else pm
            extra = "".join(f" {sp} {name} {r[sp][pm]:.4f}"
                            for sp in r["dataset"]["test_splits"] if sp in r)
            print(f"{r['path']}: final val {name} {r['final_val'][pm]:.4f}{extra}", flush=True)
    elif a.cmd == "profile":
        import time
        import uuid

        from .env import environment, gpu_contention
        from .experiments.runner import resolution_info
        from .profiling import profile_model
        from .providers import ModelSpec, get_provider

        h, w = (int(v) for v in a.size.split(","))
        spec = _spec(a)
        spec = ModelSpec(**{**spec.to_dict(), "in_channels": a.in_channels,
                            "num_classes": a.num_classes})
        prov = get_provider(a.provider)
        built = prov.build(spec, img_size=(h, w))
        env = environment(a.device)
        res = profile_model(built.model, in_channels=a.in_channels, img_size=(h, w),
                            device=a.device, precision=a.precision, compile=a.compile,
                            channels_last=a.channels_last,
                            batch_sizes=tuple(int(b) for b in a.batch_sizes.split(",")),
                            train_batch_size=a.train_batch_size)
        rec = {"run_id": f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
               "kind": "profile", "tag": a.tag, "provider": a.provider,
               "provider_version": prov.version(), "spec": spec.to_dict(),
               "weights_fingerprint": built.weights_fingerprint, "env": env,
               "resolution": resolution_info(built.model),
               "gpu_contention_at_end": gpu_contention(), **res}
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{rec['run_id']}_profile_{a.provider}_{spec.arch}_{spec.encoder}.json"
        path.write_text(json.dumps(rec, indent=1))
        print(json.dumps({k: rec[k] for k in ("params", "flops_per_image", "inference", "train")},
                         indent=1))
    elif a.cmd == "evaluate":
        from .experiments import evaluate_run

        rec = evaluate_run(a.run, variants=tuple(a.variants.split(",")), device=a.device,
                           data=a.data, out_dir=a.out, calib_batches=a.calib_batches,
                           threads=a.threads)
        for v, r in rec["variants"].items():
            print(v, {s: {k: _r4(r[s].get(k)) for k in ("miou", "agreement")}
                      for s in r if isinstance(r[s], dict) and "miou" in r[s]})
        print(rec["path"])
    elif a.cmd == "report":
        from .reporting import load_runs, training_table

        print(training_table(load_runs(a.paths)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
