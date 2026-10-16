"""FSDP2 (fully_shard) on CPU (gloo): manyseg.distributed.shard_model trains like DDP."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.slow


HEADS = {"seg": {"type": "segmentation", "channels": 2},
         "area": {"type": "regression", "channels": 1, "scope": "image"},
         "edge": {"type": "segmentation", "channels": 1, "decoder": "fpn"}}


def _build(kind):
    from manyseg import create_model

    torch.manual_seed(0)
    if kind == "single":
        return create_model(backbone="resnet18", pretrained=False, decoder="unet", heads=HEADS)
    # Multi-input + shared encoder + metadata (FiLM on decoders, bias on a head): the
    # conditioner's parameters are used outside its own forward (Codex review).
    return create_model(
        backbone="resnet18", pretrained=False, decoder="unet", heads=HEADS,
        fusion={"type": "intermediate", "op": "gated"},
        inputs={"a": {"channels": 3}, "b": {"channels": 2, "backbone": "shared:a"},
                "meta": {"kind": "vector", "conditioning": "film",
                         "fields": {"doy": {"type": "cyclic", "period": 365.25}}},
                "meta2": {"kind": "vector", "conditioning": {"mode": "head", "pixel": True},
                          "fields": {"elev": {"type": "continuous", "mean": 0.0, "std": 1.0}}}})


def _inputs(kind, rank):
    torch.manual_seed(100 + rank)  # different data per rank
    if kind == "single":
        return torch.randn(2, 3, 64, 96)
    return {"a": torch.randn(2, 3, 64, 96), "b": torch.randn(2, 2, 64, 96),
            "meta": torch.rand(2, 1) * 365, "meta2": torch.randn(2, 1)}


def _step(model, x):
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    out = model(x)
    sum(v.float().mean() for v in out.values()).backward()
    opt.step()


def _worker(rank: int, world: int, port: int, kind: str, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from manyseg.distributed import shard_model

    x = _inputs(kind, rank)
    ddp = torch.nn.parallel.DistributedDataParallel(_build(kind))
    _step(ddp, x)
    ref = {k: v.detach().clone() for k, v in ddp.module.state_dict().items()}

    fsdp = shard_model(_build(kind))
    # Units that own a parameter group (a class change alone proves nothing: Codex review).
    from torch.distributed.fsdp import FSDPModule

    units = [m for m in fsdp.modules() if isinstance(m, FSDPModule)
             and m._get_fsdp_state()._fsdp_param_group is not None]
    sharded = all(type(p).__name__ == "DTensor" for p in fsdp.parameters())
    # The default mesh follows the model's device (a CPU model on a machine with a GPU must
    # not get a CUDA mesh: gathering its weights then segfaults inside torch).
    mesh_types = {p.device_mesh.device_type for p in fsdp.parameters()}
    _step(fsdp, x)
    full = {k: (v.full_tensor() if hasattr(v, "full_tensor") else v).detach().clone()
            for k, v in fsdp.state_dict().items()}
    out[rank] = {"ref": ref, "fsdp": full, "units": len(units), "sharded": sharded,
                 "mesh_types": mesh_types}
    dist.destroy_process_group()


@pytest.mark.parametrize("kind", ["single", "multi_conditioned"])
def test_shard_model_matches_ddp(kind):
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = mp.Manager().dict()
    mp.spawn(_worker, args=(2, port, kind, out), nprocs=2, join=True)
    for rank in (0, 1):
        r = out[rank]
        assert r["sharded"]  # every parameter is owned by some unit
        assert r["mesh_types"] == {"cpu"}, r["mesh_types"]
        # backbone (or the weight-sharing multi-encoder), shared + task decoder, 3 heads
        assert r["units"] >= 6, r["units"]
        for k, v in r["ref"].items():
            if "running_" in k or "num_batches_tracked" in k:
                continue  # BN statistics are per rank under both DDP and FSDP
            torch.testing.assert_close(r["fsdp"][k], v, rtol=1e-4, atol=1e-5, msg=k)
