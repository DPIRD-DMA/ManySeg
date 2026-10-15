"""FSDP2 (fully_shard) on CPU (gloo): manyseg.distributed.shard_model trains like DDP."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.slow


def _build(seed=0):
    from manyseg import create_model

    torch.manual_seed(seed)
    return create_model(backbone="resnet18", pretrained=False, decoder="unet",
                        heads={"seg": {"type": "segmentation", "channels": 2},
                               "area": {"type": "regression", "channels": 1, "scope": "image"},
                               "edge": {"type": "segmentation", "channels": 1,
                                        "decoder": "fpn"}})


def _step(model, x):
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    out = model(x)
    sum(v.float().mean() for v in out.values()).backward()
    opt.step()


def _worker(rank: int, world: int, port: int, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from manyseg.distributed import shard_model

    torch.manual_seed(100 + rank)  # different data per rank
    x = torch.randn(2, 3, 64, 96)

    ddp = torch.nn.parallel.DistributedDataParallel(_build())
    _step(ddp, x)
    ref = {k: v.detach().clone() for k, v in ddp.module.state_dict().items()}

    fsdp = shard_model(_build())
    units = [m for m in fsdp.modules() if type(m).__name__.startswith("FSDP")]
    _step(fsdp, x)
    full = {k: (v.full_tensor() if hasattr(v, "full_tensor") else v).detach().clone()
            for k, v in fsdp.state_dict().items()}
    out[rank] = {"ref": ref, "fsdp": full, "units": len(units)}
    dist.destroy_process_group()


def test_shard_model_matches_ddp():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = mp.Manager().dict()
    mp.spawn(_worker, args=(2, port, out), nprocs=2, join=True)
    for rank in (0, 1):
        r = out[rank]
        assert r["units"] >= 6  # backbone, shared + task decoder, 3 heads, root
        for k, v in r["ref"].items():
            if "running_" in k or "num_batches_tracked" in k:
                continue  # BN statistics are per rank under both DDP and FSDP
            torch.testing.assert_close(r["fsdp"][k], v, rtol=1e-4, atol=1e-5, msg=k)
