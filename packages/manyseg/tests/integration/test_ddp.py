"""DistributedDataParallel on CPU (gloo): gradients synchronise across ranks."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.slow


def _worker(rank: int, world: int, port: int, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(0)
    from manyseg import create_model

    model = create_model(backbone="resnet18", pretrained=False, num_classes=2)
    ddp = torch.nn.parallel.DistributedDataParallel(model)
    opt = torch.optim.SGD(ddp.parameters(), lr=0.1)
    torch.manual_seed(rank)  # different data per rank
    ddp(torch.randn(2, 3, 64, 64)).mean().backward()
    opt.step()
    out[rank] = torch.cat([p.detach().flatten() for p in model.parameters()])[:1000].clone()
    dist.destroy_process_group()


def test_ddp_keeps_replicas_in_sync():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = mp.Manager().dict()
    mp.spawn(_worker, args=(2, port, out), nprocs=2, join=True)
    torch.testing.assert_close(out[0], out[1])
