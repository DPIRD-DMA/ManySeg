"""Segfault: FSDP2 default mesh is CUDA for a CPU model on gloo when a GPU is visible.

Run on a machine with a visible GPU:  python repro_fsdp_cpu_model_cuda_mesh.py
Expected: an error (or a CPU mesh). Actual (torch 2.14.1+cu130): SIGSEGV in full_tensor().
Passing mesh=init_device_mesh("cpu", (2,)) or CUDA_VISIBLE_DEVICES= avoids it.
"""
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, world, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from torch.distributed.fsdp import fully_shard

    model = torch.nn.Linear(8, 16)  # on CPU
    fully_shard(model)              # default mesh -> "cuda" because a GPU is visible
    print(rank, model.weight.device_mesh.device_type, model.weight.device, flush=True)
    model.weight.full_tensor()      # SIGSEGV
    print(rank, "ok", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(worker, args=(2, port), nprocs=2)
