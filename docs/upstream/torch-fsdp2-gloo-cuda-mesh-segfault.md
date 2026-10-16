# Draft upstream issue (PyTorch) — not posted

**Title:** FSDP2 `fully_shard` on a gloo process group moves CPU parameters to CUDA when a GPU is
visible; `DTensor.full_tensor()` then segfaults instead of raising

**Environment:** torch 2.14.1+cu130, Linux, one RTX 4090 visible, Python 3.12, CPU training with
`init_process_group("gloo")` (2 processes via `torch.multiprocessing.spawn`, no `LOCAL_RANK`,
no `torch.cuda.set_device`).

**Reproduction:** [`repro_fsdp_cpu_model_cuda_mesh.py`](repro_fsdp_cpu_model_cuda_mesh.py)

```
1 cuda cuda:0
0 cuda cuda:0
torch.multiprocessing.spawn.ProcessExitedException: process 0 terminated with signal SIGSEGV
```

**What happens**

1. `fully_shard(model)` without `mesh` builds the default mesh with the default accelerator
   (`_init_default_fully_shard_mesh`: "global CUDA mesh if possible"), although the process
   group is gloo and the model is on CPU.
2. The parameters are moved to the mesh device: both ranks put their shards on `cuda:0`
   (device heuristic `global_rank % num_devices_per_host` with one GPU), with a warning about
   no default device being set.
3. `model.weight.full_tensor()` (and `redistribute(Replicate())`, and
   `get_model_state_dict(..., full_state_dict=True)`) segfault in
   `torch/distributed/_functional_collectives.py` (`AsyncCollectiveTensor.__new__`, reached
   from a custom op's `_c_check_aliasing_constraint`).

**Expected:** either a CPU mesh when the default process group is gloo and the module is on CPU,
or a Python error explaining the backend/device mismatch. A segfault kills the worker with no
indication of the cause.

**Works:** `fully_shard(model, mesh=init_device_mesh("cpu", (world,)))`, or
`CUDA_VISIBLE_DEVICES=` (the default mesh is then CPU). Plain DTensors from
`distribute_tensor(..., init_device_mesh("cpu", ...), [Shard(0)])` gather fine, as do c10d and
functional all-gathers of CPU tensors.

**ManySeg workaround:** `manyseg.distributed.shard_model` passes a CPU mesh built from the
default group when every parameter is on CPU, the group is gloo and CUDA is available
(commit 2d7b3c6).
