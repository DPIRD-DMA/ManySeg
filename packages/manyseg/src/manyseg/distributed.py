"""FSDP2 sharding for ManySeg models.

``shard_model`` applies ``torch.distributed.fsdp.fully_shard`` stage by stage: the backbone
(each input's encoder for multi-input models), the shared decoder, each task decoder and each
head, then the root (which also owns the conditioner, fusion and adapters). One FSDP unit per
stage lets FSDP overlap communication with compute; finer units buy little at
segmentation-model sizes. DDP needs
nothing from ManySeg (wrap the model as usual).
"""

from __future__ import annotations

from torch import nn


def _stages(model: nn.Module) -> list[nn.Module]:
    bb = model.backbone
    encoders = getattr(bb, "encoders", None)
    stages: list[nn.Module] = []
    if encoders is not None:  # multi-input: one unit per encoder...
        encs = list({id(e): e for e in encoders.values()}.values())
        owned = [id(p) for e in encs for p in e.parameters()]
        if len(owned) != len(set(owned)):
            # ...unless encoders share weights across modules (shared:<input> copies with
            # their own BatchNorm): then the whole multi-input encoder is one unit, so shared
            # parameters stay gathered for every input's pass.
            stages.append(bb)
        else:
            stages += encs
    else:
        stages.append(bb)
    if model.decoder is not None:
        stages.append(model.decoder)
    # shared per-input heads reuse one decoder / head module under two names: shard it once
    stages += list({id(d): d for d in model.task_decoders.values()}.values())
    stages += list({id(h): h for h in model.heads.values()}.values())  # each head its own unit
    # The conditioner is *not* its own unit: DenseModel calls its FiLM / head-bias methods
    # outside Conditioner.forward, after a separate unit would have resharded them. Its
    # parameters belong to the root unit, which stays gathered for the whole forward.
    return [s for s in stages if any(p.requires_grad for p in s.parameters())]


def _cpu_mesh_if_needed(model: nn.Module):
    """A CPU mesh for an all-CPU model on a gloo (CPU) process group when a GPU is visible:
    ``fully_shard``'s default mesh would use CUDA, and gathering the weights then segfaults
    inside torch. Otherwise ``None`` (torch's default: e.g. a CPU model with NCCL is moved to
    the GPU, mixed / meta-device models are left to FSDP)."""
    import torch
    import torch.distributed as dist

    devices = {p.device.type for p in model.parameters()}
    if devices != {"cpu"} or not torch.cuda.is_available():
        return None
    if dist.get_backend() != "gloo":
        return None
    from torch.distributed.device_mesh import DeviceMesh

    return DeviceMesh.from_group(dist.group.WORLD, "cpu")


def shard_model(model: nn.Module, *, mesh=None, mp_policy=None, reshard_after_forward=True,
                **kwargs) -> nn.Module:
    """Shard a ``DenseModel`` with FSDP2 (``fully_shard``), stage by stage, then the root.

    ``mesh`` (a ``DeviceMesh``; default: the global process group; an all-CPU model on a gloo
    group gets a CPU mesh even when a GPU is visible), ``mp_policy`` (e.g.
    ``MixedPrecisionPolicy(param_dtype=torch.bfloat16)``) and ``reshard_after_forward`` are
    passed through, as are other ``fully_shard`` keyword arguments. Call it after
    ``torch.distributed.init_process_group`` and before building the optimizer. The model is
    modified in place and returned. BatchNorm statistics stay per rank, as with DDP.
    """
    from torch.distributed.fsdp import fully_shard

    if mesh is None:
        mesh = _cpu_mesh_if_needed(model)
    opts = dict(mesh=mesh, reshard_after_forward=reshard_after_forward, **kwargs)
    if mp_policy is not None:
        opts["mp_policy"] = mp_policy
    for stage in _stages(model):
        fully_shard(stage, **opts)
    fully_shard(model, **opts)
    return model


__all__ = ["shard_model"]
