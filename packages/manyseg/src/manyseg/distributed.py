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
    stages += list(model.task_decoders.values())
    stages += list(model.heads.values())  # a ModuleDict has no forward; shard each head
    # The conditioner is *not* its own unit: DenseModel calls its FiLM / head-bias methods
    # outside Conditioner.forward, after a separate unit would have resharded them. Its
    # parameters belong to the root unit, which stays gathered for the whole forward.
    return [s for s in stages if any(p.requires_grad for p in s.parameters())]


def shard_model(model: nn.Module, *, mesh=None, mp_policy=None, reshard_after_forward=True,
                **kwargs) -> nn.Module:
    """Shard a ``DenseModel`` with FSDP2 (``fully_shard``), stage by stage, then the root.

    ``mesh`` (a ``DeviceMesh``; default: the global process group, on the model's device
    type, so a CPU model gets a CPU mesh even when a GPU is visible), ``mp_policy`` (e.g.
    ``MixedPrecisionPolicy(param_dtype=torch.bfloat16)``) and ``reshard_after_forward`` are
    passed through, as are other ``fully_shard`` keyword arguments. Call it after
    ``torch.distributed.init_process_group`` and before building the optimizer. The model is
    modified in place and returned. BatchNorm statistics stay per rank, as with DDP.
    """
    import torch
    from torch.distributed.fsdp import fully_shard

    if mesh is None:
        # fully_shard's default mesh uses the default accelerator; a CPU model on a machine
        # with a GPU would get a CUDA mesh (gathering its weights then segfaults in torch).
        p = next(model.parameters(), None)
        if p is not None and p.device.type == "cpu" and torch.cuda.is_available():
            import torch.distributed as dist
            from torch.distributed.device_mesh import init_device_mesh

            mesh = init_device_mesh("cpu", (dist.get_world_size(),))
    opts = dict(mesh=mesh, reshard_after_forward=reshard_after_forward, **kwargs)
    if mp_policy is not None:
        opts["mp_policy"] = mp_policy
    for stage in _stages(model):
        fully_shard(stage, **opts)
    fully_shard(model, **opts)
    return model


__all__ = ["shard_model"]
