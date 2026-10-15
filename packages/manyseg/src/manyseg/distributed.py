"""FSDP2 sharding for ManySeg models.

``shard_model`` applies ``torch.distributed.fsdp.fully_shard`` stage by stage: the backbone
(each input's encoder for multi-input models), the shared decoder, each task decoder, the
heads and the conditioner, then the root. One FSDP unit per stage lets FSDP overlap
communication with compute; finer units buy little at segmentation-model sizes. DDP needs
nothing from ManySeg (wrap the model as usual).
"""

from __future__ import annotations

from torch import nn


def _stages(model: nn.Module) -> list[nn.Module]:
    bb = model.backbone
    encoders = getattr(bb, "encoders", None)
    stages: list[nn.Module] = []
    if encoders is not None:  # multi-input: one unit per distinct encoder (shared ones once)
        seen = set()
        for enc in encoders.values():
            if id(enc) not in seen:
                seen.add(id(enc))
                stages.append(enc)
    else:
        stages.append(bb)
    if model.decoder is not None:
        stages.append(model.decoder)
    stages += list(model.task_decoders.values())
    stages += list(model.heads.values())  # a ModuleDict has no forward; shard each head
    if getattr(model, "conditioner", None) is not None:
        stages.append(model.conditioner)
    return [s for s in stages if any(p.requires_grad for p in s.parameters())]


def shard_model(model: nn.Module, *, mesh=None, mp_policy=None, reshard_after_forward=True,
                **kwargs) -> nn.Module:
    """Shard a ``DenseModel`` with FSDP2 (``fully_shard``), stage by stage, then the root.

    ``mesh`` (a ``DeviceMesh``; default: the global process group), ``mp_policy`` (e.g.
    ``MixedPrecisionPolicy(param_dtype=torch.bfloat16)``) and ``reshard_after_forward`` are
    passed through, as are other ``fully_shard`` keyword arguments. Call it after
    ``torch.distributed.init_process_group`` and before building the optimizer. The model is
    modified in place and returned. BatchNorm statistics stay per rank, as with DDP.
    """
    from torch.distributed.fsdp import fully_shard

    opts = dict(mesh=mesh, reshard_after_forward=reshard_after_forward, **kwargs)
    if mp_policy is not None:
        opts["mp_policy"] = mp_policy
    for stage in _stages(model):
        fully_shard(stage, **opts)
    fully_shard(model, **opts)
    return model


__all__ = ["shard_model"]
