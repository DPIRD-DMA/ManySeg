"""The composed model: backbone -> decoder -> named heads."""

from __future__ import annotations

from collections.abc import Mapping

import torch.nn.functional as F
from torch import Tensor, nn

from ..utils.spatial import crop_to, pad_to_multiple


class DenseModel(nn.Module):
    """Single-input dense prediction model.

    Args:
        backbone: Module returning ``list[Tensor]`` with ``feature_specs`` and ``input_divisor``.
        decoder: Module mapping the feature list to one tensor.
        heads: Named heads. Each consumes the decoder output or the deepest encoder feature,
            according to ``head_sources``.
        head_sources: ``{name: "decoder" | "encoder"}``; default decoder.
        single_output: If set, ``forward`` returns only this head's tensor.
        pad_to_stride: Zero-pad bottom/right to ``backbone.input_divisor`` and crop outputs
            back, which keeps outputs pixel-aligned with inputs. If False, pixel outputs are
            resized to the input size instead.
    """

    def __init__(
        self,
        backbone: nn.Module,
        decoder: nn.Module,
        heads: Mapping[str, nn.Module],
        head_sources: Mapping[str, str] | None = None,
        single_output: str | None = None,
        pad_to_stride: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.heads = nn.ModuleDict(heads)
        sources = dict(head_sources or {})
        self._head_sources = {n: sources.get(n, "decoder") for n in self.heads}
        self._pixel_heads = {n for n, h in self.heads.items() if getattr(h, "scope", "pixel") == "pixel"}
        if single_output is not None and single_output not in self.heads:
            raise ValueError(f"single_output {single_output!r} is not a head")
        self.single_output = single_output
        self.pad_to_stride = pad_to_stride
        self.divisor = int(getattr(backbone, "input_divisor", 1)) if pad_to_stride else 1
        self._deepest = max(
            range(len(backbone.feature_specs)), key=lambda i: backbone.feature_specs[i].stride
        )

    @property
    def feature_specs(self):
        return self.backbone.feature_specs

    def _run_head(self, name: str, dec: Tensor, feats: list[Tensor], h: int, w: int) -> Tensor:
        src = dec if self._head_sources[name] == "decoder" else feats[self._deepest]
        y = self.heads[name](src)
        if name in self._pixel_heads:
            if self.pad_to_stride:
                y = crop_to(y, h, w)
            else:
                y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
        return y

    def forward(self, x: Tensor) -> Tensor | dict[str, Tensor]:
        h, w = x.shape[-2], x.shape[-1]
        x = pad_to_multiple(x, self.divisor)
        feats = self.backbone(x)
        dec = self.decoder(feats)
        if self.single_output is not None:
            return self._run_head(self.single_output, dec, feats, h, w)
        return {name: self._run_head(name, dec, feats, h, w) for name in self.heads}
