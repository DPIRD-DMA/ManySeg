"""Transformer / hybrid backbones: layout normalisation, dynamic sizes, ViT pyramid neck."""

import math

import pytest
import torch
from manyseg import create_model
from manyseg.backbones import create_backbone

HIERARCHICAL = [
    "swin_tiny_patch4_window7_224",  # NHWC natively, strict size by default
    "pvt_v2_b0",
    "mambaout_femto",  # NHWC natively
    "fastvit_t8",
    "edgenext_xx_small",
]
PLAIN_VIT = ["vit_tiny_patch16_224", "eva02_tiny_patch14_224", "deit_tiny_patch16_224"]
SIZES = [(64, 96), (97, 131)]


@pytest.mark.parametrize("name", HIERARCHICAL + PLAIN_VIT)
@pytest.mark.parametrize("hw", SIZES)
def test_features_are_nchw_and_match_specs(name, hw):
    bb = create_backbone(name, pretrained=False).eval()
    h, w = (math.ceil(v / bb.input_divisor) * bb.input_divisor for v in hw)
    feats = bb(torch.randn(1, 3, h, w))
    assert len(feats) == len(bb.feature_specs)
    for f, s in zip(feats, bb.feature_specs):
        assert f.shape[1] == s.channels, (name, f.shape, s)
        assert f.shape[-2:] == (h // s.stride, w // s.stride), (name, f.shape, s)


@pytest.mark.parametrize("name", PLAIN_VIT)
def test_plain_vit_gets_pyramid(name):
    bb = create_backbone(name, pretrained=False)
    assert bb.is_plain
    assert [s.stride for s in bb.feature_specs] == [4, 8, 16, 32]
    assert bb.input_divisor == 32


@pytest.mark.parametrize("name", PLAIN_VIT)
def test_plain_vit_takes_evenly_spaced_blocks(name):
    bb = create_backbone(name, pretrained=False)
    assert bb.block_indices == (2, 5, 8, 11)


def test_plain_vit_custom_blocks():
    bb = create_backbone("vit_tiny_patch16_224", pretrained=False, out_indices=(3, 7, 11))
    assert bb.block_indices == (3, 7, 11)
    assert [s.stride for s in bb.feature_specs] == [8, 16, 32]


@pytest.mark.parametrize("name", HIERARCHICAL + PLAIN_VIT)
def test_unet_model_odd_sizes(name):
    model = create_model(backbone=name, pretrained=False, num_classes=2).eval()
    for hw in SIZES:
        assert model(torch.randn(1, 3, *hw)).shape == (1, 2, *hw)


@pytest.mark.parametrize("name", ["vit_tiny_patch16_224", "swin_tiny_patch4_window7_224"])
def test_transformer_in_channels(name):
    model = create_model(backbone=name, pretrained=False, num_classes=2, in_channels=5).eval()
    assert model(torch.randn(1, 5, 64, 64)).shape == (1, 2, 64, 64)


@pytest.mark.parametrize("name", ["vit_tiny_patch16_224", "swin_tiny_patch4_window7_224"])
def test_transformer_gradients(name):
    model = create_model(backbone=name, pretrained=False, num_classes=2)
    model(torch.randn(2, 3, 64, 64)).mean().backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    # ViT cls token / unused final-norm params may legitimately get no gradient
    assert all(("cls_token" in n or "reg_token" in n or n.endswith("norm.weight")
                or n.endswith("norm.bias")) for n in missing), missing


def test_fixed_size_backbone_accepts_img_size():
    bb = create_backbone("hiera_tiny_224", pretrained=False, img_size=(128, 192)).eval()
    feats = bb(torch.randn(1, 3, 128, 192))
    assert feats[-1].shape[-2:] == (128 // 32, 192 // 32)
    assert bb.fixed_size == (128, 192)


def test_dynamic_backbone_has_no_fixed_size():
    assert create_backbone("vit_tiny_patch16_224", pretrained=False).fixed_size is None


def test_fixed_size_model_rejects_other_sizes_clearly():
    model = create_model(backbone="hiera_tiny_224", pretrained=False, num_classes=2,
                         backbone_kwargs={"img_size": (128, 128)}).eval()
    assert model(torch.randn(1, 3, 128, 128)).shape == (1, 2, 128, 128)
    assert model(torch.randn(1, 3, 100, 120)).shape == (1, 2, 100, 120)  # padded to 128
    with pytest.raises(ValueError, match="fixed input size"):
        model(torch.randn(1, 3, 200, 120))


def test_multi_branch_stem_all_convs_adapted():
    """Regression: fastvit lists several first convs (MobileOne branches); all must adapt."""
    bb = create_backbone("fastvit_t8", pretrained=False, in_channels=5).eval()
    assert bb(torch.randn(1, 5, 64, 64))[-1].shape[1] == bb.feature_specs[-1].channels
    assert len(bb.stem_conv_names) >= 2


@pytest.mark.parametrize("name", ["levit_128s", "efficientformerv2_s0", "coatnet_0_rw_224"])
def test_size_restricted_backbone_detected_without_img_size(name):
    """Phase 6: models that only run at their native size (window / positional constraints)
    used to fail deep in timm with shape errors at other sizes. A meta-device probe now marks
    them fixed at the native size, so smaller inputs are padded and larger ones fail clearly."""
    model = create_model(backbone=name, pretrained=False, num_classes=2).eval()
    assert model.backbone.fixed_size == (224, 224)
    assert model(torch.randn(1, 3, 97, 131)).shape == (1, 2, 97, 131)
    with pytest.raises(ValueError, match="img_size"):
        model(torch.randn(1, 3, 257, 301))


def test_any_size_backbones_stay_unrestricted():
    for name in ("resnet18", "convnext_atto", "swin_tiny_patch4_window7_224"):
        assert create_backbone(name, pretrained=False).fixed_size is None, name
