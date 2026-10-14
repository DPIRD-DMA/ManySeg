"""Pretrained weights actually load (network / cache)."""

import pytest
import timm
import torch
from manyseg.backbones import create_backbone

pytestmark = pytest.mark.network


def test_pretrained_rgb_features_match_timm():
    ref = timm.create_model("resnet18", pretrained=True, features_only=True).eval()
    bb = create_backbone("resnet18", pretrained=True).eval()
    x = torch.randn(1, 3, 64, 64)
    for a, b in zip(bb(x), ref(x)):
        torch.testing.assert_close(a, b)


def test_pretrained_repeat_matches_timm_in_chans():
    """Our default 'repeat' adapter reproduces timm's own in_chans adaptation."""
    ref = timm.create_model("resnet18", pretrained=True, features_only=True, in_chans=5).eval()
    bb = create_backbone("resnet18", pretrained=True, in_channels=5).eval()
    x = torch.randn(1, 5, 64, 64)
    for a, b in zip(bb(x), ref(x)):
        torch.testing.assert_close(a, b)
