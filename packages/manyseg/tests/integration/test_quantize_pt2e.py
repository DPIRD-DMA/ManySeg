"""manyseg.quantize.quantize_pt2e: static int8 (x86 inductor quantizer) through torch.export."""

import pytest
import torch
from manyseg import create_model
from manyseg.export import export_model
from manyseg.quantize import compare_outputs, model_size, quantize_pt2e

pytestmark = pytest.mark.slow


def _batches(n, hw=(64, 96), c=3):
    return [torch.randn(1, c, *hw) for _ in range(n)]


@pytest.fixture(scope="module")
def float_and_int8():
    torch.manual_seed(0)
    model = create_model(backbone="resnet18", pretrained=False, num_classes=3).eval()
    q = quantize_pt2e(model, torch.randn(1, 3, 64, 96), _batches(4))
    return model, q


def test_pt2e_returns_exported_program(float_and_int8):
    _, q = float_and_int8
    assert isinstance(q, torch.export.ExportedProgram)
    weights = {t.dtype for t in q.state_dict.values() if t.ndim == 4}
    assert weights == {torch.int8}


def test_pt2e_dynamic_hw_shapes(float_and_int8):
    model, q = float_and_int8
    for hw in [(64, 96), (97, 131), (200, 65)]:
        assert q.module()(torch.randn(1, 3, *hw)).shape == (1, 3, *hw)


def test_pt2e_agreement_and_size(float_and_int8):
    model, q = float_and_int8
    r = compare_outputs(model, q, _batches(3, (97, 131)))
    assert r["argmax_agreement"] > 0.9
    # int8 weights, and the float originals convert_pt2e leaves registered are dropped
    assert model_size(q) < 0.35 * model_size(export_model(model, torch.randn(1, 3, 64, 96)))


def test_pt2e_dict_inputs():
    model = create_model(inputs={"a": {"channels": 4}, "b": {"channels": 2}},
                         backbone="resnet18", pretrained=False, num_classes=2).eval()

    def mk(h, w):
        return {"a": torch.randn(1, 4, h, w), "b": torch.randn(1, 2, h, w)}

    q = quantize_pt2e(model, mk(64, 96), [mk(64, 96) for _ in range(3)])
    assert q.module()(mk(97, 131)).shape == (1, 2, 97, 131)
    assert compare_outputs(model, q, [mk(97, 131)])["argmax_agreement"] > 0.9


@pytest.mark.parametrize("backbone,decoder", [
    ("resnet18", "unetplusplus"), ("resnet18", "fastai_unet"), ("resnet18", "fpn"),
    ("resnet18", "deeplabv3plus"), ("resnet18", "upernet"), ("resnet18", "segformer"),
    ("convnext_atto", "unet"), ("mobilenetv3_small_100", "unet"), ("caformer_s18", "unet"),
    ("pvt_v2_b0", "unet"), ("vit_tiny_patch16_224", "unet"),
])
def test_pt2e_backbones_and_decoders(backbone, decoder):
    model = create_model(backbone=backbone, decoder=decoder, pretrained=False,
                         num_classes=3).eval()
    q = quantize_pt2e(model, torch.randn(1, 3, 64, 96), _batches(2))
    x = [torch.randn(1, 3, 97, 131)]
    assert q.module()(x[0]).shape == (1, 3, 97, 131)
    assert compare_outputs(model, q, x)["argmax_agreement"] > 0.8


def test_pt2e_multi_head_dict_output():
    model = create_model(backbone="resnet18", pretrained=False, heads={
        "seg": {"type": "segmentation", "channels": 3},
        "height": {"type": "regression", "channels": 1},
        "scene": {"type": "classification", "channels": 4, "scope": "image"}}).eval()
    q = quantize_pt2e(model, torch.randn(1, 3, 64, 96), _batches(2))
    out = q.module()(torch.randn(1, 3, 97, 131))
    assert out["seg"].shape == (1, 3, 97, 131) and out["scene"].shape == (1, 4)
    r = compare_outputs(model, q, _batches(2, (97, 131)))
    assert r["seg/argmax_agreement"] > 0.9


def test_pt2e_dynamic_batch():
    model = create_model(backbone="resnet18", pretrained=False, num_classes=3).eval()
    q = quantize_pt2e(model, torch.randn(2, 3, 64, 96), [torch.randn(2, 3, 64, 96)],
                      dynamic_batch=True)
    assert q.module()(torch.randn(3, 3, 97, 131)).shape == (3, 3, 97, 131)


def test_pt2e_save_load_round_trip(float_and_int8, tmp_path):
    _, q = float_and_int8
    torch.export.save(q, tmp_path / "int8.pt2")
    loaded = torch.export.load(tmp_path / "int8.pt2").module()
    x = torch.randn(1, 3, 97, 131)
    torch.testing.assert_close(loaded(x), q.module()(x))


def test_pt2e_compiles_with_freezing(float_and_int8):
    """The deployment path: inductor lowers the q/dq graph to oneDNN int8 kernels. Compiled int8
    is not numerically the eager q/dq reference (documented), so only shape and finiteness."""
    import torch._inductor.config as inductor_config

    _, q = float_and_int8
    with inductor_config.patch(freezing=True), torch.no_grad():
        out = torch.compile(q.module())(torch.randn(1, 3, 97, 131))
    assert out.shape == (1, 3, 97, 131) and out.isfinite().all()


def test_pt2e_fixed_size():
    model = create_model(backbone="resnet18", pretrained=False, num_classes=3).eval()
    q = quantize_pt2e(model, torch.randn(1, 3, 64, 96), _batches(2), dynamic=False)
    assert q.module()(torch.randn(1, 3, 64, 96)).shape == (1, 3, 64, 96)


def test_pt2e_window_attention_needs_fixed_size():
    """Swin cannot export with dynamic H/W: the error names the fixed-size way out, which
    works."""
    model = create_model(backbone="swin_tiny_patch4_window7_224", pretrained=False,
                         num_classes=2).eval()
    ex = torch.randn(1, 3, 224, 224)
    with pytest.raises(ValueError, match="dynamic=False"):
        quantize_pt2e(model, ex, [ex])
    q = quantize_pt2e(model, ex, [ex], dynamic=False)
    assert q.module()(ex).shape == (1, 2, 224, 224)


def test_pt2e_rejects_empty_calibration():
    model = create_model(backbone="resnet18", pretrained=False, num_classes=3).eval()
    with pytest.raises(ValueError, match="calibration"):
        quantize_pt2e(model, torch.randn(1, 3, 64, 96), [])
