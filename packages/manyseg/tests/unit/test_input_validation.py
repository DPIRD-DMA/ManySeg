"""Mistakes that used to be ignored silently or gave unhelpful errors (documentation audit)."""

import pytest
import torch
from manyseg import create_model

TWO = {"a": {"in_channels": 3}, "b": {"in_channels": 2}}
META = {"kind": "vector", "fields": {"doy": {"type": "cyclic", "period": 365.25}}}


def test_unknown_input_keys_are_errors():
    m = create_model(backbone="resnet18", pretrained=False, inputs=TWO, num_classes=2).eval()
    x = {"a": torch.randn(1, 3, 64, 64), "b": torch.randn(1, 2, 64, 64)}
    with pytest.raises(ValueError, match="unknown input keys"):
        m({**x, "dem": torch.randn(1, 1, 64, 64)})


def test_misspelt_time_series_mask_is_an_error():
    m = create_model(backbone="resnet18", pretrained=False, num_classes=2,
                     inputs={"s2": {"kind": "timeseries", "in_channels": 3}}).eval()
    x = torch.randn(1, 3, 3, 64, 64)
    mask = torch.ones(1, 3, dtype=torch.bool)
    m({"s2": x, "s2_mask": mask})
    with pytest.raises(ValueError, match="s2mask"):
        m({"s2": x, "s2mask": mask})


def test_missing_raster_and_unknown_keys_with_metadata():
    m = create_model(backbone="resnet18", pretrained=False, num_classes=2,
                     inputs={"s2": {"in_channels": 3}, "meta": META}).eval()
    meta = m.pack_metadata("meta", {"doy": torch.tensor([10.0])})
    with pytest.raises(KeyError, match="missing raster input 's2'"):
        m({"meta": meta})
    with pytest.raises(ValueError, match="meta_msk"):
        m({"s2": torch.randn(1, 3, 64, 64), "meta": meta, "meta_msk": meta})


def test_dict_to_a_single_input_model():
    m = create_model(backbone="resnet18", pretrained=False, num_classes=2).eval()
    with pytest.raises(TypeError, match="single tensor"):
        m({"image": torch.randn(1, 3, 64, 64)})


@pytest.mark.parametrize("kwargs, match", [
    ({"inputs": TWO, "in_channels": 7}, "in_channels= is for a single-input"),
    ({"num_classes": "3"}, "positive int"),
    ({"num_classes": 0}, "positive int"),
    ({"inputs": TWO, "fusion": "late"}, "fusion takes a dict"),
    ({"in_channels": 5, "backbone_kwargs": {"rgb_indices": [4, 3, 2]}}, "rgb_indices"),
])
def test_bad_arguments(kwargs, match):
    kwargs = {"num_classes": 2, **kwargs}
    with pytest.raises((ValueError, TypeError), match=match):
        create_model(backbone="resnet18", pretrained=False, **kwargs)


IMAGE_ONLY = {"c": {"type": "classification", "out_channels": 2, "scope": "image"}}


def test_decoder_arguments_are_checked_without_a_shared_decoder():
    with pytest.raises(ValueError, match="nonexistent"):
        create_model(backbone="resnet18", pretrained=False, heads=IMAGE_ONLY,
                     decoder="nonexistent")
    with pytest.raises(ValueError, match="decoder_kwargs"):
        create_model(backbone="resnet18", pretrained=False, heads=IMAGE_ONLY,
                     decoder_kwargs={"out_stride": 2})


def test_head_config_errors():
    def build(head):
        return create_model(backbone="resnet18", pretrained=False, heads={"h": head})

    with pytest.raises(ValueError, match="'scope': 'image'"):
        build({"type": "classification", "out_channels": 5})
    with pytest.raises(ValueError, match="True or False"):
        build({"type": "regression", "out_channels": 1, "uncertainty": "no"})
    with pytest.raises(ValueError, match="needs a 'type'"):
        build({"type": "segmentation", "out_channels": 2, "decoder": {"out_stride": 2}})


def test_select_heads_takes_one_name():
    m = create_model(backbone="resnet18", pretrained=False, heads={
        "ab": {"type": "segmentation", "out_channels": 2},
        "c": {"type": "segmentation", "out_channels": 1}})
    assert list(m.select_heads("ab").heads) == ["ab"]


def test_metadata_width_error_uses_the_config_name():
    with pytest.raises(ValueError, match="num_features"):
        create_model(backbone="resnet18", pretrained=False, num_classes=2, inputs={
            "s2": {"in_channels": 3}, "meta": {"kind": "vector", "num_features": 0}})


def test_dynamic_batch_needs_a_batch_of_two():
    from manyseg.export import export_model

    m = create_model(backbone="resnet18", pretrained=False, num_classes=2)
    with pytest.raises(ValueError, match="batch size >= 2"):
        export_model(m, torch.randn(1, 3, 64, 64), dynamic_batch=True)


def test_export_restores_each_module_mode():
    from manyseg.export import eval_mode

    m = create_model(backbone="resnet18", pretrained=False, num_classes=2).train()
    m.backbone.model.bn1.eval()  # frozen BN during training
    with eval_mode(m):
        assert not any(x.training for x in m.modules())
    assert m.training and not m.backbone.model.bn1.training and m.decoder.training


def test_instances_from_outputs_needs_logits_flag():
    from manyseg.instances import instances_from_outputs

    t = torch.rand(1, 1, 8, 8)
    with pytest.raises(TypeError, match="logits"):
        instances_from_outputs(t, t, t)


def test_in_channels_with_inputs_even_at_three():
    with pytest.raises(ValueError, match="in_channels= is for a single-input"):
        create_model(backbone="resnet18", pretrained=False, num_classes=2, in_channels=3,
                     inputs={"nir": {"in_channels": 1}})


def test_rgb_indices_without_adaptation():
    with pytest.raises(ValueError, match="no effect"):
        create_model(backbone="resnet18", pretrained=False, num_classes=2,
                     channel_adapter="rgb_plus_zero", backbone_kwargs={"rgb_indices": [2, 1, 0]})


def test_decoder_kwargs_unused_by_explicit_per_input_decoders():
    with pytest.raises(ValueError, match="decoder_kwargs"):
        create_model(backbone="resnet18", pretrained=False, inputs=TWO,
                     decoder_kwargs={"out_stride": 2}, heads={
                         "lc": {"type": "segmentation", "out_channels": 3, "input": "a",
                                "decoder": "fpn"}})


def test_backbone_kwarg_the_architecture_lacks_names_the_backbone():
    # efficientvit has no stochastic depth: timm's bare TypeError named only the class (#6)
    with pytest.raises(TypeError, match=r"efficientvit_b0.*drop_path_rate") as e:
        create_model(backbone="efficientvit_b0", pretrained=False, num_classes=2,
                     backbone_kwargs={"drop_path_rate": 0.1})
    assert isinstance(e.value.__cause__, TypeError)
    create_model(backbone="convnextv2_atto", pretrained=False, num_classes=2,
                 backbone_kwargs={"drop_path_rate": 0.1})
