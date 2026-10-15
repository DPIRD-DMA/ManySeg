"""User-chosen names that become nn.ModuleDict keys (heads, inputs, metadata fields) must not
collide with nn.Module attributes ("half", "train", "to", ...): fail early with a clear error."""

import pytest
from manyseg import create_model

HEAD = {"type": "segmentation", "channels": 2}


@pytest.mark.parametrize("name", ["half", "train", "to", "forward", "keys", "_modules", "a.b", ""])
def test_head_names_reserved(name):
    with pytest.raises(ValueError, match=r"head name .*reserved.*'half'"):
        create_model(backbone="resnet18", pretrained=False, heads={name: HEAD})


@pytest.mark.parametrize("name", ["train", "eval", "cuda"])
def test_raster_input_names_reserved(name):
    with pytest.raises(ValueError, match="input name"):
        create_model(backbone="resnet18", pretrained=False, num_classes=2,
                     inputs={name: {"channels": 3}, "sar": {"channels": 2}})


def test_vector_input_and_field_names_reserved():
    vec = {"kind": "vector", "fields": {"doy": {"type": "cyclic", "period": 365.25}}}
    with pytest.raises(ValueError, match="input name"):
        create_model(backbone="resnet18", pretrained=False, num_classes=2,
                     inputs={"img": {"channels": 3}, "to": vec})
    bad = {"kind": "vector", "fields": {"half": {"type": "continuous", "mean": 0, "std": 1}}}
    with pytest.raises(ValueError, match="field name"):
        create_model(backbone="resnet18", pretrained=False, num_classes=2,
                     inputs={"img": {"channels": 3}, "meta": bad})


def test_ordinary_names_still_work():
    model = create_model(backbone="resnet18", pretrained=False,
                         heads={"seg": HEAD, "x2": HEAD, "logits": HEAD, "height": HEAD})
    assert set(model.heads) == {"seg", "x2", "logits", "height"}


@pytest.mark.parametrize("name", ["yield", "class", "lambda", "in"])
def test_python_keywords_rejected(name):
    """torch.export's generated code reads submodules as attributes (``self.heads.yield``),
    which is a SyntaxError for keywords; non-identifiers such as 'water-area' are fine."""
    with pytest.raises(ValueError, match="keyword"):
        create_model(backbone="resnet18", pretrained=False, heads={name: HEAD})
    with pytest.raises(ValueError, match="keyword"):
        create_model(backbone="resnet18", pretrained=False, num_classes=2,
                     inputs={name: {"channels": 3}})
