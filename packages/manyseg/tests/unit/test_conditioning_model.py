"""Models conditioned on metadata vectors: FiLM / head / channels injection."""

import pytest
import torch
import torch.nn.functional as F
from manyseg import FeatureSpec, create_model
from manyseg.decoders import DECODERS
from torch import nn

FIELDS = {
    "doy": {"type": "cyclic", "period": 365.25},
    "elev": {"type": "continuous", "mean": 40.0, "std": 10.0},
    "latlon": {"type": "coords"},
    "sensor": {"type": "categorical", "num": 3},
}
HEADS = {"seg": {"type": "segmentation", "channels": 2},
         "area": {"type": "regression", "channels": 1, "scope": "image"}}


def meta(n=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.stack([torch.rand(n, generator=g) * 365, 40 + 10 * torch.randn(n, generator=g),
                        torch.rand(n, generator=g) * 180 - 90,
                        torch.rand(n, generator=g) * 360 - 180,
                        torch.randint(0, 3, (n,), generator=g).float()], dim=1)


def vec(mode="film", **kw):
    return {"kind": "vector", "fields": FIELDS, "conditioning": mode, **kw}


def build(mode="film", rasters=None, **kw):
    if "heads" not in kw:
        kw.setdefault("num_classes", 2)
    rasters = rasters or {"img": {"channels": 3}}
    return create_model(backbone="resnet18", pretrained=False,
                        inputs={**rasters, "meta": vec(mode)}, **kw)


def plain_like(cond, **kw):
    """The unconditioned model with the conditioned model's weights."""
    if "heads" not in kw:
        kw.setdefault("num_classes", 2)
    plain = create_model(backbone="resnet18", pretrained=False, **kw)
    sd = {k: v for k, v in cond.state_dict().items() if not k.startswith("conditioner.")}
    plain.load_state_dict(sd)  # strict: same architecture apart from the conditioner
    return plain.eval()


def perturb(model, scale=0.5):
    """Give the zero-initialised injection layers non-zero weights."""
    with torch.no_grad():
        for n, p in model.conditioner.named_parameters():
            if n.startswith(("film", "head_bias")):
                p.normal_(0, scale)
    return model


@pytest.mark.parametrize("decoder", sorted(DECODERS))
def test_film_identity_at_init(decoder):
    cond = build("film", decoder=decoder).eval()
    plain = plain_like(cond, decoder=decoder)
    x = torch.randn(2, 3, 33, 47)
    torch.testing.assert_close(cond({"img": x, "meta": meta()}), plain(x), rtol=0, atol=0)


def test_head_identity_at_init():
    cond = build({"mode": "head", "pixel": True}, heads=HEADS).eval()
    plain = plain_like(cond, heads=HEADS)
    x = torch.randn(2, 3, 64, 64)
    out, ref = cond({"img": x, "meta": meta()}), plain(x)
    for k in HEADS:
        torch.testing.assert_close(out[k], ref[k], rtol=0, atol=0)


def test_film_identity_with_task_decoders_and_fusion():
    heads = {"seg": {"type": "segmentation", "channels": 2},
             "depth": {"type": "regression", "channels": 1, "decoder": "fpn"}}
    rasters = {"a": {"channels": 4}, "b": {"channels": 2}}
    cond = create_model(backbone="resnet18", pretrained=False, heads=heads,
                        inputs={**rasters, "meta": vec()}).eval()
    assert set(cond.conditioner.film) == {"decoder", "task_depth"}
    plain = create_model(backbone="resnet18", pretrained=False, heads=heads, inputs=rasters)
    plain.load_state_dict({k: v for k, v in cond.state_dict().items()
                           if not k.startswith("conditioner.")})
    x = {"a": torch.randn(1, 4, 64, 96), "b": torch.randn(1, 2, 64, 96)}
    out, ref = cond({**x, "meta": meta(1)}), plain.eval()(x)
    for k in heads:
        torch.testing.assert_close(out[k], ref[k], rtol=0, atol=0)


MODES = {"film": "film", "head": {"mode": "head", "pixel": True}, "channels": "channels"}


@pytest.mark.parametrize("decoder", sorted(DECODERS))
@pytest.mark.parametrize("mode", sorted(MODES))
def test_every_decoder_and_mode(decoder, mode):
    """Odd, non-square size; dict outputs (pixel, image and a per-head task decoder)."""
    heads = {**HEADS, "edges": {"type": "segmentation", "channels": 1, "decoder": "fpn"}}
    model = perturb(build(MODES[mode], decoder=decoder, heads=heads)).eval()
    x = torch.randn(2, 3, 33, 47)
    out = model({"img": x, "meta": meta(seed=0)})
    assert out["seg"].shape == (2, 2, 33, 47) and out["edges"].shape == (2, 1, 33, 47)
    assert out["area"].shape == (2, 1)
    other = model({"img": x, "meta": meta(seed=1)})
    # film modulates decoders: an image head pooling the encoder is (by design) unaffected
    for k in (("seg", "edges") if mode == "film" else out):
        assert not torch.allclose(out[k], other[k]), f"metadata has no effect on {k}"


def test_film_levels_are_the_levels_each_decoder_reads():
    model = build("film", decoder="deeplabv3plus")
    levels = model.conditioner.film["decoder"].levels
    assert [model.feature_specs[i].stride for i in levels] == [4, 32]
    assert len(build("film", decoder="unet").conditioner.film["decoder"].levels) == 5


def test_film_on_output_only():
    model = build({"mode": "film", "on": ["output"]})
    f = model.conditioner.film["decoder"]
    assert f.levels == () and f.output is not None


def test_channels_mode_widens_the_target_input():
    model = build("channels")
    width = model.conditioner.encoders["meta"].width
    assert model.backbone.in_channels == 3 + width
    model = build({"mode": "channels", "target": "b"},
                  rasters={"a": {"channels": 4}, "b": {"channels": 2}})
    assert model.backbone.encoders["b"].in_channels == 2 + width
    assert model.backbone.encoders["a"].in_channels == 4
    out = model.eval()({"a": torch.randn(1, 4, 64, 64), "b": torch.randn(1, 2, 64, 64),
                        "meta": meta(1)})
    assert out.shape == (1, 2, 64, 64)


@pytest.mark.parametrize("ftype", ["early", "intermediate", "late"])
def test_with_raster_fusion(ftype):
    model = perturb(build("film", rasters={"a": {"channels": 4}, "b": {"channels": 2}},
                          fusion={"type": ftype})).eval()
    x = {"a": torch.randn(1, 4, 65, 97), "b": torch.randn(1, 2, 65, 97)}
    out = model({**x, "meta": meta(1)})
    assert out.shape == (1, 2, 65, 97)
    assert not torch.allclose(out, model({**x, "meta": meta(1, seed=3)}))


def test_plain_channels_vector_input():
    model = create_model(backbone="resnet18", pretrained=False, num_classes=2, inputs={
        "img": {"channels": 3}, "meta": {"kind": "vector", "channels": 4}}).eval()
    assert model({"img": torch.randn(2, 3, 64, 64), "meta": torch.randn(2, 4)}).shape == \
        (2, 2, 64, 64)


def test_two_vector_inputs_with_different_modes():
    model = perturb(create_model(backbone="resnet18", pretrained=False, heads=HEADS, inputs={
        "img": {"channels": 3}, "meta": vec(),
        "tab": {"kind": "vector", "channels": 3, "conditioning": "head"}})).eval()
    x = {"img": torch.randn(2, 3, 64, 64), "meta": meta(), "tab": torch.randn(2, 3)}
    out = model(x)
    assert not torch.allclose(out["area"], model({**x, "tab": torch.randn(2, 3)})["area"])


# --- missing values -------------------------------------------------------------------------

def test_all_metadata_missing_runs():
    model = perturb(build("film")).eval()
    x = torch.randn(2, 3, 64, 64)
    a = model({"img": x, "meta": torch.full((2, 5), float("nan"))})
    b = model({"img": x, "meta": meta(), "meta_mask": torch.zeros(2, 4)})
    assert torch.isfinite(a).all()
    torch.testing.assert_close(a, b)


def test_partial_mask_through_the_model():
    model = perturb(build("film")).eval()
    x = torch.randn(2, 3, 64, 64)
    m0, m1 = meta(seed=0), meta(seed=1)
    m1[:, 0] = m0[:, 0]  # only doy agrees; every other field differs but is masked
    mask = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(2, 4)
    torch.testing.assert_close(model({"img": x, "meta": m0, "meta_mask": mask}),
                               model({"img": x, "meta": m1, "meta_mask": mask}))
    assert not torch.allclose(model({"img": x, "meta": m0}), model({"img": x, "meta": m1}))


def test_field_dropout_in_training_mode():
    model = create_model(backbone="resnet18", pretrained=False, num_classes=2, inputs={
        "img": {"channels": 3}, "meta": {**vec(), "dropout": 0.5}}).train()
    model({"img": torch.randn(2, 3, 64, 64), "meta": meta()}).mean().backward()


def test_gradients_reach_the_conditioner():
    model = build("film").train()
    mask = torch.ones(2, 4)
    mask[0] = 0
    model({"img": torch.randn(2, 3, 64, 64), "meta": meta(), "meta_mask": mask}).mean().backward()
    for n, p in model.conditioner.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
    film = model.conditioner.film["decoder"]
    assert all(p.grad.abs().sum() > 0 for p in film.parameters())


# --- conditioning has an effect -------------------------------------------------------------

def _batch(n, seed):
    """Images are pure noise; the whole image is class 1 iff sensor == 1."""
    m = meta(n, seed=seed)
    y = (m[:, 4] == 1).long()[:, None, None].expand(n, 32, 32)
    return {"img": torch.randn(n, 3, 32, 32), "meta": m}, y


def _fit(model, steps=40):
    def run(x):
        return model(x) if model.conditioner is not None else model(x["img"])

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    for step in range(steps):
        x, y = _batch(16, seed=step)
        loss = F.cross_entropy(run(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    x, y = _batch(64, seed=12345)
    with torch.no_grad():
        return float((run(x).argmax(1) == y).float().mean())


class TinyBackbone(nn.Module):
    """Two conv stages (strides 4, 8): keeps the optimisation test fast on CPU."""

    feature_specs = (FeatureSpec(16, 4), FeatureSpec(32, 8))
    input_divisor = 8
    in_channels = 3

    def __init__(self) -> None:
        super().__init__()
        self.s4 = nn.Sequential(nn.Conv2d(3, 16, 4, stride=4), nn.ReLU())
        self.s8 = nn.Sequential(nn.Conv2d(16, 32, 2, stride=2), nn.ReLU())

    def forward(self, x):
        a = self.s4(x)
        return [a, self.s8(a)]


def test_conditioning_fits_a_metadata_only_task():
    kw = dict(pretrained=False, num_classes=2, decoder="segformer",
              decoder_kwargs={"channels": 32})
    torch.manual_seed(0)
    acc_c = _fit(create_model(backbone=TinyBackbone(), **kw,
                              inputs={"img": {"channels": 3}, "meta": vec()}))
    torch.manual_seed(0)
    acc_p = _fit(create_model(backbone=TinyBackbone(), **kw))
    assert acc_c > 0.95, acc_c
    assert acc_p < 0.8, acc_p  # majority class is ~2/3: images carry no information


# --- errors ---------------------------------------------------------------------------------

def test_vector_only_model_is_an_error():
    with pytest.raises(ValueError, match="raster"):
        create_model(backbone="resnet18", pretrained=False, num_classes=2,
                     inputs={"meta": vec()})


def test_token_mode_not_implemented():
    with pytest.raises(NotImplementedError, match="token"):
        build("token")


def test_unknown_mode_and_keys():
    with pytest.raises(ValueError, match="film"):
        build("nope")
    with pytest.raises(ValueError, match="unknown keys"):
        create_model(backbone="resnet18", pretrained=False, num_classes=2, inputs={
            "img": {"channels": 3}, "meta": {**vec(), "colour": 1}})
    with pytest.raises(ValueError, match="kind"):
        create_model(backbone="resnet18", pretrained=False, num_classes=2, inputs={
            "img": {"channels": 3, "kind": "graph"}})


def test_head_mode_without_image_heads_needs_pixel():
    with pytest.raises(ValueError, match="pixel"):
        build("head")


def test_channels_target_must_be_a_raster():
    with pytest.raises(ValueError, match="target"):
        build({"mode": "channels", "target": "sar"})


def test_dict_of_fields_rejected_at_call():
    model = build("film").eval()
    with pytest.raises(TypeError, match="pack"):
        model({"img": torch.randn(1, 3, 64, 64), "meta": {"doy": torch.zeros(1)}})


def test_missing_vector_input_and_tensor_input():
    model = build("film").eval()
    with pytest.raises(KeyError, match="meta"):
        model({"img": torch.randn(1, 3, 64, 64)})
    with pytest.raises(TypeError, match="dict"):
        model(torch.randn(1, 3, 64, 64))


def test_explicit_raster_kind_without_vectors():
    model = create_model(backbone="resnet18", pretrained=False, num_classes=2, inputs={
        "a": {"kind": "raster", "channels": 3}, "b": {"channels": 1}})
    assert model.conditioner is None
