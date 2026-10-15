"""describe(): stage-level descriptions read from the built modules (structure, not strings)."""

import shutil

import pytest
import torch
from manyseg import create_model
from manyseg.describe import describe, model_graph, render_dot
from manyseg.heads.heads import PixelHead
from manyseg.instances import instance_heads
from manyseg.models import DenseModel

H, W = 65, 97
FIELDS = {"doy": {"type": "cyclic", "period": 365.25}, "latlon": {"type": "coords"}}
TWO = {"optical": {"channels": 4, "backbone": "resnet18"},
       "sar": {"channels": 2, "backbone": "convnext_atto"}}


def make(**kw):
    kw.setdefault("backbone", "resnet18")
    return create_model(pretrained=False, **kw)


def shape(t):
    return "x".join(map(str, t.shape))


def feeds(g, head):
    """The node feeding a head."""
    (src,) = g.sources(g.find("head", head).id)
    return src


def test_single_input_structure_and_shapes():
    model = make(in_channels=4, num_classes=3)
    x = torch.randn(1, 4, H, W)
    g = model_graph(model, x)
    (enc,) = g.by_stage("encoder")
    assert enc.title == "resnet18 (random init)"
    assert "4 ch" in enc.attr("stem") and "random init" in enc.attr("stem")  # nothing to adapt
    assert enc.shape("padded in") == "1x4x96x128"
    assert feeds(g, "logits").stage == "decoder"
    assert g.find("head", "logits").shape("out") == shape(model.eval()(x))
    assert not g.by_stage("fusion")
    assert sum(g.stage_params.values()) == sum(p.numel() for p in model.parameters())


def test_training_flags_restored():
    model = make(num_classes=2).train()
    model.decoder.eval()
    model_graph(model, torch.randn(1, 3, H, W))
    assert model.training and model.backbone.training and not model.decoder.training


def test_per_head_decoders_feed_their_heads():
    model = make(heads={**instance_heads(decoders={"boundary": "fpn"}),
                        "area": {"type": "regression", "channels": 1, "scope": "image"}})
    g = model_graph(model)
    assert feeds(g, "boundary").name == "decoder[boundary]"
    assert "fpn" in feeds(g, "boundary").title
    assert feeds(g, "extent").name == feeds(g, "distance").name == "decoder"
    assert feeds(g, "area").stage == "encoder"  # image heads pool the deepest encoder level


@pytest.mark.parametrize("op", ["concat", "gated", "cross_attention"])
def test_intermediate_fusion(op):
    strides = [16, 32] if op == "cross_attention" else None
    model = make(inputs=TWO, fusion={"type": "intermediate", "op": op, "strides": strides},
                 num_classes=2)
    g = model_graph(model)
    (fuse,) = g.by_stage("fusion")
    assert fuse.title == f"intermediate, op {op}"
    assert {n.name for n in g.sources(fuse.id)} == {"optical", "sar"}
    assert feeds(g, "logits").stage == "decoder"
    assert g.sources(feeds(g, "logits").id)[0] is fuse
    if strides:
        assert fuse.attr("fused strides") == "/16, /32"


def test_early_fusion_records_input_channels():
    model = make(inputs={"a": {"channels": 4}, "b": {"channels": 2}}, fusion={"type": "early"},
                 num_classes=2)
    g = model_graph(model)
    (fuse,) = g.by_stage("fusion")
    assert fuse.title.startswith("early")
    assert g.find("input", "a").title == "raster, 4 ch"
    assert g.find("input", "b").title == "raster, 2 ch"
    (enc,) = g.by_stage("encoder")
    assert g.sources(enc.id) == [fuse] and "6 ch" in enc.attr("stem")


def test_late_fusion_per_input_decoders():
    model = make(inputs=TWO, fusion={"type": "late", "op": "add"}, num_classes=2)
    g = model_graph(model)
    fuse = feeds(g, "logits")
    assert fuse.stage == "fusion" and fuse.title == "late, op add"
    decs = g.sources(fuse.id)
    assert [d.name for d in decs] == ["decoder [optical]", "decoder [sar]"]
    assert [g.sources(d.id)[0].name for d in decs] == ["optical", "sar"]


def test_shared_encoder_and_projection_adapter():
    model = make(inputs={"optical": {"channels": 4, "backbone": "resnet18"},
                         "sar": {"channels": 2, "backbone": "shared:optical"}},
                 fusion={"type": "intermediate"}, num_classes=2)
    g = model_graph(model, {"optical": torch.randn(1, 4, H, W),
                            "sar": torch.randn(1, 2, H, W)})
    (enc,) = g.by_stage("encoder")
    assert enc.name == "shared (optical, sar)"
    (adapter,) = g.by_stage("adapter")
    assert adapter.attr("channels") == "2 -> 4"
    assert {n.id for n in g.sources(enc.id)} == {"in_optical", adapter.id}
    assert enc.shape("[sar] padded in") == "1x4x96x128"


def test_same_extent_shapes_show_resampling():
    model = make(inputs={**TWO, "sar": {**TWO["sar"], "alignment": "same_extent"}},
                 fusion={"type": "intermediate"}, num_classes=2)
    sample = {"optical": torch.randn(1, 4, H, W), "sar": torch.randn(1, 2, 33, 49)}
    g = model_graph(model, sample)
    assert g.find("input", "sar").attr("alignment") == "same_extent"
    assert g.find("encoder", "sar").shape("padded in") == "1x2x64x64"
    fuse = g.by_stage("fusion")[0]
    assert "sar" in fuse.attr("resampled")
    # At /4 the sar feature arrives on the primary's padded grid (96 / 4, 128 / 4).
    assert fuse.shape("/4 in") == "1x64x24x32, 1x40x24x32"
    assert g.find("head", "logits").shape("out") == f"1x2x{H}x{W}"


@pytest.mark.parametrize("mode", ["film", "head", "channels"])
def test_metadata_targets(mode):
    cond = {"mode": "head", "pixel": True} if mode == "head" else mode
    model = make(inputs={"s2": {"channels": 13},
                         "meta": {"kind": "vector", "fields": FIELDS, "conditioning": cond}},
                 heads={"seg": {"type": "segmentation", "channels": 2, "decoder": "fpn"},
                        "dist": {"type": "regression", "channels": 1},
                        "scene": {"type": "classification", "channels": 3, "scope": "image"}})
    sample = {"s2": torch.randn(1, 13, H, W), "meta": torch.randn(1, 3)}
    g = model_graph(model, sample)
    meta = g.find("input", "meta")
    targets = {n.name for n in g.targets(meta.id, kind="condition")}
    if mode == "film":
        assert targets == {"decoder", "decoder[seg]", "scene"}  # scene pools the encoder
        assert g.find("decoder", "decoder").attr("FiLM").endswith("output yes")
    elif mode == "head":
        assert targets == {"seg", "dist", "scene"}
    else:
        assert targets == {"s2"}
        assert g.find("input", "s2").title == "raster, 13 ch + 5 metadata ch"
    out = model.eval()(sample)
    for name, t in out.items():
        assert g.find("head", name).shape("out") == shape(t)


def test_head_scales_uncertainty_and_out_stride():
    model = make(decoder_kwargs={"out_stride": 4},
                 heads={"s4": {"type": "segmentation", "channels": 2, "scale": 1 / 4},
                        "s2": {"type": "segmentation", "channels": 2, "scale": 1 / 2},
                        "s8": {"type": "regression", "channels": 1, "scale": 1 / 8},
                        "h": {"type": "regression", "channels": 1, "scope": "image",
                              "uncertainty": True}})
    g = model_graph(model, torch.randn(1, 3, H, W))
    assert g.find("decoder", "decoder").title == "unet, out_stride 4, width 64"
    assert g.find("head", "s4").attr("resample") == "/4 feature, none"
    assert g.find("head", "s2").attr("resample") == "/4 feature, bilinear x2"
    assert g.find("head", "s8").attr("resample") == "/4 feature, block mean /2"
    assert [g.find("head", n).shape("out") for n in ("s4", "s2", "s8")] == [
        "1x2x17x25", "1x2x33x49", "1x1x9x13"]
    assert g.find("head", "h").attr("uncertainty") == "yes"
    assert g.find("head", "h").shape("out") == "1x2"


def test_input_cross_and_fastai_image_edges():
    g = model_graph(make(in_channels=4, decoder_kwargs={"input_cross": True}, num_classes=2))
    dec = g.find("decoder", "decoder")
    assert dec.attr("input_cross") == "True"
    assert [n.name for n in g.sources(dec.id, kind="image")] == ["x"]
    g = model_graph(make(decoder="fastai_unet", decoder_kwargs={"out_stride": 2},
                         num_classes=2))
    assert g.find("decoder", "decoder").attr("input image") == "size only"


def test_single_output_marks_unused_heads():
    full = make(heads={"a": {"type": "segmentation", "channels": 2},
                       "b": {"type": "regression", "channels": 1}})
    model = DenseModel(full.backbone, full.decoder, dict(full.heads), single_output="a")
    g = model_graph(model, torch.randn(1, 3, H, W))
    assert g.find("head", "a").shape("out") == f"1x2x{H}x{W}"
    assert g.find("head", "b").attr("runs").startswith("no")
    assert g.find("head", "b").shape("out") is None


@pytest.mark.parametrize("backbone,layout", [("vit_tiny_patch16_224", "plain ViT"),
                                             ("pvt_v2_b0", None)])
def test_transformer_backbones(backbone, layout):
    model = make(backbone=backbone, decoder="segformer", num_classes=2)
    g = model_graph(model, torch.randn(1, 3, H, W))
    enc = g.by_stage("encoder")[0]
    assert enc.attr("levels").split(", ")[0].endswith("@/4")
    assert (enc.attr("layout") or "").startswith(layout or "")
    assert g.find("head", "logits").shape("out") == f"1x2x{H}x{W}"


def test_formats():
    model = make(inputs={"s2": {"channels": 4}, "meta": {"kind": "vector", "fields": FIELDS}},
                 num_classes=2)
    mm = describe(model, format="mermaid")
    assert mm.startswith("flowchart LR\n")
    assert "in_meta -.->|FiLM| dec_decoder" in mm
    dot = model.describe(format="dot")
    assert dot.startswith("digraph manyseg {") and dot.rstrip().endswith("}")
    assert '"in_meta" -> "dec_decoder"' in dot
    with pytest.raises(ValueError, match="unknown format"):
        describe(model, format="svg")


@pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz not installed")
def test_render_dot(tmp_path):
    src = make(num_classes=2).describe(format="dot")
    path = render_dot(src, str(tmp_path / "m.png"))
    assert (tmp_path / "m.png").read_bytes()[:4] == b"\x89PNG" and path.endswith("m.png")


def test_render_dot_without_graphviz(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="Graphviz"):
        render_dot("digraph {}", str(tmp_path / "m.png"))


GOLDEN = """\
DenseModel
  params: 13.05M
  returns: dict of 2 heads
  input: pad to multiple of 32
  params by stage: encoders 11.18M, decoders 1.87M, heads 3.33k
inputs
  x: raster, 3 ch
      primary: yes
encoders
  encoder: resnet18 (random init)  [11.18M params]
      stem: 3 ch, unchanged
      levels: 64@/2, 64@/4, 128@/8, 256@/16, 512@/32
      divisor: 32
      <- input: x
decoders
  decoder: fpn, out_stride 4, width 128  [1.87M params]
      reads: /4, /8, /16, /32
      <- encoder (/4, /8, /16, /32)
heads
  seg: segmentation, pixel, 2 ch  [2.31k params]
      from: decoder
      scale: 1
      resample: /4 feature, bilinear x4
      <- decoder
  area: regression, image, 1 ch + log-var (2 ch)  [1.03k params]
      from: encoder level 4 (512@/32)
      pooling: global average -> linear
      uncertainty: yes
      <- encoder
"""


def test_text_golden():
    model = make(decoder="fpn", heads={
        "seg": {"type": "segmentation", "channels": 2},
        "area": {"type": "regression", "channels": 1, "scope": "image", "uncertainty": True}})
    assert model.describe() == GOLDEN
    assert isinstance(model.heads["seg"], PixelHead)


def test_dot_escapes_quotes_in_user_names(tmp_path):
    """Regression (Codex review): a head named with a quote broke the DOT summary label."""
    import shutil
    import subprocess

    from manyseg import create_model

    m = create_model(backbone="resnet18", pretrained=False,
                     heads={'a"b': {"type": "segmentation", "channels": 2}})
    m.single_output = 'a"b'  # the summary line names it: returns: Tensor (head 'a"b')
    src = m.describe(format="dot")
    if shutil.which("dot") is None:
        pytest.skip("graphviz not installed")
    r = subprocess.run(["dot", "-Tsvg"], input=src, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_random_init_stem_does_not_claim_adapted_weights():
    from manyseg import create_model

    text = create_model(backbone="resnet18", pretrained=False, in_channels=4,
                        num_classes=2).describe()
    assert "random init (no pretrained weights to adapt)" in text
    assert "weights from 3 ch" not in text


def test_intermediate_fusion_image_head_reads_a_fused_level():
    from manyseg import create_model

    m = create_model(inputs={"a": {"channels": 3}, "b": {"channels": 2}}, backbone="resnet18",
                     pretrained=False, fusion={"type": "intermediate"},
                     heads={"seg": {"type": "segmentation", "channels": 2},
                            "area": {"type": "regression", "channels": 1, "scope": "image"}})
    assert "from: fused level" in m.describe()


def test_weight_shared_encoders_are_one_node_with_per_input_bn():
    """Default `shared:` encoders are weight-sharing copies with their own BatchNorm (E015);
    describe() still shows one encoder and says so, and counts every parameter once."""
    model = make(inputs={"optical": {"channels": 3, "backbone": "resnet18"},
                         "sar": {"channels": 3, "backbone": "shared:optical"}},
                 fusion={"type": "intermediate"}, num_classes=2)
    g = model_graph(model)
    (enc,) = g.by_stage("encoder")
    assert "BatchNorm per input" in enc.attr("sharing")
    assert sum(n.params or 0 for n in g.nodes.values()) == \
        sum(p.numel() for p in model.parameters())
