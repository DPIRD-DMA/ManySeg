"""Boundary + distance instance outputs: targets, presets, post-processing round trips."""

import pytest
import torch
from manyseg import create_model
from manyseg.instances import (
    edt,
    instance_heads,
    instance_targets,
    instances_from_outputs,
    relabel_sequential,
)


def two_fields(h=40, w=50):
    """Background 0; field 1 and field 2 touch along a vertical edge; field 3 separate."""
    m = torch.zeros(h, w, dtype=torch.long)
    m[5:25, 5:20] = 1
    m[5:25, 20:35] = 2
    m[28:38, 38:48] = 3
    return m


def same_partition(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Label maps are equal up to a relabelling of instance ids (0 = background)."""
    if not torch.equal(a > 0, b > 0):
        return False
    pairs = torch.unique(torch.stack([a[a > 0], b[b > 0]]), dim=1)
    return pairs.shape[1] == len(torch.unique(a[a > 0])) == len(torch.unique(b[b > 0]))


# --- exact Euclidean distance transform -------------------------------------------------

def test_edt_matches_brute_force():
    torch.manual_seed(0)
    feat = torch.rand(1, 17, 23) < 0.05
    feat[0, 0, 0] = True
    d = edt(feat)
    ys, xs = torch.nonzero(feat[0], as_tuple=True)
    gy, gx = torch.meshgrid(torch.arange(17), torch.arange(23), indexing="ij")
    ref = ((gy[..., None] - ys) ** 2 + (gx[..., None] - xs) ** 2).float().min(-1).values.sqrt()
    torch.testing.assert_close(d[0], ref)


def test_edt_no_features_is_inf():
    assert torch.isinf(edt(torch.zeros(1, 4, 4, dtype=torch.bool))).all()


# --- targets ----------------------------------------------------------------------------

def test_extent_target():
    m = two_fields()
    t = instance_targets(m)
    torch.testing.assert_close(t["extent"][0, 0], (m > 0).float())


def test_distance_known_values_on_rectangle():
    m = torch.zeros(1, 11, 11, dtype=torch.long)
    m[0, 2:9, 2:9] = 1  # 7x7 square, centre at (5, 5)
    t = instance_targets(m, distance="absolute")
    d = t["distance"][0, 0]
    assert d[2, 5] == 1 and d[3, 5] == 2 and d[4, 5] == 3 and d[5, 5] == 4
    assert d[0, 0] == 0  # background


def test_distance_normalised_per_instance():
    t = instance_targets(two_fields())
    d = t["distance"][0, 0]
    m = two_fields()
    for k in (1, 2, 3):
        assert torch.isclose(d[m == k].max(), torch.tensor(1.0))
        assert d[m == k].min() > 0
    assert (d[m == 0] == 0).all()


@pytest.mark.parametrize("width", [1, 2])
def test_inner_boundary_width_and_touching_edge(width):
    m = two_fields()
    b = instance_targets(m, boundary_width=width)["boundary"][0, 0]
    # inside each instance only
    assert (b[m == 0] == 0).all()
    # the shared edge between fields 1 and 2 is marked on both sides, `width` px each
    row = b[15]
    assert row[20 - width:20 + width].eq(1).all()
    assert row[20 - width - 1] == 0 and row[20 + width] == 0
    # a pixel in the middle of field 1 is not boundary
    assert b[15, 12] == 0


def test_outer_boundary_is_outside_instances():
    m = two_fields()
    b = instance_targets(m, boundary_mode="outer")["boundary"][0, 0]
    assert (b[m > 0] == 0).all() and b.sum() > 0


def test_tile_edge_is_not_a_boundary():
    """An instance cut by the tile border must not get a boundary along the border."""
    m = torch.zeros(1, 20, 20, dtype=torch.long)
    m[0, :, :10] = 1
    b = instance_targets(m)["boundary"][0, 0]
    assert b[0, 2:8].eq(0).all() and b[10, 0] == 0
    assert b[10, 9] == 1  # real edge against background


def test_targets_batched_and_on_device(device):
    m = torch.stack([two_fields(), two_fields().flip(-1)]).to(device)
    t = instance_targets(m)
    assert t["boundary"].device.type == device.type
    assert t["distance"].shape == (2, 1, 40, 50)


def test_multiclass_extent():
    m = two_fields()
    classes = torch.zeros_like(m)
    classes[m == 3] = 2
    classes[(m == 1) | (m == 2)] = 1
    t = instance_targets(m, semantic=classes)
    assert t["extent"].shape == (1, 40, 50) and t["extent"].dtype == torch.long
    assert torch.equal(t["extent"][0], classes)


# --- post-processing round trip -----------------------------------------------------------

@pytest.mark.parametrize("backend", ["torch", "skimage"])
@pytest.mark.parametrize("width", [1, 2])
def test_round_trip_recovers_instances(backend, width):
    if backend == "skimage":
        pytest.importorskip("skimage")
    m = two_fields()
    t = instance_targets(m, boundary_width=width)
    labels = instances_from_outputs(t["extent"], t["boundary"], t["distance"], backend=backend)
    assert labels.shape == (1, 40, 50)
    assert same_partition(labels[0], m)


def test_round_trip_random_blobs():
    torch.manual_seed(3)
    m = torch.zeros(64, 64, dtype=torch.long)
    k = 1
    for y in range(4, 60, 14):
        for x in range(4, 60, 14):
            h, w = torch.randint(6, 13, (2,)).tolist()
            m[y:y + h, x:x + w] = k
            k += 1
    t = instance_targets(m)
    labels = instances_from_outputs(t["extent"], t["boundary"], t["distance"], backend="torch")
    assert same_partition(labels[0], m)


def test_postprocess_on_logits_flag():
    m = two_fields()
    t = instance_targets(m)
    logit = lambda p: torch.logit(p.clamp(1e-4, 1 - 1e-4))  # noqa: E731
    labels = instances_from_outputs(logit(t["extent"]), logit(t["boundary"]), t["distance"],
                                    logits=True, backend="torch")
    assert same_partition(labels[0], m)


def test_relabel_sequential():
    out = relabel_sequential(torch.tensor([[0, 7, 7], [3, 0, 9]]))
    assert sorted(out.unique().tolist()) == [0, 1, 2, 3]


# --- presets through create_model ---------------------------------------------------------

def test_preset_head_configs():
    h = instance_heads()
    assert set(h) == {"extent", "boundary", "distance"}
    assert h["boundary"]["channels"] == 1 and h["distance"]["type"] == "regression"
    assert instance_heads(num_classes=3)["extent"]["channels"] == 4  # + background


@pytest.mark.parametrize("decoders", [None, "separate", {"distance": "fpn"}])
@pytest.mark.parametrize("hw", [(64, 64), (65, 97)])
def test_preset_model_shapes_and_gradients(decoders, hw):
    model = create_model(backbone="resnet18", pretrained=False,
                         heads=instance_heads(decoders=decoders))
    out = model(torch.randn(2, 3, *hw))
    assert {k: tuple(v.shape) for k, v in out.items()} == {
        "extent": (2, 1, *hw), "boundary": (2, 1, *hw), "distance": (2, 1, *hw)}
    sum(v.mean() for v in out.values()).backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    if decoders == "separate":
        assert set(model.task_decoders) == {"extent", "boundary", "distance"}
