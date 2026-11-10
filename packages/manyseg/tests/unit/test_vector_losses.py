"""Target encoders -> stacked batch -> the library's loss terms, as a user without the benchmark
trains. manyseg-bench's test_vector_tasks checks these values against the experiments' own code."""

import pytest
import torch

pytest.importorskip("shapely")
pytest.importorskip("networkx")
pytest.importorskip("rasterio")

import shapely  # noqa: E402
from manyseg import create_model  # noqa: E402
from manyseg.instances import instance_heads, instance_losses  # noqa: E402
from manyseg.vectors import (  # noqa: E402
    corner_line_heads,
    corner_line_losses,
    field_targets,
    footprint_targets,
    road_graph,
    road_heads,
    road_losses,
    road_targets,
)

SIZE = 64


def _batch(targets):
    return {k: torch.as_tensor(v)[None] for k, v in targets.items()}


def _model(heads):
    return create_model("resnet18", pretrained=False, heads=heads, in_channels=3)


def _trains(model, losses_of):
    """The terms are finite scalars, and their sum reaches every head."""
    out = model(torch.randn(1, 3, SIZE, SIZE))
    parts = losses_of(out)
    assert all(v.ndim == 0 and v.isfinite() for v in parts.values())
    sum(parts.values()).backward()
    assert all(p.grad is not None for p in model.heads.parameters() if p.requires_grad)
    return parts


def test_footprints_train_from_the_encoder_output():
    t = _batch(footprint_targets([shapely.box(10, 10, 40, 40)], (SIZE, SIZE)))
    parts = _trains(_model(corner_line_heads()), lambda out: corner_line_losses(out, t))
    assert set(parts) == {"instances", "corner", "direction"}


def test_fields_train_from_the_encoder_output():
    t = _batch(field_targets([shapely.box(5, 5, 30, 50), shapely.box(30, 5, 60, 50)], (SIZE, SIZE)))
    parts = _trains(_model(instance_heads()),
                    lambda out: instance_losses(out, t))
    assert set(parts) == {"extent", "boundary", "distance"}


def test_roads_train_from_the_encoder_output():
    lines = [shapely.LineString([(0, 30), (63, 30)]), shapely.LineString([(30, 0), (30, 30)])]
    g, crossings = road_graph(lines)
    t = _batch(road_targets(g, (SIZE, SIZE), crossings=crossings))
    parts = _trains(_model(road_heads()), lambda out: road_losses(out, t))
    assert set(parts) == {"mask", "line", "keypoints", "direction"}


def test_instance_targets_take_either_shape():
    out = {k: torch.zeros(2, 1, 8, 8) for k in ("extent", "boundary", "distance")}
    flat = {k: torch.ones(2, 8, 8) for k in ("extent", "boundary", "distance")}
    channel = {k: v[:, None] for k, v in flat.items()}
    a, b = instance_losses(out, flat), instance_losses(out, channel)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_a_distance_mask_limits_the_distance_term():
    out = {k: torch.zeros(1, 1, 4, 4) for k in ("extent", "boundary", "distance")}
    t = {"extent": torch.ones(1, 4, 4), "boundary": torch.zeros(1, 4, 4),
         "distance": torch.ones(1, 4, 4)}
    assert instance_losses(out, t)["distance"] == 1
    empty = {**t, "distance_mask": torch.zeros(1, 4, 4, dtype=torch.bool)}
    assert instance_losses(out, empty)["distance"] == 0


