import hashlib
from pathlib import Path

import numpy as np
import pytest

from examples.generate_demo import generate
from geometry.ply_preserve import (
    iter_ply_vertex_chunks, read_ply_schema, transform_ply_preserving_properties,
)
from viewer.project_model import ProjectModel


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_synthetic_project_opens_and_counts_points(tmp_path):
    cloud, project_path = generate(tmp_path)
    model = ProjectModel.open(project_path)
    assert model.metadata["synthetic"] is True
    assert len(model.scans) == 1
    assert read_ply_schema(cloud).vertex.count == 1323
    assert model.scans[0].layers[0].point_count == 1323


def test_generation_is_deterministic(tmp_path):
    first, _ = generate(tmp_path / "first")
    second, _ = generate(tmp_path / "second")
    assert first.read_bytes() == second.read_bytes()


def test_transform_preserves_source_and_custom_fields(tmp_path):
    cloud, _ = generate(tmp_path)
    before = digest(cloud)
    target = tmp_path / "derived.ply"
    matrix = np.eye(4)
    matrix[:3, 3] = (0.25, 0.5, 0.75)
    transform_ply_preserving_properties(cloud, target, matrix)
    original = np.concatenate(list(iter_ply_vertex_chunks(cloud)))
    derived = np.concatenate(list(iter_ply_vertex_chunks(target)))
    assert digest(cloud) == before
    assert len(original) == len(derived) == 1323
    for field in ("red", "green", "blue", "conf_mean", "n_views"):
        np.testing.assert_array_equal(original[field], derived[field])
    for index, field in enumerate(("x", "y", "z")):
        np.testing.assert_allclose(derived[field], original[field] + matrix[index, 3], atol=1e-6)


def test_refuses_transform_over_source(tmp_path):
    cloud, _ = generate(tmp_path)
    before = digest(cloud)
    with pytest.raises(ValueError):
        transform_ply_preserving_properties(cloud, cloud, np.eye(4))
    assert digest(cloud) == before


def test_brand_assets_are_source_svg_only():
    from viewer.brand import load_brand_assets
    assets = load_brand_assets()
    assert all(path.suffix == ".svg" for path in assets.all_paths())


def test_default_project_discovery_is_opt_in(monkeypatch):
    from viewer import desktop_app
    monkeypatch.delenv("CCA_LEFEBVRE_PROJECT", raising=False)
    assert desktop_app.configured_default_project() is None
