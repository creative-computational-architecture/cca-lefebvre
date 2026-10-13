"""Meaningful synthetic integration evidence for the application's survey exporter."""
import hashlib
import json
from pathlib import Path

import ezdxf
import numpy as np

from examples.synthetic_survey import generate_inputs, mathematical_room, publish_demo_outputs, run_demo


def test_mathematical_room_is_reproducible_and_fictional(tmp_path):
    first, rgb, known = mathematical_room()
    second, second_rgb, second_known = mathematical_room()
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(rgb, second_rgb)
    assert known == second_known
    assert known["synthetic"] is True
    assert known["interior_width_m"] == 8.0
    assert known["interior_depth_m"] == 5.0
    assert sum(known["primitive_point_counts"].values()) == len(first)
    one, _, _ = generate_inputs(tmp_path / "one")
    two, _, _ = generate_inputs(tmp_path / "two")
    assert hashlib.sha256(one.read_bytes()).digest() == hashlib.sha256(two.read_bytes()).digest()


def test_actual_survey_pipeline_preserves_input_and_writes_auditable_cad(tmp_path):
    summary_path, payload = run_demo(tmp_path, sheets=False)
    assert payload["input"]["raw_unchanged"] is True
    assert payload["input"]["point_count"] == payload["known_geometry"]["point_count"]
    assert payload["survey"]["extents_within_demo_tolerance"] is True
    assert payload["survey"]["walls"] >= 4
    assert payload["dxf"]["reopened_audit_errors"] == 0
    document = ezdxf.readfile(tmp_path / payload["dxf"]["path"])
    assert document.header["$INSUNITS"] == 5
    expected_layers = {"CCA_SURVEY_WALL", "CCA_SURVEY_DIM", "CCA_SURVEY_ELEV_FRONT", "CCA_SURVEY_SECTION"}
    assert expected_layers <= set(payload["dxf"]["entity_counts_by_layer"])
    assert all(payload["dxf"]["entity_counts_by_layer"][layer] > 0 for layer in expected_layers)
    text = summary_path.read_text(encoding="ascii")
    assert str(tmp_path) not in text
    assert json.loads(text)["synthetic"] is True
    publication = tmp_path / "publication"
    public_dxf = publish_demo_outputs(tmp_path, publication)
    assert len(ezdxf.readfile(public_dxf).modelspace()) == len(document.modelspace())
    public_text = public_dxf.read_text(encoding="utf-8")
    assert str(tmp_path) not in public_text
    assert "mathematical synthetic demo" in public_text
    assert "{00000000-0000-0000-0000-000000000000}" in public_text
    tags = public_text.splitlines()
    assert all(float(tags[tags.index(field) + 2]) == 2451545.0 for field in (
        "$TDCREATE", "$TDUCREATE", "$TDUPDATE", "$TDUUPDATE"))
    assert json.loads((publication / "summary.json").read_text())["dxf"]["metadata_normalized"] is True
