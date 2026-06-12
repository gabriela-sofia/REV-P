import json

import pytest

import scripts.protocolo_c.revp_v2as_common as common
from tests.test_revp_v2as_common import install_all


def _prep_through_export():
    common.run_deep_probe_priority_builder(common.parse_args([]))
    common.run_geometry_payload_detector(common.parse_args([]))
    common.run_coordinate_sanity_validator(common.parse_args([]))
    common.run_geometry_candidate_classifier(common.parse_args([]))
    common.run_geojson_candidate_exporter(common.parse_args([]))


def test_validation_passes_on_null(tmp_path, monkeypatch):
    install_all(tmp_path, monkeypatch)
    _prep_through_export()
    rows = common.run_geojson_validation(common.parse_args([]))
    assert all(r["validation_status"] == "VALID" for r in rows)
    assert all(r["geometry_null_allowed"] == "true" for r in rows)


def test_validation_fails_on_geometry_without_source(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson, cache = install_all(tmp_path, monkeypatch)
    _prep_through_export()
    target = geojson / "v2as_event_geometry_pet-2022-02-15.geojson"
    fc = json.load(open(target, encoding="utf-8"))
    fc["features"][0]["geometry"] = {"type": "Point", "coordinates": [-43.0, -22.0]}
    target.write_text(json.dumps(fc), encoding="utf-8")
    with pytest.raises(ValueError):
        common.run_geojson_validation(common.parse_args([]))


def test_validation_fails_on_missing_properties(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson, cache = install_all(tmp_path, monkeypatch)
    _prep_through_export()
    target = geojson / "v2as_event_geometry_pet-2024-03-21-28.geojson"
    fc = json.load(open(target, encoding="utf-8"))
    fc["features"][0]["properties"] = {"candidate_id": "PET_2024_03_21_28"}
    target.write_text(json.dumps(fc), encoding="utf-8")
    with pytest.raises(ValueError):
        common.run_geojson_validation(common.parse_args([]))
