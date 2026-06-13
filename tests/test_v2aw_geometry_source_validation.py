"""v2aw - geometry source validation tests."""

from __future__ import annotations

import csv
import json


def _run(engine, ds, tmp_path):
    code, _ = engine.run(dataset_dir=str(ds), output_dir=str(tmp_path / "out"),
                         config_dir=str(tmp_path / "cfg"))
    assert code == 0


def _read(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_patch(engine, dataset, tmp_path, source):
    ds = dataset(provided_patch=[source])
    _run(engine, ds, tmp_path)
    return _read(ds / "v2aw_geometry_source_validation_registry.csv")[0]


def test_missing_geometry_blocks(v2aw_engine, v2aw_dataset, tmp_path, v2aw_make_patch_source):
    source = v2aw_make_patch_source(source_type="missing", value="", crs="")
    row = _validate_patch(v2aw_engine, v2aw_dataset, tmp_path, source)
    assert row["geometry_present"] == "false"
    assert row["can_be_used_by_v2av"] == "false"
    assert row["blocking_reason"] == "BLOCKED_MISSING_GEOMETRY"


def test_unknown_crs_blocks(v2aw_engine, v2aw_dataset, tmp_path, v2aw_make_patch_source):
    row = _validate_patch(v2aw_engine, v2aw_dataset, tmp_path,
                          v2aw_make_patch_source(crs="UNKNOWN"))
    assert row["geometry_valid"] == "true"
    assert row["crs_status"] == "UNKNOWN"
    assert row["blocking_reason"] == "BLOCKED_UNKNOWN_CRS"


def test_valid_bbox_passes(v2aw_engine, v2aw_dataset, tmp_path, v2aw_make_patch_source):
    row = _validate_patch(v2aw_engine, v2aw_dataset, tmp_path,
                          v2aw_make_patch_source(source_type="bbox", value="0,0,10,10"))
    assert row["geometry_valid"] == "true"
    assert row["can_be_used_by_v2av"] == "true"
    assert row["blocking_reason"] == ""


def test_valid_wkt_passes(v2aw_engine, v2aw_dataset, tmp_path, v2aw_make_patch_source):
    source = v2aw_make_patch_source(
        source_type="wkt", value="POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")
    row = _validate_patch(v2aw_engine, v2aw_dataset, tmp_path, source)
    assert row["geometry_valid"] == "true"
    assert row["can_be_used_by_v2av"] == "true"


def test_valid_geojson_passes(v2aw_engine, v2aw_dataset, tmp_path, v2aw_make_patch_source):
    value = json.dumps({"type": "Polygon",
                        "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]})
    source = v2aw_make_patch_source(source_type="geojson_inline", value=value)
    row = _validate_patch(v2aw_engine, v2aw_dataset, tmp_path, source)
    assert row["geometry_valid"] == "true"
    assert row["can_be_used_by_v2av"] == "true"


def test_point_rejected_as_patch_boundary(v2aw_engine, v2aw_dataset, tmp_path,
                                          v2aw_make_patch_source):
    source = v2aw_make_patch_source(source_type="wkt", value="POINT(1 2)")
    row = _validate_patch(v2aw_engine, v2aw_dataset, tmp_path, source)
    assert row["geometry_valid"] == "true"
    assert row["can_be_used_by_v2av"] == "false"
    assert row["blocking_reason"] == "BLOCKED_POINT_NOT_PATCH_BOUNDARY"


def test_event_point_is_anchor_not_overlay(v2aw_engine, v2aw_dataset, tmp_path,
                                           v2aw_make_event_source):
    source = v2aw_make_event_source(source_type="wkt", value="POINT(1 2)",
                                    role="observed_event_point_anchor", crs="EPSG:4326")
    ds = v2aw_dataset(provided_event=[source])
    _run(v2aw_engine, ds, tmp_path)
    row = _read(ds / "v2aw_geometry_source_validation_registry.csv")[0]
    assert row["geometry_valid"] == "true"
    assert row["can_be_used_by_v2au"] == "false"
    assert row["blocking_reason"] == "POINT_ANCHOR_NOT_OVERLAY"
