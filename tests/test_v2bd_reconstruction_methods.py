"""v2bd safe and unsafe reconstruction method tests."""

import csv
from pathlib import Path

import scripts.v2bd_sentinel_patch_footprint_recovery_drilldown as engine

ROOT = Path(__file__).resolve().parents[1]


def test_unsafe_methods_are_blocked():
    with (ROOT / "datasets" / "v2bd_footprint_reconstruction_method_audit.csv").open(encoding="utf-8", newline="") as handle:
        rows = {row["method_name"]: row for row in csv.DictReader(handle)}
    assert rows["filename_inference"]["can_build_boundary"] == "false"
    assert rows["default_patch_size"]["can_build_boundary"] == "false"
    assert rows["center_plus_size"]["can_build_boundary"] == "false"
    assert rows["raster_bounds"]["can_build_boundary"] == "true"


def test_explicit_bbox_and_affine_window_helpers():
    geometry = engine.reconstruct_bbox([280830, 9090040, 281860, 9091050], "EPSG:32725")
    assert geometry["type"] == "Polygon"
    bounds = engine.reconstruct_affine([10, 0, 100, 0, -10, 200], 3, 2)
    assert bounds == [100.0, 180.0, 130.0, 200.0]
    assert engine.reconstruct_bbox(None, "EPSG:4326") is None
