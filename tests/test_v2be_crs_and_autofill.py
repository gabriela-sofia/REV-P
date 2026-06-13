"""v2be CRS and reversible autofill tests."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "datasets/external_sources/recife_minimal_tp/patch_boundary_REC_00019"


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_original_and_normalized_crs_are_audited():
    row = read_rows(ROOT / "datasets/v2be_tp1_crs_reprojection_audit.csv")[0]
    assert row["original_crs"] == "EPSG:32725"
    assert row["normalized_crs"] == "EPSG:4326"
    assert row["crs_preserved"] == "true"
    assert row["crs_normalized"] == "true"
    assert row["original_bounds"] == "280830.0,9090040.0,281860.0,9091050.0"


def test_autofill_is_separate_and_original_remains_empty():
    original = read_rows(BASE / "FILL_THIS_PATCH_BOUNDARY.csv")[0]
    autofill = read_rows(BASE / "FILL_THIS_PATCH_BOUNDARY.autofill_tp1_candidate_v2be.csv")[0]
    assert original["source_type"] == "missing"
    assert original["geometry_path"] == ""
    assert autofill["crs"] == "EPSG:4326"
    assert autofill["original_crs"] == "EPSG:32725"
    assert autofill["review_status"] == "provided_unreviewed"
