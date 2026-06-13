"""v2ay - geometry gap, absence certificate and TP contract tests."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rows(name):
    with open(ROOT / "datasets" / name, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_geometry_gaps_are_55_patches_plus_one_recife_event():
    gaps = rows("v2ay_geometry_gap_analysis.csv")
    patches = [row for row in gaps if row["target_type"] == "patch_boundary"]
    events = [row for row in gaps if row["target_type"] == "event_observed_polygon"]
    assert len(gaps) == 56 and len(patches) == 55 and len(events) == 1
    assert events[0]["target_id"] == "REC_2022_05_24_30"
    assert all(row["is_required_for_turning_point"] == "true" for row in gaps)


def test_minimum_real_geometry_contract_defines_tp0_to_tp4():
    contract = rows("v2ay_minimum_real_geometry_contract.csv")
    assert [row["turning_point_level"] for row in contract] == [
        "TP0_DOCUMENTED_ABSENCE", "TP1_ONE_PATCH_BOUNDARY_VALIDATED",
        "TP2_ONE_EVENT_POLYGON_VALIDATED", "TP3_ONE_PATCH_EVENT_PAIR_READY_FOR_OVERLAY",
        "TP4_ONE_OVERLAY_CONFIRMED_REQUIRES_HUMAN_REVIEW",
    ]
    assert all("label" in row["does_not_unlock"].lower() for row in contract)


def test_absence_certificate_is_auditable_and_point_anchors_are_not_overlay():
    certificate = rows("v2ay_spatial_metadata_absence_certificate.csv")[0]
    assert certificate["patch_boundaries_found"] == "0"
    assert certificate["event_polygons_found"] == "0"
    assert certificate["usable_overlay_geometries_found"] == "0"
    assert certificate["status"] == "TP0_DOCUMENTED_ABSENCE"
    assert "valid_point_anchors=9" in certificate["evidence_summary"]
