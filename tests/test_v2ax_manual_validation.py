"""v2ax - fail-closed manual geometry validation tests."""

from __future__ import annotations

import scripts.v2ax_recife_geometry_intake_pack_engine as engine


CONFIG = dict(engine.DEFAULT_CONFIG)


def patch(**overrides):
    row = {
        "patch_id": "SYNTHETIC_PATCH", "source_type": "missing", "geometry_value": "",
        "geometry_path": "", "crs": "", "provenance_type": "unknown", "provenance_note": "",
        "license_status": "", "review_status": "not_started",
    }
    row.update(overrides)
    return row


def event(**overrides):
    row = dict(patch(**overrides))
    row.pop("patch_id")
    row["event_id"] = "SYNTHETIC_EVENT"
    return row


def test_missing_geometry_and_unknown_crs_block(tmp_path):
    missing = engine.validate_intake("patch", patch(), "", CONFIG, str(tmp_path))
    unknown = engine.validate_intake("patch", patch(
        source_type="bbox", geometry_value="0,0,10,10", crs="UNKNOWN"), "", CONFIG, str(tmp_path))
    assert missing["blocking_reason"] == "BLOCKED_PENDING_MANUAL_GEOMETRY"
    assert unknown["blocking_reason"] == "BLOCKED_UNKNOWN_CRS"
    assert missing["can_feed_v2av"] == unknown["can_feed_v2av"] == "false"


def test_point_is_rejected_for_patch_and_anchor_only_for_event(tmp_path):
    fields = {"source_type": "wkt", "geometry_value": "POINT(1 2)", "crs": "EPSG:4326"}
    patch_result = engine.validate_intake("patch", patch(**fields), "", CONFIG, str(tmp_path))
    event_result = engine.validate_intake("event", event(**fields), "", CONFIG, str(tmp_path))
    assert patch_result["blocking_reason"] == "BLOCKED_POINT_NOT_PATCH_BOUNDARY"
    assert event_result["blocking_reason"] == "POINT_ANCHOR_NOT_OVERLAY"
    assert event_result["can_feed_v2au"] == "false"


def test_real_polygon_requires_crs_provenance_license_and_review(tmp_path):
    valid = patch(source_type="wkt", geometry_value="POLYGON((0 0,10 0,10 10,0 10,0 0))",
                  crs="EPSG:3857", provenance_type="manual_digitization",
                  provenance_note="Synthetic validation source", license_status="test_license",
                  review_status="approved_for_v2av")
    result = engine.validate_intake("patch", valid, "", CONFIG, str(tmp_path))
    assert result["can_feed_v2aw"] == "true"
    assert result["can_feed_v2av"] == "true"
    assert result["blocking_reason"] == ""
