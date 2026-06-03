"""Tests for v1uh — CRS and Geometry Quality Audit."""

import csv
import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c",
                       "revp_v1uh_crs_and_geometry_quality_audit.py")


def _make_inputs(tmp_path, candidates, assets):
    cand_path = os.path.join(tmp_path, "candidates.csv")
    cand_cols = ["candidate_id", "event_id", "asset_id", "candidate_class",
                 "has_geometry", "geometry_type", "crs",
                 "can_be_ground_reference_candidate"]
    with open(cand_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cand_cols)
        writer.writeheader()
        writer.writerows(candidates)

    asset_path = os.path.join(tmp_path, "assets.csv")
    asset_cols = ["asset_id", "extension", "has_prj", "crs",
                  "feature_count", "geometry_type"]
    with open(asset_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=asset_cols)
        writer.writeheader()
        writer.writerows(assets)
    return cand_path, asset_path


def _run(tmp_path, candidates, assets):
    cand_path, asset_path = _make_inputs(tmp_path, candidates, assets)
    out = os.path.join(tmp_path, "crs_audit.csv")
    result = subprocess.run(
        [sys.executable, SCRIPT,
         "--candidates", cand_path, "--assets", asset_path, "--out", out],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    with open(out, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestCrsGeometryQualityAudit:
    def test_empty_input(self, tmp_path):
        rows = _run(str(tmp_path), [], [])
        assert len(rows) == 0

    def test_geometry_with_crs_passes(self, tmp_path):
        rows = _run(str(tmp_path),
                    [{"candidate_id": "C1", "event_id": "E1", "asset_id": "A1",
                      "candidate_class": "OBSERVED_EVENT_GEOMETRY_CANDIDATE",
                      "has_geometry": "true", "geometry_type": "Polygon",
                      "crs": "EPSG:4326",
                      "can_be_ground_reference_candidate": "true"}],
                    [{"asset_id": "A1", "extension": ".geojson", "has_prj": "",
                      "crs": "EPSG:4326", "feature_count": "10",
                      "geometry_type": "Polygon"}])
        assert rows[0]["crs_present"] == "true"
        assert rows[0]["crs_status"] == "PRESENT"
        assert rows[0]["blocking"] == "false"

    def test_geometry_without_crs_blocked(self, tmp_path):
        rows = _run(str(tmp_path),
                    [{"candidate_id": "C1", "event_id": "E1", "asset_id": "A1",
                      "candidate_class": "OBSERVED_EVENT_GEOMETRY_CANDIDATE",
                      "has_geometry": "true", "geometry_type": "Point",
                      "crs": "",
                      "can_be_ground_reference_candidate": "true"}],
                    [{"asset_id": "A1", "extension": ".shp", "has_prj": "false",
                      "crs": "", "feature_count": "5", "geometry_type": "Point"}])
        assert rows[0]["blocking"] == "true"
        assert "no_crs" in rows[0]["required_action"]

    def test_shapefile_without_prj_blocked(self, tmp_path):
        rows = _run(str(tmp_path),
                    [{"candidate_id": "C1", "event_id": "E1", "asset_id": "A1",
                      "candidate_class": "OBSERVED_EVENT_GEOMETRY_CANDIDATE",
                      "has_geometry": "true", "crs": "",
                      "can_be_ground_reference_candidate": "true"}],
                    [{"asset_id": "A1", "extension": ".shp", "has_prj": "false",
                      "crs": "", "feature_count": "", "geometry_type": ""}])
        assert rows[0]["prj_status"] == "MISSING"
        assert rows[0]["blocking"] == "true"

    def test_document_not_applicable(self, tmp_path):
        rows = _run(str(tmp_path),
                    [{"candidate_id": "C1", "event_id": "E1", "asset_id": "A1",
                      "candidate_class": "DOCUMENT_ONLY",
                      "has_geometry": "false", "crs": "",
                      "can_be_ground_reference_candidate": "false"}],
                    [{"asset_id": "A1", "extension": ".pdf", "has_prj": "",
                      "crs": "", "feature_count": "", "geometry_type": ""}])
        assert rows[0]["quality_status"] == "NOT_APPLICABLE"
        assert rows[0]["blocking"] == "false"

    def test_audit_candidate_unit(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1uh_crs_and_geometry_quality_audit import audit_candidate
        result = audit_candidate(
            {"candidate_class": "OBSERVED_EVENT_GEOMETRY_CANDIDATE",
             "has_geometry": "true", "crs": "EPSG:4326", "geometry_type": "Polygon"},
            {"crs": "EPSG:4326", "has_prj": "", "extension": ".geojson",
             "feature_count": "5", "geometry_type": "Polygon"})
        assert result["blocking"] == "false"
        assert result["crs_present"] == "true"
