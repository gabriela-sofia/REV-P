"""Tests for v1ui — Event Geometry Candidate Audit."""

import csv
import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1ui_event_geometry_candidate_audit.py")
EVENTS = os.path.join("datasets", "protocolo_c", "event_candidate_registry.csv")


def _make_extractions(tmp_path, items):
    path = os.path.join(tmp_path, "extractions.csv")
    cols = ["extraction_id", "event_id", "artifact_id", "source_id",
            "geometry_candidate_class", "has_geometry", "geometry_type", "crs",
            "feature_count", "has_date_field", "has_hazard_field",
            "has_locality_field", "has_coordinate_fields",
            "is_observed_occurrence", "is_modeled_product", "is_event_specific",
            "can_be_observed_geometry_candidate", "can_create_ground_reference",
            "can_create_training_label", "blocking_reason"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for item in items:
            row = {c: "" for c in cols}
            row["can_create_ground_reference"] = "false"
            row["can_create_training_label"] = "false"
            row.update(item)
            writer.writerow(row)
    return path


def _run(tmp_path, items):
    ext_path = _make_extractions(tmp_path, items)
    out = os.path.join(tmp_path, "candidates.csv")
    result = subprocess.run(
        [sys.executable, SCRIPT, "--extractions", ext_path,
         "--events", EVENTS, "--out", out],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    with open(out, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestEventGeometryCandidateAudit:
    def test_empty_input(self, tmp_path):
        rows = _run(str(tmp_path), [])
        assert len(rows) == 0

    def test_g12_g13_g14_always_fail(self, tmp_path):
        rows = _run(str(tmp_path), [{
            "extraction_id": "E1", "event_id": "REC_2022_05_24_30",
            "artifact_id": "A1", "source_id": "S1",
            "can_be_observed_geometry_candidate": "true",
            "has_geometry": "true", "has_date_field": "true",
            "has_hazard_field": "true", "has_locality_field": "true",
            "has_coordinate_fields": "true", "crs": "EPSG:4326",
            "is_observed_occurrence": "true",
        }])
        assert rows[0]["G12_supervisor_review_pending"] == "FAIL"
        assert rows[0]["G13_patch_overlay_not_executed"] == "FAIL"
        assert rows[0]["G14_label_forbidden"] == "FAIL"

    def test_complete_candidate_max_status(self, tmp_path):
        rows = _run(str(tmp_path), [{
            "extraction_id": "E1", "event_id": "REC_2022_05_24_30",
            "artifact_id": "A1", "source_id": "S1",
            "can_be_observed_geometry_candidate": "true",
            "has_geometry": "true", "has_date_field": "true",
            "has_hazard_field": "true", "has_locality_field": "true",
            "has_coordinate_fields": "true", "crs": "EPSG:4326",
            "is_observed_occurrence": "true",
        }])
        assert rows[0]["max_status"] == "OBSERVED_GEOMETRY_CANDIDATE_READY_FOR_SUPERVISOR_REVIEW"
        assert rows[0]["can_create_ground_reference"] == "false"

    def test_pet_mixed_g07_fails(self, tmp_path):
        rows = _run(str(tmp_path), [{
            "extraction_id": "E1", "event_id": "PET_2022_02_15",
            "artifact_id": "A1", "source_id": "S1",
            "can_be_observed_geometry_candidate": "true",
            "has_geometry": "true", "has_date_field": "true",
            "has_hazard_field": "true", "has_locality_field": "true",
            "has_coordinate_fields": "true", "crs": "EPSG:4326",
            "is_observed_occurrence": "true",
        }])
        assert rows[0]["G07_phenomenon_separated"] == "FAIL"
        assert rows[0]["max_status"] == "CANDIDATE_WITH_BLOCKERS"

    def test_no_candidate_gets_not_geometry_status(self, tmp_path):
        rows = _run(str(tmp_path), [{
            "extraction_id": "E1", "event_id": "PET_2022_02_15",
            "artifact_id": "A1", "source_id": "S1",
            "can_be_observed_geometry_candidate": "false",
            "has_geometry": "false",
        }])
        assert rows[0]["max_status"] == "NOT_A_GEOMETRY_CANDIDATE"

    def test_never_ground_reference(self, tmp_path):
        rows = _run(str(tmp_path), [{
            "extraction_id": "E1", "event_id": "REC_2022_05_24_30",
            "artifact_id": "A1", "source_id": "S1",
            "can_be_observed_geometry_candidate": "true",
            "has_geometry": "true", "has_date_field": "true",
            "has_hazard_field": "true", "has_locality_field": "true",
            "has_coordinate_fields": "true", "crs": "EPSG:4326",
            "is_observed_occurrence": "true",
        }])
        assert rows[0]["can_create_ground_reference"] == "false"
        assert rows[0]["can_create_training_label"] == "false"
        assert "GROUND_REFERENCE" not in rows[0]["max_status"]
