"""Tests for v1ui — Supervisor Review Prequeue."""

import csv
import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1ui_supervisor_review_prequeue.py")


def _make_inputs(tmp_path, candidates, extractions):
    cand_path = os.path.join(tmp_path, "candidates.csv")
    cand_cols = ["candidate_audit_id", "event_id", "extraction_id", "source_id",
                 "gates_passed", "gates_failed", "max_status",
                 "can_create_ground_reference", "can_create_training_label",
                 "G01_official_public_source", "G02_artifact_traceable",
                 "G03_license_public_access", "G04_event_date_available",
                 "G05_event_date_compatible", "G06_hazard_type_available",
                 "G07_phenomenon_separated", "G08_locality_or_geometry",
                 "G09_geometry_available", "G10_crs_available",
                 "G11_geometry_quality_sufficient",
                 "G12_supervisor_review_pending", "G13_patch_overlay_not_executed",
                 "G14_label_forbidden"]
    with open(cand_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cand_cols)
        writer.writeheader()
        for c in candidates:
            row = {k: "" for k in cand_cols}
            row["can_create_ground_reference"] = "false"
            row["can_create_training_label"] = "false"
            row.update(c)
            writer.writerow(row)

    ext_path = os.path.join(tmp_path, "extractions.csv")
    ext_cols = ["extraction_id", "geometry_candidate_class", "has_geometry",
                "has_date_field", "has_hazard_field"]
    with open(ext_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ext_cols)
        writer.writeheader()
        writer.writerows(extractions)

    return cand_path, ext_path


def _run(tmp_path, candidates, extractions):
    cand_path, ext_path = _make_inputs(tmp_path, candidates, extractions)
    out = os.path.join(tmp_path, "prequeue.csv")
    result = subprocess.run(
        [sys.executable, SCRIPT, "--candidates", cand_path,
         "--extractions", ext_path, "--out", out],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    with open(out, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestSupervisorReviewPrequeue:
    def test_empty_input(self, tmp_path):
        rows = _run(str(tmp_path), [], [])
        assert len(rows) == 0

    def test_ready_candidate(self, tmp_path):
        rows = _run(str(tmp_path),
                    [{"candidate_audit_id": "G1", "event_id": "E1",
                      "extraction_id": "EX1", "source_id": "S1",
                      "gates_passed": "11", "gates_failed": "3",
                      "max_status": "OBSERVED_GEOMETRY_CANDIDATE_READY_FOR_SUPERVISOR_REVIEW"}],
                    [{"extraction_id": "EX1",
                      "geometry_candidate_class": "OBSERVED_OCCURRENCE_POLYGONS_CANDIDATE",
                      "has_geometry": "true", "has_date_field": "true",
                      "has_hazard_field": "true"}])
        assert rows[0]["review_status"] == "READY_FOR_REVIEW"
        assert rows[0]["can_be_reviewed_now"] == "true"
        assert rows[0]["can_create_ground_reference"] == "false"

    def test_blocked_candidate(self, tmp_path):
        rows = _run(str(tmp_path),
                    [{"candidate_audit_id": "G1", "event_id": "E1",
                      "extraction_id": "EX1", "source_id": "S1",
                      "gates_passed": "5", "gates_failed": "9",
                      "max_status": "CANDIDATE_WITH_BLOCKERS",
                      "G04_event_date_available": "FAIL",
                      "G06_hazard_type_available": "FAIL"}],
                    [{"extraction_id": "EX1",
                      "geometry_candidate_class": "OBSERVED_OCCURRENCE_POINTS_CANDIDATE",
                      "has_geometry": "false", "has_date_field": "false",
                      "has_hazard_field": "false"}])
        assert rows[0]["review_status"] == "BLOCKED_PENDING_GATES"
        assert rows[0]["can_be_reviewed_now"] == "false"

    def test_not_candidate(self, tmp_path):
        rows = _run(str(tmp_path),
                    [{"candidate_audit_id": "G1", "event_id": "E1",
                      "extraction_id": "EX1", "source_id": "S1",
                      "gates_passed": "0", "gates_failed": "14",
                      "max_status": "NOT_A_GEOMETRY_CANDIDATE"}],
                    [{"extraction_id": "EX1",
                      "geometry_candidate_class": "DOCUMENT_ONLY",
                      "has_geometry": "false", "has_date_field": "false",
                      "has_hazard_field": "false"}])
        assert rows[0]["review_status"] == "NOT_REVIEWABLE"

    def test_never_auto_approves(self, tmp_path):
        rows = _run(str(tmp_path),
                    [{"candidate_audit_id": "G1", "event_id": "E1",
                      "extraction_id": "EX1", "source_id": "S1",
                      "gates_passed": "11", "gates_failed": "3",
                      "max_status": "OBSERVED_GEOMETRY_CANDIDATE_READY_FOR_SUPERVISOR_REVIEW"}],
                    [{"extraction_id": "EX1",
                      "geometry_candidate_class": "OBSERVED_OCCURRENCE_POLYGONS_CANDIDATE",
                      "has_geometry": "true", "has_date_field": "true",
                      "has_hazard_field": "true"}])
        assert rows[0]["can_create_ground_reference"] == "false"
        assert rows[0]["can_create_training_label"] == "false"

    def test_prequeue_ids_unique(self, tmp_path):
        rows = _run(str(tmp_path),
                    [{"candidate_audit_id": "G1", "event_id": "E1",
                      "extraction_id": "EX1", "source_id": "S1",
                      "gates_passed": "5", "gates_failed": "9",
                      "max_status": "CANDIDATE_WITH_BLOCKERS"},
                     {"candidate_audit_id": "G2", "event_id": "E2",
                      "extraction_id": "EX2", "source_id": "S2",
                      "gates_passed": "5", "gates_failed": "9",
                      "max_status": "CANDIDATE_WITH_BLOCKERS"}],
                    [{"extraction_id": "EX1", "geometry_candidate_class": "C",
                      "has_geometry": "false", "has_date_field": "false",
                      "has_hazard_field": "false"},
                     {"extraction_id": "EX2", "geometry_candidate_class": "C",
                      "has_geometry": "false", "has_date_field": "false",
                      "has_hazard_field": "false"}])
        ids = [r["prequeue_id"] for r in rows]
        assert len(ids) == len(set(ids))
