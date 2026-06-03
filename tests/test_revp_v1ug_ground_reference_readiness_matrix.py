"""Tests for v1ug — Ground Reference Readiness Matrix."""

import csv
import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1ug_ground_reference_readiness_matrix.py")
EVENTS = os.path.join("datasets", "protocolo_c", "event_candidate_registry.csv")
GAP_MATRIX = os.path.join("datasets", "protocolo_c", "v1ug_event_gap_matrix.csv")
PACKAGES = os.path.join("datasets", "protocolo_c", "v1ug_event_review_package_registry.csv")

DIMENSIONS = [
    "temporal_readiness", "hydromet_readiness", "phenomenon_readiness",
    "locality_readiness", "geometry_readiness", "overlay_readiness",
    "supervisor_review_readiness", "label_readiness",
]

COLUMNS = (
    ["event_id", "overall_readiness"]
    + [f"{d}_status" for d in DIMENSIONS]
    + ["missing_dimensions", "blocking_dimensions_count",
       "can_create_ground_reference", "can_create_training_label",
       "next_required_action"]
)

VALID_OVERALL = {
    "NOT_READY_FOR_GROUND_REFERENCE",
    "WAITING_OBSERVED_GEOMETRY",
    "WAITING_PHENOMENON_SEPARATION",
    "READY_FOR_FORMAL_REQUEST",
    "READY_FOR_DOCUMENT_REVIEW",
}


def _run(tmp_path):
    out = os.path.join(tmp_path, "v1ug_ground_reference_readiness_matrix.csv")
    result = subprocess.run(
        [sys.executable, SCRIPT,
         "--events", EVENTS, "--gap-matrix", GAP_MATRIX,
         "--packages", PACKAGES, "--out", out],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}\nSTDOUT: {result.stdout}"
    return out


class TestGroundReferenceReadinessMatrix:
    def test_runs_and_produces_output(self, tmp_path):
        out = _run(str(tmp_path))
        assert os.path.exists(out)

    def test_columns(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            cols = csv.DictReader(f).fieldnames
        for col in COLUMNS:
            assert col in cols, f"Column missing: {col}"

    def test_one_row_per_event(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        with open(EVENTS, "r", encoding="utf-8") as f:
            n_events = sum(1 for _ in csv.DictReader(f))
        assert len(rows) == n_events

    def test_can_create_ground_reference_always_false(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["can_create_ground_reference"] == "false"
            assert r["can_create_training_label"] == "false"

    def test_geometry_and_overlay_always_fail(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["geometry_readiness_status"] == "FAIL"
            assert r["overlay_readiness_status"] == "FAIL"
            assert r["supervisor_review_readiness_status"] == "FAIL"
            assert r["label_readiness_status"] == "FAIL"

    def test_valid_overall_readiness(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["overall_readiness"] in VALID_OVERALL, (
                f"Invalid overall_readiness={r['overall_readiness']} for {r['event_id']}"
            )

    def test_blocking_count_at_least_3(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert int(r["blocking_dimensions_count"]) >= 3, (
                f"At least geometry+overlay+supervisor must block; event={r['event_id']}"
            )

    def test_pet_mixed_waiting_phenomenon(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        pet = [r for r in rows if r["event_id"].startswith("PET")]
        for r in pet:
            assert r["phenomenon_readiness_status"] == "FAIL"
            assert r["overall_readiness"] == "WAITING_PHENOMENON_SEPARATION"

    def test_evaluate_readiness_unit(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1ug_ground_reference_readiness_matrix import evaluate_readiness
        event = {"event_id": "T1", "start_date": "2022-01-01", "city": "X", "hazard_scope": "flood"}
        gaps = [
            {"event_id": "T1", "gap_name": "event_date_confirmed", "current_status": "PASS"},
            {"event_id": "T1", "gap_name": "hydromet_temporal_anchor", "current_status": "PASS"},
            {"event_id": "T1", "gap_name": "phenomenon_separated", "current_status": "PASS"},
            {"event_id": "T1", "gap_name": "locality_confirmed", "current_status": "PASS"},
            {"event_id": "T1", "gap_name": "observed_geometry_available", "current_status": "FAIL"},
        ]
        row = evaluate_readiness(event, gaps, {})
        assert row["overall_readiness"] == "WAITING_OBSERVED_GEOMETRY"
        assert row["can_create_ground_reference"] == "false"
        assert int(row["blocking_dimensions_count"]) >= 3
