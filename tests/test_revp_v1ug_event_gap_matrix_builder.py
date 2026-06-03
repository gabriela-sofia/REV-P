"""Tests for v1ug — Event Gap Matrix Builder."""

import csv
import os
import sys
import subprocess

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1ug_event_gap_matrix_builder.py")
EVENTS = os.path.join("datasets", "protocolo_c", "event_candidate_registry.csv")
HYDROMET_SC = os.path.join("datasets", "protocolo_c", "v1uf_event_hydromet_scorecard.csv")
V1UE_SC = os.path.join("datasets", "protocolo_c", "v1ue_event_evidence_scorecard.csv")

GAP_COLUMNS = [
    "gap_id", "event_id", "gap_name", "current_status", "evidence_support",
    "blocking_severity", "required_action", "target_institution",
    "can_be_resolved_by_programming", "can_be_resolved_by_formal_request",
    "can_be_resolved_by_human_review", "notes",
]

EXPECTED_GAPS = [
    "event_date_confirmed",
    "official_source_traceable",
    "hydromet_temporal_anchor",
    "official_station_coordinates",
    "phenomenon_type_confirmed",
    "phenomenon_separated",
    "locality_confirmed",
    "observed_geometry_available",
    "geometry_crs_available",
    "patch_overlay_possible",
    "supervisor_review_possible",
    "training_label_allowed",
]


def _run(tmp_path):
    out = os.path.join(tmp_path, "v1ug_event_gap_matrix.csv")
    result = subprocess.run(
        [sys.executable, SCRIPT, "--events", EVENTS,
         "--hydromet-scorecard", HYDROMET_SC,
         "--v1ue-scorecard", V1UE_SC,
         "--out", out],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}\nSTDOUT: {result.stdout}"
    return out


class TestEventGapMatrix:
    def test_runs_and_produces_output(self, tmp_path):
        out = _run(str(tmp_path))
        assert os.path.exists(out)

    def test_columns(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            cols = csv.DictReader(f).fieldnames
        for col in GAP_COLUMNS:
            assert col in cols, f"Column missing: {col}"

    def test_all_gap_names_present(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        gap_names = {r["gap_name"] for r in rows}
        for g in EXPECTED_GAPS:
            assert g in gap_names, f"Gap missing: {g}"

    def test_training_label_always_fail(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        label_rows = [r for r in rows if r["gap_name"] == "training_label_allowed"]
        assert len(label_rows) >= 1
        for r in label_rows:
            assert r["current_status"] == "FAIL", (
                f"training_label_allowed must be FAIL; got {r['current_status']}"
            )

    def test_observed_geometry_always_fail(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        geom_rows = [r for r in rows if r["gap_name"] == "observed_geometry_available"]
        for r in geom_rows:
            assert r["current_status"] == "FAIL", (
                f"observed_geometry must be FAIL for all events; event={r['event_id']}"
            )

    def test_pet_mixed_events_phenomenon_fail(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        pet_phenom = [
            r for r in rows
            if r["event_id"].startswith("PET") and r["gap_name"] == "phenomenon_separated"
        ]
        assert len(pet_phenom) >= 1, "Expected PET rows for phenomenon_separated"
        for r in pet_phenom:
            assert r["current_status"] == "FAIL", (
                f"PET mixed events must FAIL phenomenon_separated; got {r['current_status']}"
            )

    def test_patch_overlay_fail_without_geometry(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        overlay_rows = [r for r in rows if r["gap_name"] == "patch_overlay_possible"]
        for r in overlay_rows:
            assert r["current_status"] == "FAIL", (
                f"patch_overlay must be FAIL without geometry; event={r['event_id']}"
            )

    def test_gap_ids_unique(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        ids = [r["gap_id"] for r in rows]
        assert len(ids) == len(set(ids)), "gap_id values are not unique"

    def test_blocking_severity_values(self, tmp_path):
        out = _run(str(tmp_path))
        valid_severities = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["blocking_severity"] in valid_severities, (
                f"Invalid severity: {r['blocking_severity']} for {r['gap_name']}"
            )

    def test_evaluate_gap_unit_date_confirmed(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1ug_event_gap_matrix_builder import evaluate_gap
        status, _ = evaluate_gap(
            "event_date_confirmed",
            {"start_date": "2022-02-15", "hazard_scope": "mixed"},
            {}, {}
        )
        assert status == "PASS"

    def test_evaluate_gap_unit_date_missing(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1ug_event_gap_matrix_builder import evaluate_gap
        status, _ = evaluate_gap(
            "event_date_confirmed",
            {"start_date": "", "hazard_scope": "flood"},
            {}, {}
        )
        assert status == "FAIL"

    def test_evaluate_gap_unit_mixed_phenomenon_fail(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1ug_event_gap_matrix_builder import evaluate_gap
        status, _ = evaluate_gap(
            "phenomenon_separated",
            {"hazard_scope": "mixed"},
            {}, {}
        )
        assert status == "FAIL"

    def test_evaluate_gap_unit_single_hazard_pass(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1ug_event_gap_matrix_builder import evaluate_gap
        status, _ = evaluate_gap(
            "phenomenon_separated",
            {"hazard_scope": "urban_flooding"},
            {}, {}
        )
        assert status == "PASS"

    def test_evaluate_gap_unit_training_label_always_fail(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1ug_event_gap_matrix_builder import evaluate_gap
        for hazard in ("mixed", "flood", "urban_flooding", "landslide"):
            status, _ = evaluate_gap(
                "training_label_allowed",
                {"hazard_scope": hazard},
                {"has_official_station_series": "true", "has_precipitation_during_event": "true"},
                {}
            )
            assert status == "FAIL", f"training_label_allowed must be FAIL for hazard={hazard}"
