"""Tests for v1ug — Human Review Package Builder."""

import csv
import os
import sys
import subprocess

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1ug_human_review_package_builder.py")
EVENTS = os.path.join("datasets", "protocolo_c", "event_candidate_registry.csv")
HYDROMET_SC = os.path.join("datasets", "protocolo_c", "v1uf_event_hydromet_scorecard.csv")
V1UE_SC = os.path.join("datasets", "protocolo_c", "v1ue_event_evidence_scorecard.csv")
V1UE_RES = os.path.join("datasets", "protocolo_c", "v1ue_official_dataset_resolution_registry.csv")
ASSETS = os.path.join("datasets", "protocolo_c", "v1uf_station_series_asset_registry.csv")
POLICY = os.path.join("configs", "protocolo_c", "v1ug_review_package_policy.yaml")

PACKAGE_COLUMNS = [
    "review_package_id", "event_id", "region", "city", "event_start", "event_end",
    "current_protocol_level", "hydromet_anchor_status", "station_evidence_status",
    "official_sources_count", "local_only_assets_count", "has_event_specific_document",
    "has_official_station_series", "has_observed_geometry", "has_phenomenon_separation",
    "has_patch_overlay", "has_supervisor_review", "review_package_status",
    "reviewer_task", "cannot_promote_reason", "next_required_evidence",
]

VALID_STATUSES = {
    "READY_FOR_DOCUMENT_REVIEW",
    "READY_FOR_FORMAL_REQUEST",
    "BLOCKED_GEOMETRY_MISSING",
    "BLOCKED_PHENOMENON_SEPARATION_REQUIRED",
    "BLOCKED_INSUFFICIENT_COVERAGE",
    "CONTEXT_ONLY",
}


def _run(tmp_path):
    out = os.path.join(tmp_path, "v1ug_event_review_package_registry.csv")
    result = subprocess.run(
        [sys.executable, SCRIPT,
         "--events", EVENTS,
         "--hydromet-scorecard", HYDROMET_SC,
         "--v1ue-scorecard", V1UE_SC,
         "--v1ue-resolution", V1UE_RES,
         "--assets", ASSETS,
         "--policy", POLICY,
         "--out", out],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}\nSTDOUT: {result.stdout}"
    return out


class TestHumanReviewPackageBuilder:
    def test_runs_and_produces_output(self, tmp_path):
        out = _run(str(tmp_path))
        assert os.path.exists(out)

    def test_columns(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            cols = csv.DictReader(f).fieldnames
        for col in PACKAGE_COLUMNS:
            assert col in cols, f"Column missing: {col}"

    def test_one_package_per_event(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        with open(EVENTS, "r", encoding="utf-8") as f:
            n_events = sum(1 for _ in csv.DictReader(f))
        assert len(rows) == n_events, f"Expected {n_events} packages, got {len(rows)}"

    def test_observed_geometry_always_false(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["has_observed_geometry"] == "false", (
                f"has_observed_geometry must be false; event={r['event_id']}"
            )

    def test_patch_overlay_always_false(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["has_patch_overlay"] == "false", (
                f"has_patch_overlay must be false; event={r['event_id']}"
            )

    def test_supervisor_review_always_false(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["has_supervisor_review"] == "false", (
                f"has_supervisor_review must be false; event={r['event_id']}"
            )

    def test_cannot_promote_reason_present(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["cannot_promote_reason"], (
                f"cannot_promote_reason must not be empty; event={r['event_id']}"
            )

    def test_valid_review_package_status(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["review_package_status"] in VALID_STATUSES, (
                f"Invalid status {r['review_package_status']} for event {r['event_id']}"
            )

    def test_pet_mixed_blocked_phenomenon_separation(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        pet_mixed = [
            r for r in rows
            if r["event_id"].startswith("PET") and r["has_phenomenon_separation"] == "false"
        ]
        for r in pet_mixed:
            assert r["review_package_status"] == "BLOCKED_PHENOMENON_SEPARATION_REQUIRED", (
                f"PET mixed event must be BLOCKED_PHENOMENON_SEPARATION_REQUIRED; "
                f"got {r['review_package_status']}"
            )

    def test_review_package_ids_unique(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        ids = [r["review_package_id"] for r in rows]
        assert len(ids) == len(set(ids)), "review_package_id values are not unique"

    def test_next_required_evidence_present(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["next_required_evidence"], (
                f"next_required_evidence must not be empty; event={r['event_id']}"
            )

    def test_reviewer_task_present(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["reviewer_task"], (
                f"reviewer_task must not be empty; event={r['event_id']}"
            )

    def test_determine_package_status_mixed(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1ug_human_review_package_builder import determine_package_status
        event = {"hazard_scope": "mixed", "city": "Petrópolis"}
        status, _, _ = determine_package_status(event, {}, {})
        assert status == "BLOCKED_PHENOMENON_SEPARATION_REQUIRED"

    def test_determine_package_status_blocked_coverage(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1ug_human_review_package_builder import determine_package_status
        event = {"hazard_scope": "urban_flooding", "city": "Recife"}
        hydromet = {
            "hydromet_evidence_level": "BLOCKED_INSUFFICIENT_COVERAGE",
            "has_official_station_series": "false",
            "has_precipitation_during_event": "false",
        }
        status, _, _ = determine_package_status(event, hydromet, {})
        assert status == "BLOCKED_INSUFFICIENT_COVERAGE"

    def test_determine_package_status_formal_request_with_series(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1ug_human_review_package_builder import determine_package_status
        event = {"hazard_scope": "urban_flooding", "city": "Recife"}
        hydromet = {
            "hydromet_evidence_level": "TEMPORAL_HYDROMET_ANCHOR_CONFIRMED",
            "has_official_station_series": "true",
            "has_precipitation_during_event": "true",
        }
        status, _, _ = determine_package_status(event, hydromet, {})
        assert status == "READY_FOR_FORMAL_REQUEST"

    def test_guardrails_enforced_in_all_rows(self, tmp_path):
        """No package can have has_observed_geometry=true or has_patch_overlay=true."""
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["has_observed_geometry"] == "false"
            assert r["has_patch_overlay"] == "false"
            assert r["has_supervisor_review"] == "false"
