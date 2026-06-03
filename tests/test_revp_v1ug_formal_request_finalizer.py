"""Tests for v1ug — Formal Request Finalizer."""

import csv
import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1ug_formal_request_finalizer.py")
EVENTS = os.path.join("datasets", "protocolo_c", "event_candidate_registry.csv")
TARGETS = os.path.join("configs", "protocolo_c", "v1ug_formal_request_targets.yaml")
GAP_MATRIX = os.path.join("datasets", "protocolo_c", "v1ug_event_gap_matrix.csv")

COLUMNS = [
    "request_id", "event_id", "institution_id", "institution_name",
    "priority", "gates_to_resolve", "requested_data_summary",
    "requested_formats", "contact_url", "sensitive_data_policy",
    "request_status", "human_action_required",
]


def _run(tmp_path):
    out = os.path.join(tmp_path, "v1ug_formal_request_queue.csv")
    result = subprocess.run(
        [sys.executable, SCRIPT,
         "--events", EVENTS, "--targets", TARGETS,
         "--gap-matrix", GAP_MATRIX, "--out", out],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}\nSTDOUT: {result.stdout}"
    return out


class TestFormalRequestFinalizer:
    def test_runs_and_produces_output(self, tmp_path):
        out = _run(str(tmp_path))
        assert os.path.exists(out)

    def test_columns(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            cols = csv.DictReader(f).fieldnames
        for col in COLUMNS:
            assert col in cols, f"Column missing: {col}"

    def test_all_pending_human_action(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["request_status"] == "PENDING_HUMAN_ACTION"
            assert r["human_action_required"] == "true"

    def test_request_ids_unique(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        ids = [r["request_id"] for r in rows]
        assert len(ids) == len(set(ids))

    def test_at_least_one_request_per_event(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        event_ids = {r["event_id"] for r in rows}
        assert len(event_ids) >= 2, "Expected requests for at least 2 events"

    def test_institution_contact_url_when_present(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        with_url = [r for r in rows if r["contact_url"]]
        assert len(with_url) >= 1, "At least one institution should have a contact URL"
        for r in with_url:
            assert r["contact_url"].startswith("http"), (
                f"Invalid contact URL for {r['institution_id']}: {r['contact_url']}"
            )

    def test_gates_to_resolve_not_empty(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["gates_to_resolve"], (
                f"gates_to_resolve empty for {r['request_id']}"
            )

    def test_build_request_queue_unit(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1ug_formal_request_finalizer import build_request_queue
        events = [{"event_id": "TEST_01"}]
        targets = {
            "institutions": [{
                "institution_id": "TEST_INST",
                "name": "Test",
                "applicable_events": ["TEST_01"],
                "priority": 1,
                "gates_to_resolve": ["gate_a"],
                "requested_data": ["data_a", "data_b"],
                "requested_formats": ["CSV"],
                "contact_url": "https://test.example.com",
                "sensitive_data_policy": "none",
            }]
        }
        gaps = [{"event_id": "TEST_01", "gap_name": "gate_a",
                 "current_status": "FAIL",
                 "can_be_resolved_by_formal_request": "true"}]
        result = build_request_queue(events, targets, gaps)
        assert len(result) == 1
        assert result[0]["request_status"] == "PENDING_HUMAN_ACTION"
        assert result[0]["human_action_required"] == "true"

    def test_no_request_for_unknown_event(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1ug_formal_request_finalizer import build_request_queue
        events = [{"event_id": "KNOWN_01"}]
        targets = {
            "institutions": [{
                "institution_id": "INST",
                "applicable_events": ["UNKNOWN_99"],
                "gates_to_resolve": ["g"],
                "requested_data": ["d"],
                "requested_formats": ["CSV"],
                "contact_url": "https://x.example.com",
            }]
        }
        result = build_request_queue(events, targets, [])
        assert len(result) == 0
