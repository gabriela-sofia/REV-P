"""Tests for v1ug — Supervisor Review Checklist Generator."""

import csv
import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1ug_supervisor_review_checklist.py")
EVENTS = os.path.join("datasets", "protocolo_c", "event_candidate_registry.csv")
POLICY = os.path.join("configs", "protocolo_c", "v1ug_supervisor_review_policy.yaml")

COLUMNS = [
    "checklist_entry_id", "event_id", "item_id", "item_text",
    "reviewer_role", "blocking_if_missing", "decision_options",
    "current_decision", "decision_rationale", "supervisor_review_completed",
]


def _run(tmp_path):
    out = os.path.join(tmp_path, "v1ug_supervisor_review_checklist.csv")
    result = subprocess.run(
        [sys.executable, SCRIPT,
         "--events", EVENTS, "--policy", POLICY, "--out", out],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}\nSTDOUT: {result.stdout}"
    return out


class TestSupervisorReviewChecklist:
    def test_runs_and_produces_output(self, tmp_path):
        out = _run(str(tmp_path))
        assert os.path.exists(out)

    def test_columns(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            cols = csv.DictReader(f).fieldnames
        for col in COLUMNS:
            assert col in cols, f"Column missing: {col}"

    def test_15_items_per_event(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        events = {}
        for r in rows:
            events.setdefault(r["event_id"], []).append(r)
        for eid, items in events.items():
            assert len(items) == 15, f"Expected 15 items for {eid}, got {len(items)}"

    def test_supervisor_review_completed_always_false(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["supervisor_review_completed"] == "false"

    def test_sr15_label_always_nao(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        sr15 = [r for r in rows if r["item_id"] == "SR_15"]
        assert len(sr15) >= 1
        for r in sr15:
            assert r["current_decision"] == "NAO", (
                f"SR_15 must default to NAO; got {r['current_decision']}"
            )

    def test_most_items_not_evaluated(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        not_eval = [r for r in rows if r["current_decision"] == "NOT_EVALUATED"]
        nao = [r for r in rows if r["current_decision"] == "NAO"]
        assert len(not_eval) + len(nao) == len(rows), (
            "All items must be NOT_EVALUATED or NAO (SR_15)"
        )

    def test_checklist_ids_unique(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        ids = [r["checklist_entry_id"] for r in rows]
        assert len(ids) == len(set(ids))

    def test_all_reviewer_role_supervisor(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["reviewer_role"] == "supervisor"

    def test_generate_checklists_unit(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1ug_supervisor_review_checklist import generate_checklists
        events = [{"event_id": "TEST_01"}]
        policy = {
            "checklist_items": [
                {"item_id": "SR_01", "item": "Test?", "reviewer_role": "supervisor",
                 "blocking_if_missing": True, "decision_options": ["SIM", "NAO"]},
                {"item_id": "SR_15", "item": "Label?", "reviewer_role": "supervisor",
                 "blocking_if_missing": False, "default_response": "NAO",
                 "decision_options": ["NAO"]},
            ]
        }
        result = generate_checklists(events, policy)
        assert len(result) == 2
        assert result[0]["current_decision"] == "NOT_EVALUATED"
        assert result[1]["current_decision"] == "NAO"
        assert all(r["supervisor_review_completed"] == "false" for r in result)
