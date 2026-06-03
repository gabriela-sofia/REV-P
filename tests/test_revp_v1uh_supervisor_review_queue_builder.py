"""Tests for v1uh — Supervisor Review Queue Builder."""

import csv
import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c",
                       "revp_v1uh_supervisor_review_queue_builder.py")

QUEUE_COLUMNS = [
    "queue_id", "event_id", "candidate_id", "institution",
    "review_priority", "review_status", "reviewer_task",
    "can_be_reviewed_now",
    "can_create_ground_reference", "can_create_training_label",
]


def _make_inputs(tmp_path, candidates, crs_audits, phenom_gates):
    cand_path = os.path.join(tmp_path, "candidates.csv")
    cand_cols = ["candidate_id", "event_id", "asset_id", "institution",
                 "candidate_class", "can_be_ground_reference_candidate",
                 "required_next_action"]
    with open(cand_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cand_cols)
        writer.writeheader()
        writer.writerows(candidates)

    crs_path = os.path.join(tmp_path, "crs.csv")
    crs_cols = ["candidate_id", "blocking", "crs_value", "required_action"]
    with open(crs_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=crs_cols)
        writer.writeheader()
        writer.writerows(crs_audits)

    gate_path = os.path.join(tmp_path, "gates.csv")
    gate_cols = ["candidate_id", "temporal_gate_status", "phenomenon_gate_status",
                 "event_date_status"]
    with open(gate_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=gate_cols)
        writer.writeheader()
        writer.writerows(phenom_gates)

    return cand_path, crs_path, gate_path


def _run(tmp_path, candidates, crs_audits, phenom_gates):
    cand_path, crs_path, gate_path = _make_inputs(
        tmp_path, candidates, crs_audits, phenom_gates)
    out = os.path.join(tmp_path, "queue.csv")
    result = subprocess.run(
        [sys.executable, SCRIPT,
         "--candidates", cand_path, "--crs-audit", crs_path,
         "--phenom-gates", gate_path, "--out", out],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    with open(out, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestSupervisorReviewQueueBuilder:
    def test_empty_input(self, tmp_path):
        rows = _run(str(tmp_path), [], [], [])
        assert len(rows) == 0

    def test_ready_candidate_enters_queue(self, tmp_path):
        rows = _run(str(tmp_path),
                    [{"candidate_id": "C1", "event_id": "E1", "asset_id": "A1",
                      "institution": "INST", "candidate_class": "OBSERVED_EVENT_GEOMETRY_CANDIDATE",
                      "can_be_ground_reference_candidate": "true",
                      "required_next_action": ""}],
                    [{"candidate_id": "C1", "blocking": "false",
                      "crs_value": "EPSG:4326", "required_action": ""}],
                    [{"candidate_id": "C1", "temporal_gate_status": "PASS",
                      "phenomenon_gate_status": "PASS", "event_date_status": "PRESENT"}])
        assert rows[0]["review_status"] == "READY_FOR_REVIEW"
        assert rows[0]["can_be_reviewed_now"] == "true"
        assert rows[0]["can_create_ground_reference"] == "false"

    def test_blocked_candidate_not_reviewable(self, tmp_path):
        rows = _run(str(tmp_path),
                    [{"candidate_id": "C1", "event_id": "E1", "asset_id": "A1",
                      "institution": "", "candidate_class": "DOCUMENT_ONLY",
                      "can_be_ground_reference_candidate": "false",
                      "required_next_action": ""}],
                    [{"candidate_id": "C1", "blocking": "false",
                      "crs_value": "", "required_action": ""}],
                    [{"candidate_id": "C1", "temporal_gate_status": "BLOCKED",
                      "phenomenon_gate_status": "BLOCKED", "event_date_status": "MISSING"}])
        assert rows[0]["review_status"] == "NOT_REVIEWABLE"
        assert rows[0]["can_be_reviewed_now"] == "false"

    def test_crs_blocked_not_immediately_reviewable(self, tmp_path):
        rows = _run(str(tmp_path),
                    [{"candidate_id": "C1", "event_id": "E1", "asset_id": "A1",
                      "institution": "", "candidate_class": "OBSERVED_EVENT_GEOMETRY_CANDIDATE",
                      "can_be_ground_reference_candidate": "true",
                      "required_next_action": ""}],
                    [{"candidate_id": "C1", "blocking": "true",
                      "crs_value": "", "required_action": "no_crs"}],
                    [{"candidate_id": "C1", "temporal_gate_status": "PASS",
                      "phenomenon_gate_status": "PASS", "event_date_status": "PRESENT"}])
        assert rows[0]["can_be_reviewed_now"] == "false"

    def test_never_auto_approves(self, tmp_path):
        rows = _run(str(tmp_path),
                    [{"candidate_id": "C1", "event_id": "E1", "asset_id": "A1",
                      "institution": "INST", "candidate_class": "OBSERVED_EVENT_GEOMETRY_CANDIDATE",
                      "can_be_ground_reference_candidate": "true",
                      "required_next_action": ""}],
                    [{"candidate_id": "C1", "blocking": "false",
                      "crs_value": "EPSG:4326", "required_action": ""}],
                    [{"candidate_id": "C1", "temporal_gate_status": "PASS",
                      "phenomenon_gate_status": "PASS", "event_date_status": "PRESENT"}])
        assert rows[0]["can_create_ground_reference"] == "false"
        assert rows[0]["can_create_training_label"] == "false"

    def test_queue_ids_unique(self, tmp_path):
        rows = _run(str(tmp_path),
                    [{"candidate_id": "C1", "event_id": "E1", "asset_id": "A1",
                      "institution": "", "candidate_class": "OBSERVED_EVENT_GEOMETRY_CANDIDATE",
                      "can_be_ground_reference_candidate": "true",
                      "required_next_action": ""},
                     {"candidate_id": "C2", "event_id": "E2", "asset_id": "A2",
                      "institution": "", "candidate_class": "TABLE_WITH_COORDINATES_CANDIDATE",
                      "can_be_ground_reference_candidate": "true",
                      "required_next_action": ""}],
                    [{"candidate_id": "C1", "blocking": "false", "crs_value": "", "required_action": ""},
                     {"candidate_id": "C2", "blocking": "false", "crs_value": "", "required_action": ""}],
                    [{"candidate_id": "C1", "temporal_gate_status": "PASS",
                      "phenomenon_gate_status": "PASS", "event_date_status": "PRESENT"},
                     {"candidate_id": "C2", "temporal_gate_status": "BLOCKED",
                      "phenomenon_gate_status": "PASS", "event_date_status": "MISSING"}])
        ids = [r["queue_id"] for r in rows]
        assert len(ids) == len(set(ids))

    def test_build_queue_unit(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1uh_supervisor_review_queue_builder import build_queue
        result = build_queue(
            [{"candidate_id": "C1", "event_id": "E1", "institution": "",
              "candidate_class": "OBSERVED_EVENT_GEOMETRY_CANDIDATE",
              "can_be_ground_reference_candidate": "true",
              "required_next_action": ""}],
            {"C1": {"blocking": "false", "crs_value": "EPSG:4326", "required_action": ""}},
            {"C1": {"temporal_gate_status": "PASS", "phenomenon_gate_status": "PASS",
                    "event_date_status": "PRESENT"}})
        assert result[0]["can_create_ground_reference"] == "false"
        assert result[0]["review_status"] == "READY_FOR_REVIEW"
