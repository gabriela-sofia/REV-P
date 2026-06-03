"""Tests for v1ug — Event Priority Queue."""

import csv
import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1ug_event_priority_queue.py")
EVENTS = os.path.join("datasets", "protocolo_c", "event_candidate_registry.csv")
READINESS = os.path.join("datasets", "protocolo_c", "v1ug_ground_reference_readiness_matrix.csv")
GAP_MATRIX = os.path.join("datasets", "protocolo_c", "v1ug_event_gap_matrix.csv")
REQUESTS = os.path.join("datasets", "protocolo_c", "v1ug_formal_request_queue.csv")

COLUMNS = [
    "rank", "event_id", "region", "city", "review_package_status",
    "blocking_dimensions_count", "fail_gap_count", "pass_gap_count",
    "formal_request_count", "priority_score", "recommended_next_step",
    "can_create_ground_reference", "can_create_training_label",
]


def _run(tmp_path):
    out = os.path.join(tmp_path, "v1ug_event_priority_queue.csv")
    result = subprocess.run(
        [sys.executable, SCRIPT,
         "--events", EVENTS, "--readiness", READINESS,
         "--gap-matrix", GAP_MATRIX, "--requests", REQUESTS,
         "--out", out],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}\nSTDOUT: {result.stdout}"
    return out


class TestEventPriorityQueue:
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

    def test_ranks_sequential(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        ranks = [int(r["rank"]) for r in rows]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_sorted_by_priority_score_desc(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        scores = [int(r["priority_score"]) for r in rows]
        assert scores == sorted(scores, reverse=True)

    def test_can_create_ground_reference_always_false(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["can_create_ground_reference"] == "false"
            assert r["can_create_training_label"] == "false"

    def test_recommended_next_step_not_empty(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["recommended_next_step"], (
                f"Empty recommended_next_step for {r['event_id']}"
            )

    def test_compute_priority_unit(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1ug_event_priority_queue import compute_priority
        event = {"event_id": "T1", "region": "X", "city": "Y", "priority": "1"}
        readiness = {"blocking_dimensions_count": "3",
                     "overall_readiness": "WAITING_OBSERVED_GEOMETRY"}
        gaps = [
            {"event_id": "T1", "current_status": "PASS"},
            {"event_id": "T1", "current_status": "FAIL"},
        ]
        result = compute_priority(event, readiness, gaps, [])
        assert result["can_create_ground_reference"] == "false"
        assert int(result["priority_score"]) != 0
