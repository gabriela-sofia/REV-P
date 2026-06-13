"""v2be integrated feed and TP1 gate tests."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_four_candidate_feeds_are_ready_and_unreviewed():
    for stage in ("v2ba", "v2aw", "v2av", "v2az"):
        rows = read_rows(ROOT / f"datasets/v2be_ready_patch_boundary_feed_for_{stage}.csv")
        assert len(rows) == 1
        assert rows[0]["ready"] == "true"
        assert rows[0]["review_status"] == "provided_unreviewed"
        assert rows[0]["requires_human_review"] == "true"


def test_all_tp1_gates_pass_and_later_tps_remain_false():
    gates = read_rows(ROOT / "datasets/v2be_tp1_readiness_gate.csv")
    assert len(gates) == 13
    assert all(row["gate_passed"] == "true" for row in gates)
    summary = json.loads((ROOT / "outputs_public/execution_reports/v2be_tp1_patch_boundary_integration_summary.json").read_text(encoding="utf-8"))
    assert summary["tp1_gate_passed"] is True
    assert summary["tp1_requires_human_review"] is True
    assert summary["tp2_available"] is False
    assert summary["tp3_available"] is False
    assert summary["tp4_available"] is False
