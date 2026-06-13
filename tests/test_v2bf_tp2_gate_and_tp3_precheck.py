"""v2bf TP2 gate and TP3 precheck tests."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_tp2_blocked_tp1_recognized_and_tp4_false():
    summary = json.loads((ROOT / "outputs_public/execution_reports/v2bf_recife_observed_event_polygon_tp2_summary.json").read_text(encoding="utf-8"))
    assert summary["tp1_available"] is True
    assert summary["tp2_gate_passed"] is False
    assert summary["tp3_precheck_ready"] is False
    assert summary["tp4_available"] is False
    with (ROOT / "datasets/v2bf_tp3_pair_precheck.csv").open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["tp1_patch_boundary_available"] == "true"
    assert row["tp2_event_polygon_available"] == "false"
