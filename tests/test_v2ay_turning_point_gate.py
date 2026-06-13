"""v2ay - turning point gate tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rows(name):
    with open(ROOT / "datasets" / name, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_current_turning_point_is_tp0_not_ready():
    summary = json.loads((ROOT / "outputs_public" / "execution_reports" /
                          "v2ay_event_scope_reconciliation_turning_point_summary.json").read_text(encoding="utf-8"))
    assert summary["turning_point_level"] == "TP0_DOCUMENTED_ABSENCE"
    assert summary["turning_point_ready"] is False


def test_reconciliation_and_safety_gates_pass_geometry_gates_fail():
    gates = {row["gate_name"]: row for row in rows("v2ay_turning_point_readiness_gate.csv")}
    for name in ("TP_GATE_01_EVENT_SCOPE_RECONCILED", "TP_GATE_02_NO_EVENT_INVENTED",
                 "TP_GATE_12_NO_LABEL_CREATED", "TP_GATE_13_NO_MODEL_TRAINED",
                 "TP_GATE_14_C4_NOT_PROMOTED_AUTOMATICALLY"):
        assert gates[name]["gate_passed"] == "true"
    for name in ("TP_GATE_03_PATCH_BOUNDARY_SOURCE_EXISTS", "TP_GATE_04_EVENT_POLYGON_SOURCE_EXISTS",
                 "TP_GATE_10_V2AV_READY", "TP_GATE_11_V2AU_REPLAY_READY"):
        assert gates[name]["gate_passed"] == "false"
