"""v2bh TP2 gate and TP3 precheck tests."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    with (ROOT / "datasets" / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_tp2_candidate_passes_but_non_intersecting_tp3_pair_is_blocked():
    gates = {r["gate_name"]: r for r in read("v2bh_tp2_georeferencing_digitization_gate.csv")}
    assert gates["TP2G_10_TP2_FEED_READY"]["gate_passed"] == "true"
    assert gates["TP2G_11_HUMAN_REVIEW_REQUIRED"]["gate_passed"] == "true"
    precheck = read("v2bh_tp3_precheck_after_charter_digitization.csv")[0]
    assert precheck["tp1_patch_boundary_available"] == "true"
    assert precheck["tp2_event_polygon_available"] == "true"
    assert precheck["ready_for_v2au_overlay"] == "false"
    assert precheck["blocking_reason"] == "TP1_TP2_NO_SPATIAL_INTERSECTION"
