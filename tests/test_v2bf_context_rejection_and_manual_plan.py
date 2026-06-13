"""v2bf context rejection and manual recovery tests."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    with (ROOT / "datasets" / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_context_rejections_and_manual_plan_are_explicit():
    rejects = read("v2bf_context_rejection_audit.csv")
    assert len(rejects) == 4
    assert all(row["can_feed_pipeline"] == "false" for row in rejects)
    assert {row["blocking_reason"] for row in rejects} >= {
        "POINT_CONTEXT_NOT_OBSERVED_EVENT", "QUICKVIEW_NOT_GEOREFERENCED_VERIFIED_PRODUCT",
        "PRECIPITATION_NOT_EVENT_POLYGON"}
    assert len(read("v2bf_tp2_manual_digitization_update_plan.csv")) == 1
    recovery = ROOT / "datasets/external_sources/recife_minimal_tp/event_polygon_REC_2022_05_24_30/FILL_THIS_EVENT_POLYGON.tp2_recovery_required.csv"
    assert recovery.is_file()
