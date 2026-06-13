"""v2bf search and public-download tests."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(name):
    with (ROOT / "datasets" / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_search_plan_and_download_attempts_are_recorded():
    plan = rows("v2bf_tp2_public_event_polygon_search_plan.csv")
    attempts = rows("v2bf_tp2_public_download_attempts.csv")
    assert len(plan) >= 17
    assert len(attempts) == 5
    assert all(row["attempted"] == "true" for row in attempts)
    assert any("Charter Activation 758" in row["source_name"] for row in plan)
    assert any("Copernicus EMS" in row["source_name"] for row in plan)
