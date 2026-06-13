"""v2be replay synchronization and blocker tests."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(name):
    with (ROOT / "datasets" / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_replay_plan_is_non_destructive():
    plan = rows("v2be_tp1_replay_synchronization_plan.csv")
    assert len(plan) == 9
    assert all(row["will_run_now"] == "false" for row in plan)
    assert any(row["target_stage"] == "v2az_dry_run" for row in plan)


def test_dry_run_recognizes_tp1_and_blockers_cover_tp1_to_tp4():
    dry = rows("v2be_tp1_replay_dry_run_status.csv")[0]
    blockers = rows("v2be_remaining_turning_point_blockers.csv")
    assert dry["executed"] == "true"
    assert dry["tp1_recognized"] == "true"
    assert all(dry[field] == "false" for field in ("tp2_available", "tp3_available", "tp4_available"))
    assert len(blockers) == 4
    assert "NO_VALID_OBSERVED_EVENT_POLYGON" in {row["blocking_reason"] for row in blockers}
