"""v2bc manual digitization queue and draft tests."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUAL = ROOT / "datasets" / "gis_workbench" / "recife_minimal_tp" / "manual_digitization"


def read(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_manual_tasks_cover_patch_event_fill_and_replay():
    tasks = read(ROOT / "datasets" / "v2bc_manual_digitization_task_queue.csv")
    kinds = {row["task_type"] for row in tasks}
    assert {"digitize_patch_boundary", "digitize_observed_event_polygon", "update_fill_this", "run_v2ba_v2az"} <= kinds


def test_fill_drafts_contain_no_fake_geometry():
    for path in MANUAL.glob("*.draft.csv"):
        row = read(path)[0]
        assert row["source_type"] == "missing"
        assert row["geometry_value"] == row["geometry_path"] == ""
        assert row["crs"] == "UNKNOWN"
