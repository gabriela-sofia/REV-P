"""v2ax - canonical Recife manual intake pack tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUAL = ROOT / "datasets" / "manual_intake" / "recife_p1"


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_recife_pack_has_55_fail_closed_patch_rows():
    rows = read_csv(MANUAL / "recife_p1_patch_geometry_intake.csv")
    assert len(rows) == 55
    assert len({row["patch_id"] for row in rows}) == 55
    assert all(row["region"] == "Recife" and row["priority_rank"] == "1" for row in rows)
    assert all(row["source_type"] == "missing" for row in rows)
    assert all(not row["geometry_value"] and not row["geometry_path"] for row in rows)
    assert all(row["blocking_reason"] == "BLOCKED_PENDING_MANUAL_GEOMETRY" for row in rows)


def test_only_proven_recife_events_are_listed_and_mismatch_is_explicit():
    events = read_csv(MANUAL / "recife_p1_event_geometry_intake.csv")
    assert [row["event_id"] for row in events] == ["REC_2022_05_24_30"]
    assert events[0]["required_geometry_kind"] == "observed_event_polygon"
    summary = json.loads((ROOT / "outputs_public" / "execution_reports" /
                          "v2ax_recife_geometry_intake_pack_summary.json").read_text(encoding="utf-8"))
    assert summary["total_recife_events"] == 1
    assert summary["expected_recife_events"] == 3
    assert summary["event_count_mismatch_blocker"] == "EXPECTED_RECIFE_EVENTS_NOT_FOUND"


def test_checklists_matrix_and_collection_plan_exist():
    assert len(read_csv(MANUAL / "recife_p1_patch_checklist.csv")) == 55 * 11
    assert len(read_csv(MANUAL / "recife_p1_event_checklist.csv")) == 10
    assert len(read_csv(MANUAL / "recife_p1_package_review_matrix.csv")) == 55
    assert len(read_csv(MANUAL / "recife_p1_geometry_collection_plan.csv")) == 56
