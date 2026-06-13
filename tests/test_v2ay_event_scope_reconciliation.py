"""v2ay - event scope reconciliation tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rows(name):
    with open(ROOT / "datasets" / name, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_recife_observed_is_one_not_three_and_no_event_invented():
    summary = json.loads((ROOT / "outputs_public" / "execution_reports" /
                          "v2ay_event_scope_reconciliation_turning_point_summary.json").read_text(encoding="utf-8"))
    assert summary["recife_events_expected_from_previous_prompt"] == 3
    assert summary["recife_events_observed"] == 1
    assert summary["event_scope_reconciled"] is True
    assert summary["event_invention_detected"] is False


def test_canonical_registry_separates_regions():
    events = rows("v2ay_region_event_canonical_registry.csv")
    by_region = {}
    for event in events:
        by_region.setdefault(event["region"], []).append(event["source_event_id"])
    assert by_region["Recife"] == ["REC_2022_05_24_30"]
    assert set(by_region["Petropolis"]) == {"PET_2022_02_15", "PET_2024_03_21_28"}
    assert by_region["Curitiba"] == ["CUR_EVENT_REGISTRY_MISSING"]
    assert next(e for e in events if e["region"] == "Curitiba")["event_scope_status"] == "UNRESOLVED"


def test_reconciliation_audit_explains_petropolis_confusion():
    audit = rows("v2ay_event_scope_reconciliation_audit.csv")
    divergence = next(row for row in audit if row["claim_or_expectation"] ==
                      "previous_prompt_recife_event_count")
    origin = next(row for row in audit if row["claim_or_expectation"] == "probable_origin_of_divergence")
    assert divergence["expected_value"] == "3" and divergence["observed_value"] == "1"
    assert "Petropolis" in origin["observed_value"]
