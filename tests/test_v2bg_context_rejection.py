"""v2bg context and municipality-scope rejection tests."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_context_and_municipality_conflict_never_feed_tp2():
    with (ROOT / "datasets/v2bg_charter758_context_rejection_audit.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert all(row["can_feed_pipeline"] == "false" for row in rows)
    assert any(row["blocking_reason"] == "MUNICIPALITY_SCOPE_CONFLICT_NOT_RECIFE_TP2" for row in rows)
