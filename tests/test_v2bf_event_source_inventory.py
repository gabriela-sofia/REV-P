"""v2bf source inventory tests."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_inventory_survives_absent_event_vector():
    with (ROOT / "datasets/v2bf_tp2_event_source_inventory.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert all(row["hash_sha256"] for row in rows)
    assert all(row["contains_context_only"] == "true" for row in rows)
