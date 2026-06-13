"""v2bg Charter probe and product enumeration tests."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(name):
    with (ROOT / "datasets" / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_activation_page_archived_and_all_products_enumerated():
    probes = rows("v2bg_charter758_activation_probe_registry.csv")
    products = rows("v2bg_charter758_product_enumeration.csv")
    assert len(probes) == 2
    assert all(row["success"] == "true" for row in probes)
    assert len(products) == 51
    priority = [row for row in products if row["is_priority_product"] == "true"]
    assert len(priority) == 1
    assert priority[0]["product_id"] == "MEDIA-871-1"
