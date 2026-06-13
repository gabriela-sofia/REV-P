"""v2bh official product inspection tests."""

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(name):
    with (ROOT / "datasets" / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_official_product_exists_and_hash_is_registered():
    row = rows("v2bh_charter_product_inspection_registry.csv")[0]
    product = Path(row["product_file"])
    assert product.is_file()
    assert row["product_id"] == "MEDIA-871-16"
    assert row["hash_sha256"] == hashlib.sha256(product.read_bytes()).hexdigest()
    assert row["contains_coordinate_grid"] == "true"
    assert row["contains_drawn_scars"] == "true"
