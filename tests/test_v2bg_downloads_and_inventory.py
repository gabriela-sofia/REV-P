"""v2bg download and inventory tests."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    with (ROOT / "datasets" / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_all_public_product_urls_are_archived_and_hashed():
    attempts = read("v2bg_charter758_download_attempts.csv")
    inventory = read("v2bg_charter758_product_file_inventory.csv")
    assert len(attempts) == 100
    assert len(inventory) == 100
    assert all(row["success"] == "true" and row["hash_sha256"] for row in attempts)
    assert all(Path(row["local_path"]).is_file() for row in inventory)
