"""v2bd reference inventory tests."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(name):
    with (ROOT / "datasets" / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_rec_00019_reference_inventory_is_nonempty_and_has_asset_hints():
    refs = rows("v2bd_REC_00019_reference_inventory.csv")
    assert len(refs) > 100
    assert all(row["patch_id"] == "REC_00019" for row in refs)
    assert any(row["contains_asset_hint"] == "true" for row in refs)
    assert any("asset_sanity_audit_v1fs.csv" in row["file_path"] for row in refs)
