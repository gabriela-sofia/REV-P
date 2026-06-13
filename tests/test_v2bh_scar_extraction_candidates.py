"""v2bh observed scar extraction tests."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_scars_are_traceable_to_official_product_and_legend():
    with (ROOT / "datasets/v2bh_observed_scar_extraction_candidate_registry.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) > 0
    assert all(r["product_id"] == "MEDIA-871-16" for r in rows)
    assert all(r["matches_legend"] == "true" and r["georeferenced"] == "true" for r in rows)
    assert all(r["requires_human_review"] == "true" for r in rows)
