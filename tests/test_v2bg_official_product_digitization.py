"""v2bg official public-product digitization tests."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_verified_recife_product_is_digitization_ready_not_tp2():
    with (ROOT / "datasets/v2bg_charter758_official_product_digitization_registry.csv").open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["product_id"] == "MEDIA-871-16"
    assert row["contains_drawn_observed_features"] == "true"
    assert row["contains_map_scale"] == "true"
    assert row["contains_north_arrow"] == "true"
    assert row["contains_coordinates_or_grid"] == "true"
    assert row["is_georeferenceable"] == "true"
    assert row["can_digitize_from_product"] == "true"
    assert row["digitization_status"] == "GEOREFERENCE_REQUIRED"
    assert Path(row["official_product_file"]).is_file()
