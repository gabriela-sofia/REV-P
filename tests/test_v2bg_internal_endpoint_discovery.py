"""v2bg internal endpoint discovery tests."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_api_and_article_image_endpoints_are_registered():
    with (ROOT / "datasets/v2bg_charter758_internal_endpoint_discovery.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 102
    assert any(row["endpoint_type"] == "json" for row in rows)
    assert sum(row["endpoint_type"] == "article_image" for row in rows) == 100
