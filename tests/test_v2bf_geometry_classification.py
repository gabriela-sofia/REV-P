"""v2bf geometry classification tests."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_context_and_points_never_become_tp2():
    with (ROOT / "datasets/v2bf_tp2_event_geometry_classification.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    points = [row for row in rows if row["is_point"] == "true"]
    assert len(points) == 400
    assert all(row["can_be_tp2_candidate"] == "false" for row in rows)
    assert any(row["geometry_role"] == "visual_support_only" for row in rows)
    assert any(row["geometry_role"] == "precipitation_context" for row in rows)
