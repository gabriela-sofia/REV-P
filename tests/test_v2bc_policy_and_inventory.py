"""v2bc context policy and inventory tests."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(name):
    with (ROOT / "datasets" / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_policy_forbids_context_promotion():
    policy = rows("v2bc_contextual_geometry_use_policy.csv")
    assert len(policy) == 4
    assert all(r["can_feed_v2aw"] == r["can_feed_v2av"] == r["can_feed_v2au"] == "false" for r in policy)
    assert all(r["can_be_ground_truth"] == "false" for r in policy)


def test_inventory_records_400_points_not_area_polygons():
    inventory = rows("v2bc_recife_risk_area_context_inventory.csv")
    assert len(inventory) == 400
    assert all(r["geometry_type"] == "Point" and r["area_m2"] == "0" for r in inventory)
    assert all(r["allowed_use"] == "context_only" for r in inventory)
