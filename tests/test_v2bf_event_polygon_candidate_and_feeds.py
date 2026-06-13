"""v2bf candidate and feed tests."""

import csv
from pathlib import Path

import scripts.v2bf_recife_observed_event_polygon_tp2_gate as engine

ROOT = Path(__file__).resolve().parents[1]


def count_rows(name):
    with (ROOT / "datasets" / name).open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def test_real_result_has_no_candidate_or_feed():
    assert count_rows("v2bf_REC_2022_05_24_30_observed_event_polygon_candidate_registry.csv") == 0
    for stage in ("v2ba", "v2aw", "v2au", "v2az"):
        assert count_rows(f"v2bf_ready_event_polygon_feed_for_{stage}.csv") == 0


def test_synthetic_explicit_observed_polygon_passes_without_affecting_real_outputs():
    config = {"priority_event_id": "REC_2022_05_24_30", "priority_package_id": "PKG_34713b8aab96",
              "priority_patch_id": "REC_00019", "accepted_crs": ["EPSG:4326"]}
    feature = {"type": "Feature", "properties": {"event_id": "REC_2022_05_24_30",
               "geometry_role": "observed_event_polygon", "crs": "EPSG:4326"},
               "geometry": {"type": "Polygon", "coordinates": [[[-35, -8], [-34.9, -8],
               [-34.9, -7.9], [-35, -7.9], [-35, -8]]]}}
    row = engine.candidate_from_feature(feature, {"source_name": "synthetic test"}, config)
    assert row["geometry_valid"] == "true"
    assert row["can_support_tp2"] == "true"
