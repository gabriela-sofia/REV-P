"""v2be TP1 boundary integration tests."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(name):
    with (ROOT / "datasets" / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_v2bd_geojson_is_valid_tp1_candidate():
    path = ROOT / "datasets/external_sources/recife_minimal_tp/derived/patch_boundary_REC_00019_from_lineage.geojson"
    feature = json.loads(path.read_text(encoding="utf-8"))
    row = rows("v2be_tp1_patch_boundary_integration_registry.csv")[0]
    assert feature["geometry"]["type"] in {"Polygon", "MultiPolygon"}
    assert row["patch_id"] == "REC_00019"
    assert row["geometry_valid"] == "true"
    assert row["requires_human_review"] == "true"
    assert float(row["area_m2_approx"]) > 0
    assert int(row["vertex_count"]) >= 4


def test_candidate_feeds_all_required_stages():
    row = rows("v2be_tp1_patch_boundary_integration_registry.csv")[0]
    assert all(row[field] == "true" for field in ("can_feed_v2ba", "can_feed_v2aw", "can_feed_v2av", "can_feed_v2az"))


def test_geometry_hash_preserves_v2bd_lineage():
    integrated = rows("v2be_tp1_patch_boundary_integration_registry.csv")[0]
    source = rows("v2bd_REC_00019_patch_boundary_candidate_registry.csv")[0]
    assert integrated["geometry_hash"] == source["geometry_hash"]
