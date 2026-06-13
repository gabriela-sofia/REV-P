"""v2bb ready feed tests."""

import scripts.v2bb_public_geometry_retrieval_feed_builder as engine
from tests.v2bb_test_helpers import dirs, feature, write_geojson


def test_valid_pair_builds_tp3_feeds_but_not_tp4(tmp_path):
    paths = dirs(tmp_path)
    write_geojson(paths["external_dir"]/"raw"/"patch.geojson", feature("patch_boundary", "REC_00019"))
    write_geojson(paths["external_dir"]/"raw"/"event.geojson", feature("event_polygon", "REC_2022_05_24_30"))
    _, summary = engine.run("build_feeds", **paths)
    assert summary["ready_patch_feed_rows"] == summary["ready_event_feed_rows"] == summary["ready_pair_feed_rows"] == 1
    assert summary["turning_point_level"] == "TP3_ONE_PATCH_EVENT_PAIR_READY_FOR_OVERLAY"


def test_canonical_public_context_creates_no_ready_feed():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for name in ("v2bb_ready_patch_boundary_feed.csv", "v2bb_ready_event_polygon_feed.csv", "v2bb_ready_turning_point_pair_feed.csv"):
        assert engine.load_csv(root/"datasets"/name) == []
