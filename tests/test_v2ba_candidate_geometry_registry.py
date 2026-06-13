"""v2ba candidate geometry validation tests."""

from tests.v2ba_test_helpers import event_wkt, patch_bbox, read_csv, run


def test_unknown_crs_and_points_are_blocked(tmp_path):
    paths, _, _ = run(tmp_path, patch=patch_bbox("UNKNOWN"), event=event_wkt(point=True))
    rows = read_csv(paths["dataset_dir"] / "v2ba_candidate_geometry_registry.csv")
    assert len(rows) == 2
    assert all(row["geometry_valid"] == "false" for row in rows)
    assert any("CRS_MISSING_OR_UNACCEPTED" in row["blocking_reason"] for row in rows)
    assert any(row["is_point"] == "true" and row["can_feed_v2au"] == "false" for row in rows)


def test_valid_bbox_and_event_wkt_pass_independently(tmp_path):
    paths, _, summary = run(tmp_path, patch=patch_bbox(), event=event_wkt())
    rows = read_csv(paths["dataset_dir"] / "v2ba_candidate_geometry_registry.csv")
    assert summary["valid_patch_boundaries"] == 1
    assert summary["valid_event_polygons"] == 1
    assert all(row["geometry_valid"] == "true" for row in rows)
