"""v2bb geometry extraction and normalization tests."""

import scripts.v2bb_public_geometry_retrieval_feed_builder as engine
from tests.v2bb_test_helpers import dirs, feature, read_csv, write_geojson


def test_valid_patch_geojson_normalizes(tmp_path):
    paths = dirs(tmp_path)
    write_geojson(paths["external_dir"]/"raw"/"patch.geojson", feature("patch_boundary", "REC_00019"))
    _, summary = engine.run("normalize", **paths)
    assert summary["valid_patch_boundaries"] == 1
    assert (paths["external_dir"]/"derived"/"patch_boundary_REC_00019_normalized.geojson").is_file()


def test_unknown_crs_and_points_block(tmp_path):
    paths = dirs(tmp_path)
    write_geojson(paths["external_dir"]/"raw"/"point.geojson", feature("event_polygon", "REC_2022_05_24_30", crs="UNKNOWN", point=True))
    _, summary = engine.run("normalize", **paths)
    rows = read_csv(paths["dataset_dir"]/"v2bb_extracted_public_geometry_registry.csv")
    assert summary["valid_event_polygons"] == 0
    assert "POINT_ANCHOR_NOT_OVERLAY" in rows[0]["blocking_reason"]
    assert "CRS_UNKNOWN_OR_UNACCEPTED" in rows[0]["blocking_reason"]
