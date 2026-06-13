"""v2ba ready feeds and turning-point gate tests."""

from tests.v2ba_test_helpers import event_wkt, patch_bbox, read_csv, run


def test_no_external_files_means_empty_feeds_and_tp0(tmp_path):
    paths, code, summary = run(tmp_path)
    assert code == 0
    assert summary["turning_point_level"] == "TP0_DOCUMENTED_ABSENCE_WITH_ACQUISITION_DOSSIER"
    for name in ("v2ba_ready_patch_boundary_feed.csv", "v2ba_ready_event_polygon_feed.csv",
                 "v2ba_ready_turning_point_pair_feed.csv"):
        assert read_csv(paths["dataset_dir"] / name) == []


def test_valid_pair_unlocks_tp1_tp2_tp3_but_not_tp4(tmp_path):
    paths, _, summary = run(tmp_path, patch=patch_bbox(), event=event_wkt())
    assert summary["ready_pair_feed_rows"] == 1
    assert summary["turning_point_level"] == "TP3_ONE_PATCH_EVENT_PAIR_READY_FOR_OVERLAY"
    gates = {row["gate_name"]: row for row in read_csv(paths["dataset_dir"] / "v2ba_minimal_tp_acquisition_gate.csv")}
    assert gates["TP1_PATCH_BOUNDARY_GEOMETRY_VALID"]["gate_passed"] == "true"
    assert gates["TP2_EVENT_POLYGON_GEOMETRY_VALID"]["gate_passed"] == "true"
    assert gates["TP3_PAIR_FEED_READY"]["gate_passed"] == "true"
    assert gates["TP4_OVERLAY_NOT_YET_CONFIRMED"]["gate_passed"] == "false"


def test_incompatible_package_does_not_create_pair(tmp_path):
    paths, _, summary = run(
        tmp_path,
        patch=patch_bbox(package_id="PKG_OTHER"),
        event=event_wkt(package_id="PKG_34713b8aab96"),
    )
    assert summary["ready_patch_feed_rows"] == 1
    assert summary["ready_event_feed_rows"] == 1
    assert summary["ready_pair_feed_rows"] == 0
    assert read_csv(paths["dataset_dir"] / "v2ba_ready_turning_point_pair_feed.csv") == []
