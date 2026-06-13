"""v2ba external file inventory tests."""

from tests.v2ba_test_helpers import patch_bbox, read_csv, run


def test_canonical_inventory_ignores_generated_fill_files_and_does_not_promote_context():
    from pathlib import Path
    import scripts.v2ba_minimal_real_geometry_acquisition_workbench as engine
    root = Path(__file__).resolve().parents[1]
    inventory = engine.load_csv(root / "datasets" / "v2ba_external_file_inventory.csv")
    ignored_templates = {"FILL_THIS_PATCH_BOUNDARY.csv", "FILL_THIS_EVENT_POLYGON.csv"}
    assert all(row["file_name"] not in ignored_templates for row in inventory)
    assert engine.load_csv(root / "datasets" / "v2ba_ready_patch_boundary_feed.csv") == []
    assert engine.load_csv(root / "datasets" / "v2ba_ready_event_polygon_feed.csv") == []


def test_inventory_hashes_supplied_external_file(tmp_path):
    paths, code, summary = run(tmp_path, patch=patch_bbox())
    rows = read_csv(paths["dataset_dir"] / "v2ba_external_file_inventory.csv")
    assert code == 0 and summary["external_files_found"] == 1
    assert len(rows[0]["hash_sha256"]) == 64
    assert rows[0]["can_parse"] == "true"
