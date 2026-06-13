"""v2ba assisted manual-fill tests."""

from pathlib import Path

import scripts.v2ba_minimal_real_geometry_acquisition_workbench as engine


ROOT = Path(__file__).resolve().parents[1]


def test_fill_files_exist_and_contain_no_geometry():
    root = ROOT / "datasets" / "external_sources" / "recife_minimal_tp"
    paths = [
        root / "patch_boundary_REC_00019" / "FILL_THIS_PATCH_BOUNDARY.csv",
        root / "event_polygon_REC_2022_05_24_30" / "FILL_THIS_EVENT_POLYGON.csv",
    ]
    for path in paths:
        row = engine.load_csv(path)[0]
        assert row["source_type"] == "missing"
        assert row["geometry_value"] == ""
        assert row["geometry_path"] == ""
        assert row["crs"] == "UNKNOWN"
        assert row["source_public"] == "true"


def test_adapters_are_ready_but_fail_closed():
    dataset = ROOT / "datasets"
    patch = engine.load_csv(dataset / "v2ba_minimal_candidate_patch_intake_adapter.csv")[0]
    event = engine.load_csv(dataset / "v2ba_minimal_candidate_event_intake_adapter.csv")[0]
    assert patch["patch_id"] == "REC_00019"
    assert event["event_id"] == "REC_2022_05_24_30"
    assert patch["blocking_reason"] == event["blocking_reason"] == "REAL_GEOMETRY_REQUIRED"
