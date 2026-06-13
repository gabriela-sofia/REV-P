"""v2ba external source manifest and search plan tests."""

from pathlib import Path

import scripts.v2ba_minimal_real_geometry_acquisition_workbench as engine


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_and_search_plan_cover_minimal_pair_and_strong_sources():
    manifest = engine.load_csv(ROOT / "datasets" / "v2ba_external_source_acquisition_manifest.csv")
    plan = engine.load_csv(ROOT / "datasets" / "v2ba_external_search_and_download_plan.csv")
    assert {row["target_id"] for row in manifest} == {"REC_00019", "REC_2022_05_24_30"}
    assert any("Charter" in row["source_name"] for row in manifest)
    assert any("Sentinel" in row["source_name"] for row in plan)
    assert all("license" not in key.lower() for row in manifest for key in row)


def test_external_structure_exists():
    root = ROOT / "datasets" / "external_sources" / "recife_minimal_tp"
    for name in ("patch_boundary_REC_00019", "event_polygon_REC_2022_05_24_30",
                 "source_documents", "raw", "derived"):
        assert (root / name).is_dir()
    assert (root / "README.md").is_file()
