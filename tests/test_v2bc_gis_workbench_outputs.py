"""v2bc GIS package output tests."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WB = ROOT / "datasets" / "gis_workbench" / "recife_minimal_tp"


def test_gis_workbench_structure_and_qgis_files_exist():
    for name in ("layers", "qgis", "maps", "manual_digitization"):
        assert (WB / name).is_dir()
    assert (WB / "qgis" / "recife_minimal_tp_digitization.qgs").is_file()
    assert (WB / "qgis" / "README_QGIS.md").is_file()
    assert (WB / "maps" / "charter_758_recife_quickview.png").is_file()


def test_manifest_has_no_pipeline_feed_artifact():
    with (ROOT / "datasets" / "v2bc_gis_workbench_manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert all(row["can_feed_pipeline"] == "false" for row in rows)
