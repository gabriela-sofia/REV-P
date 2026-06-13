"""v2bd direct lineage and spatial metadata tests."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    with (ROOT / "datasets" / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_direct_patch_asset_link_and_preserved_header_metadata_exist():
    lineage = read("v2bd_patch_asset_lineage_registry.csv")[0]
    metadata = read("v2bd_spatial_metadata_inventory.csv")[0]
    assert lineage["has_direct_link"] == "true"
    assert lineage["candidate_asset_id"] == "e07eacbc8a366650"
    assert lineage["asset_file"] == "data/sentinel/patch_recife_00019.tif"
    assert metadata["crs"] == "EPSG:32725"
    assert metadata["has_bbox"] == "true"
    assert metadata["can_support_boundary"] == "true"
