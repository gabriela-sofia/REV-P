"""v2bh QGIS package, docs, schemas and outputs tests."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "datasets/external_sources/recife_minimal_tp/event_polygon_REC_2022_05_24_30/charter758/georeferencing"


def test_qgis_package_is_review_oriented_and_empty_template_has_no_polygon():
    assert (BASE / "README.md").is_file()
    assert (BASE / "qgis/README_QGIS_GEOREFERENCE.md").is_file()
    assert (BASE / "qgis/charter758_recife_georeferencing_project.qgs").is_file()
    empty = json.loads((BASE / "qgis/charter758_recife_digitized_scars_empty.geojson").read_text(encoding="utf-8"))
    assert empty["features"] == []


def test_docs_schemas_and_public_outputs_exist():
    assert len(list((ROOT / "docs").glob("v2bh_*.md"))) == 4
    assert len(list((ROOT / "datasets/schemas").glob("v2bh_*.schema.json"))) == 15
    assert (ROOT / "outputs_public/execution_reports/v2bh_charter758_recife_product_georeferencing_digitization_summary.json").is_file()
