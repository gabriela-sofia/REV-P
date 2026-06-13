"""v2bc contextual GIS layer tests."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAYERS = ROOT / "datasets" / "gis_workbench" / "recife_minimal_tp" / "layers"


def test_400_risk_locations_are_context_points():
    obj = json.loads((LAYERS / "recife_risk_areas_context.geojson").read_text(encoding="utf-8"))
    assert len(obj["features"]) == 400
    assert all(f["geometry"]["type"] == "Point" for f in obj["features"])
    assert all(f["properties"]["geometry_role"] == "context_risk_location_point" for f in obj["features"])
    assert all(f["properties"]["can_feed_pipeline"] is False for f in obj["features"])


def test_placeholders_have_null_geometry_and_aoi_is_support_only():
    placeholders = json.loads((LAYERS / "recife_missing_geometries_placeholders.geojson").read_text(encoding="utf-8"))
    assert len(placeholders["features"]) == 2
    assert all(f["geometry"] is None for f in placeholders["features"])
    aoi = json.loads((LAYERS / "recife_digitization_aoi_context.geojson").read_text(encoding="utf-8"))
    assert aoi["features"][0]["properties"]["geometry_role"] == "digitization_support_only"
    assert aoi["features"][0]["properties"]["can_feed_overlay"] is False
