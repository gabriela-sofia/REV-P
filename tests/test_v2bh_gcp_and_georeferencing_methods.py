"""v2bh GCP and georeferencing method tests."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    with (ROOT / "datasets" / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_explicit_grid_gcps_allow_affine_but_visual_only_does_not():
    gcps = read("v2bh_georeferencing_gcp_registry.csv")
    assert len(gcps) == 20
    assert all(r["gcp_valid"] == "true" and r["crs"] == "EPSG:32725" for r in gcps)
    methods = {r["method_name"]: r for r in read("v2bh_georeferencing_method_audit.csv")}
    assert methods["gcp_affine_transform"]["can_georeference"] == "true"
    assert methods["visual_only_ungeoreferenced"]["can_georeference"] == "false"


def test_gcp_template_does_not_invent_controls():
    assert read("v2bh_georeferencing_gcp_template.csv") == []
