"""v2au - geometry inventory tests."""

from __future__ import annotations

import csv
import json
import os


def _run(engine, ds, tmp_path):
    code, summary = engine.run(dataset_dir=str(ds), output_dir=str(tmp_path / "out"),
                               config_dir=str(tmp_path / "cfg"))
    assert code == 0, f"engine returned {code}"
    return summary


def _read(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


REQUIRED_COLUMNS = {
    "geometry_id", "geometry_role", "linked_event_id", "linked_patch_id", "source_id",
    "source_name", "geometry_type", "geometry_format", "geometry_path", "crs",
    "crs_status", "area_m2", "bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy",
    "is_valid_geometry", "geometry_hash", "blocking_reason", "notes",
}


def test_inventory_required_columns(v2au_engine, v2au_dataset, tmp_path):
    ds = v2au_dataset()
    _run(v2au_engine, ds, tmp_path)
    rows = _read(ds / "v2au_geometry_inventory.csv")
    # No private fields leak into the CSV.
    assert all(not c.startswith("_") for c in rows[0].keys()) if rows else True
    assert REQUIRED_COLUMNS == set(rows[0].keys()) if rows else True


def test_cprm_points_inventoried_as_point_anchors(v2au_engine, v2au_dataset, tmp_path, v2au_make_package):
    ground = [{"event_id": "EVENT_PET2022_CPRM_ANEXOII", "region": "PET",
               "event_or_survey_date": "19/02/2022", "coordinate_status": "EXPLICIT_COORDINATE",
               "latitude": "-22.48", "longitude": "-43.21", "phenomenon_group": "MOVEMENT_OF_MASS"}]
    pkgs = [v2au_make_package("PKG_pet1", "PET_2022_02_15", "PET_00016", region="Petropolis")]
    ds = v2au_dataset(packages=pkgs, ground_events=ground)
    _run(v2au_engine, ds, tmp_path)
    rows = _read(ds / "v2au_geometry_inventory.csv")
    points = [r for r in rows if r["geometry_role"] == "point_anchor"]
    assert points, "expected CPRM point anchors"
    p = points[0]
    assert p["crs"] == "EPSG:4326" and p["crs_status"] == "KNOWN"
    assert p["geometry_type"] == "point"
    assert p["blocking_reason"] == "POINT_GEOMETRY_NOT_OVERLAY"
    assert p["linked_event_id"].startswith("PET")


def test_unknown_crs_flagged_in_inventory(v2au_engine, v2au_dataset, tmp_path, v2au_make_geom):
    geoms = [v2au_make_geom("patch_boundary", "bbox", "0,0,10,10", crs="EPSG:9999",
                            linked_patch_id="REC_00205")]
    ds = v2au_dataset(geometry_sources=geoms)
    _run(v2au_engine, ds, tmp_path)
    rows = _read(ds / "v2au_geometry_inventory.csv")
    bad = [r for r in rows if r["geometry_role"] == "patch_boundary"]
    assert bad and bad[0]["crs_status"] == "UNKNOWN"


def test_invalid_geometry_flagged(v2au_engine, v2au_dataset, tmp_path, v2au_make_geom):
    geoms = [v2au_make_geom("patch_boundary", "wkt", "POLYGON((0 0))", linked_patch_id="REC_00205")]
    ds = v2au_dataset(geometry_sources=geoms)
    _run(v2au_engine, ds, tmp_path)
    rows = _read(ds / "v2au_geometry_inventory.csv")
    bad = [r for r in rows if r["geometry_role"] == "patch_boundary"]
    assert bad and bad[0]["is_valid_geometry"] == "false"


def test_polygon_area_computed(v2au_engine, v2au_dataset, tmp_path, v2au_make_geom):
    # 10x10 box in EPSG:3857 -> 100 m^2.
    geoms = [v2au_make_geom("patch_boundary", "bbox", "0,0,10,10", crs="EPSG:3857",
                            linked_patch_id="REC_00205")]
    ds = v2au_dataset(geometry_sources=geoms)
    _run(v2au_engine, ds, tmp_path)
    rows = _read(ds / "v2au_geometry_inventory.csv")
    box = [r for r in rows if r["geometry_role"] == "patch_boundary"][0]
    assert abs(float(box["area_m2"]) - 100.0) < 1e-6


def test_inventory_matches_schema(v2au_engine, v2au_dataset, tmp_path):
    ds = v2au_dataset()
    _run(v2au_engine, ds, tmp_path)
    rows = _read(ds / "v2au_geometry_inventory.csv")
    schema = json.load(open(os.path.join(v2au_engine.PROJECT_ROOT, "datasets", "schemas",
                                         "v2au_geometry_inventory.schema.json"), encoding="utf-8"))
    if not rows:
        return
    assert set(schema["required"]).issubset(set(rows[0].keys()))
    enums = {c: p["enum"] for c, p in schema["properties"].items() if "enum" in p}
    for row in rows:
        for col, allowed in enums.items():
            assert row[col] in allowed, f"{col}={row[col]} not in enum"


def test_deterministic_inventory(v2au_engine, v2au_dataset, tmp_path, v2au_make_geom):
    geoms = [v2au_make_geom("patch_boundary", "bbox", "0,0,10,10", linked_patch_id="REC_00205")]
    ds = v2au_dataset(geometry_sources=geoms)
    _run(v2au_engine, ds, tmp_path)
    first = (ds / "v2au_geometry_inventory.csv").read_bytes()
    _run(v2au_engine, ds, tmp_path)
    assert (ds / "v2au_geometry_inventory.csv").read_bytes() == first
