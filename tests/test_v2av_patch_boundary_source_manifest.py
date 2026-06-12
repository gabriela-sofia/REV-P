"""v2av - patch boundary source manifest tests."""

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
    "patch_id", "region", "city", "source_file", "source_field", "source_type",
    "has_bbox", "has_wkt", "has_geojson", "has_raster_transform", "has_center_point",
    "has_resolution", "has_crs", "crs", "can_build_boundary", "boundary_build_method",
    "blocking_reason", "notes",
}


def test_manifest_required_columns(v2av_engine, v2av_dataset, tmp_path):
    ds = v2av_dataset()
    _run(v2av_engine, ds, tmp_path)
    rows = _read(ds / "v2av_patch_boundary_source_manifest.csv")
    assert rows
    assert REQUIRED_COLUMNS == set(rows[0].keys())


def test_patch_without_metadata_is_blocked(v2av_engine, v2av_dataset, tmp_path):
    ds = v2av_dataset()  # no geometry sources
    _run(v2av_engine, ds, tmp_path)
    rows = _read(ds / "v2av_patch_boundary_source_manifest.csv")
    for r in rows:
        assert r["can_build_boundary"] == "false"
        assert r["source_type"] == "none"
        assert r["blocking_reason"] == "NO_SPATIAL_METADATA_FOUND"


def test_all_unique_patches_discovered(v2av_engine, v2av_dataset, tmp_path, v2av_make_patch):
    patches = [v2av_make_patch("REC_00205", "REC"), v2av_make_patch("PET_00016", "PET", "Petropolis"),
               v2av_make_patch("CUR_00038", "CUR", "Curitiba")]
    ds = v2av_dataset(patches=patches)
    _run(v2av_engine, ds, tmp_path)
    rows = _read(ds / "v2av_patch_boundary_source_manifest.csv")
    assert {r["patch_id"] for r in rows} == {"REC_00205", "PET_00016", "CUR_00038"}


def test_bbox_source_marks_can_build(v2av_engine, v2av_dataset, tmp_path, v2av_make_geom_source):
    geoms = [v2av_make_geom_source("REC_00205", "bbox", "0,0,10,10", crs="EPSG:3857")]
    ds = v2av_dataset(geometry_sources=geoms)
    _run(v2av_engine, ds, tmp_path)
    rows = _read(ds / "v2av_patch_boundary_source_manifest.csv")
    r = next(x for x in rows if x["patch_id"] == "REC_00205")
    assert r["has_bbox"] == "true"
    assert r["has_crs"] == "true"
    assert r["can_build_boundary"] == "true"
    assert r["boundary_build_method"] == "from_bbox"


def test_manifest_matches_schema(v2av_engine, v2av_dataset, tmp_path):
    ds = v2av_dataset()
    _run(v2av_engine, ds, tmp_path)
    rows = _read(ds / "v2av_patch_boundary_source_manifest.csv")
    schema = json.load(open(os.path.join(v2av_engine.PROJECT_ROOT, "datasets", "schemas",
                                         "v2av_patch_boundary_source_manifest.schema.json"), encoding="utf-8"))
    assert set(schema["required"]).issubset(set(rows[0].keys()))
    enums = {c: p["enum"] for c, p in schema["properties"].items() if "enum" in p}
    for row in rows:
        for col, allowed in enums.items():
            assert row[col] in allowed, f"{col}={row[col]} not in enum"
