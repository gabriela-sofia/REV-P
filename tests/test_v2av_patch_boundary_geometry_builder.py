"""v2av - patch boundary geometry builder tests (geometry math + blocking rules)."""

from __future__ import annotations

import csv


def _run(engine, ds, tmp_path, sub="out"):
    code, summary = engine.run(dataset_dir=str(ds), output_dir=str(tmp_path / sub),
                               config_dir=str(tmp_path / "cfg"))
    assert code == 0, f"engine returned {code}"
    return summary


def _read(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


ALL_OUTPUTS = [
    "v2av_patch_boundary_source_manifest.csv", "v2av_patch_boundary_geometry_registry.csv",
    "v2av_patch_boundary_build_audit.csv", "v2av_patch_boundary_recovery_queue.csv",
]


def test_all_v2av_csvs_generated(v2av_engine, v2av_dataset, tmp_path):
    ds = v2av_dataset()
    _run(v2av_engine, ds, tmp_path)
    for name in ALL_OUTPUTS:
        assert (ds / name).exists(), f"missing {name}"


def test_no_metadata_blocks_all(v2av_engine, v2av_dataset, tmp_path):
    ds = v2av_dataset()
    summary = _run(v2av_engine, ds, tmp_path)
    reg = _read(ds / "v2av_patch_boundary_geometry_registry.csv")
    assert all(r["is_valid_geometry"] == "false" for r in reg)
    assert summary["patch_boundaries_built"] == 0
    assert summary["geojson_files_written"] == 0


def test_bbox_builds_polygon(v2av_engine, v2av_dataset, tmp_path, v2av_make_geom_source):
    geoms = [v2av_make_geom_source("REC_00205", "bbox", "0,0,10,10", crs="EPSG:3857")]
    ds = v2av_dataset(geometry_sources=geoms)
    _run(v2av_engine, ds, tmp_path)
    r = _read(ds / "v2av_patch_boundary_geometry_registry.csv")[0]
    assert r["is_valid_geometry"] == "true"
    assert r["geometry_type"] == "polygon"
    assert r["build_method"] == "from_bbox"
    assert abs(float(r["area_m2"]) - 100.0) < 1e-6
    assert r["geometry_wkt"].startswith("POLYGON")
    # GeoJSON file written and referenced.
    assert r["geometry_geojson_path"].endswith("patch_boundary_REC_00205.geojson")
    assert (ds / "geometries" / "patch_boundaries" / "patch_boundary_REC_00205.geojson").exists()


def test_wkt_builds_boundary(v2av_engine, v2av_dataset, tmp_path, v2av_make_geom_source):
    geoms = [v2av_make_geom_source("REC_00205", "wkt",
                                   "POLYGON((0 0, 20 0, 20 20, 0 20, 0 0))", crs="EPSG:3857")]
    ds = v2av_dataset(geometry_sources=geoms)
    _run(v2av_engine, ds, tmp_path)
    r = _read(ds / "v2av_patch_boundary_geometry_registry.csv")[0]
    assert r["is_valid_geometry"] == "true"
    assert r["build_method"] == "from_wkt"
    assert abs(float(r["area_m2"]) - 400.0) < 1e-6


def test_geojson_inline_builds_boundary(v2av_engine, v2av_dataset, tmp_path, v2av_make_geom_source):
    gj = '{"type":"Polygon","coordinates":[[[0,0],[10,0],[10,10],[0,10],[0,0]]]}'
    geoms = [v2av_make_geom_source("REC_00205", "geojson_inline", gj, crs="EPSG:3857")]
    ds = v2av_dataset(geometry_sources=geoms)
    _run(v2av_engine, ds, tmp_path)
    r = _read(ds / "v2av_patch_boundary_geometry_registry.csv")[0]
    assert r["is_valid_geometry"] == "true"
    assert r["build_method"] == "from_geojson"


def test_unknown_crs_blocks_boundary(v2av_engine, v2av_dataset, tmp_path, v2av_make_geom_source):
    geoms = [v2av_make_geom_source("REC_00205", "bbox", "0,0,10,10", crs="EPSG:9999")]
    ds = v2av_dataset(geometry_sources=geoms)
    _run(v2av_engine, ds, tmp_path)
    r = _read(ds / "v2av_patch_boundary_geometry_registry.csv")[0]
    assert r["is_valid_geometry"] == "false"
    assert r["blocking_reason"] == "UNACCEPTED_OR_UNKNOWN_CRS"


def test_missing_crs_blocks_boundary(v2av_engine, v2av_dataset, tmp_path, v2av_make_geom_source):
    geoms = [v2av_make_geom_source("REC_00205", "bbox", "0,0,10,10", crs="")]
    ds = v2av_dataset(geometry_sources=geoms)
    _run(v2av_engine, ds, tmp_path)
    r = _read(ds / "v2av_patch_boundary_geometry_registry.csv")[0]
    assert r["is_valid_geometry"] == "false"
    assert r["blocking_reason"] == "MISSING_CRS"


def test_center_point_without_optin_not_built(v2av_engine, v2av_dataset, tmp_path, v2av_make_geom_source):
    # allow_center_point_buffer defaults to false.
    geoms = [v2av_make_geom_source("REC_00205", "center_point", "", crs="EPSG:3857",
                                   center_lat="0", center_lon="0", size_meters="100")]
    ds = v2av_dataset(geometry_sources=geoms)
    _run(v2av_engine, ds, tmp_path)
    r = _read(ds / "v2av_patch_boundary_geometry_registry.csv")[0]
    assert r["is_valid_geometry"] == "false"
    assert r["blocking_reason"] == "CENTER_POINT_BUFFER_NOT_ALLOWED"


def test_default_size_not_used_when_disallowed(v2av_engine, v2av_dataset, tmp_path, v2av_make_geom_source):
    # Even with center buffer opt-in (via config override), no size + no default => blocked.
    geoms = [v2av_make_geom_source("REC_00205", "center_point", "", crs="EPSG:3857",
                                   center_lat="0", center_lon="0", size_meters="")]
    ds = v2av_dataset(geometry_sources=geoms)
    import json
    cfg = tmp_path / "cfg"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "v2av_patch_boundary_geometry_builder_config.json").write_text(json.dumps({
        "accepted_crs": ["EPSG:4326", "EPSG:3857"], "target_crs": "EPSG:3857",
        "allow_center_point_buffer": True, "allow_default_patch_size": False,
    }), encoding="utf-8")
    code, _ = v2av_engine.run(dataset_dir=str(ds), output_dir=str(tmp_path / "o"), config_dir=str(cfg))
    assert code == 0
    r = _read(ds / "v2av_patch_boundary_geometry_registry.csv")[0]
    assert r["is_valid_geometry"] == "false"
    assert r["blocking_reason"] == "DEFAULT_PATCH_SIZE_NOT_ALLOWED"


def test_determinism_and_stable_ids(v2av_engine, v2av_dataset, tmp_path, v2av_make_geom_source):
    geoms = [v2av_make_geom_source("REC_00205", "bbox", "0,0,10,10", crs="EPSG:3857")]
    ds = v2av_dataset(geometry_sources=geoms)
    _run(v2av_engine, ds, tmp_path, sub="r1")
    first = {n: (ds / n).read_bytes() for n in ALL_OUTPUTS}
    _run(v2av_engine, ds, tmp_path, sub="r2")
    for n in ALL_OUTPUTS:
        assert (ds / n).read_bytes() == first[n], f"{n} not deterministic"


def test_v2at_v2au_not_overwritten(v2av_engine, v2av_dataset, tmp_path):
    ds = v2av_dataset()
    before = (ds / "v2at_event_patch_package_registry.csv").read_bytes()
    _run(v2av_engine, ds, tmp_path)
    assert (ds / "v2at_event_patch_package_registry.csv").read_bytes() == before
