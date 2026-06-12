"""v2au - patch-event overlay engine tests (geometry math + blocking rules)."""

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
    "v2au_geometry_inventory.csv", "v2au_patch_event_overlay_registry.csv",
    "v2au_event_patch_package_overlay_update.csv", "v2au_overlay_gate_decision_audit.csv",
    "v2au_overlay_review_queue.csv",
]


def _overlap_geoms(make_geom, event_id, patch_id, event_box="5,5,15,15", patch_box="0,0,10,10",
                   crs="EPSG:3857", event_role="event_observed_geometry"):
    return [
        make_geom("patch_boundary", "bbox", patch_box, crs=crs, linked_patch_id=patch_id),
        make_geom(event_role, "bbox", event_box, crs=crs, linked_event_id=event_id),
    ]


def test_all_v2au_csvs_generated(v2au_engine, v2au_dataset, tmp_path):
    ds = v2au_dataset()
    _run(v2au_engine, ds, tmp_path)
    for name in ALL_OUTPUTS:
        assert (ds / name).exists(), f"missing {name}"


def test_missing_geometry_blocks_overlay(v2au_engine, v2au_dataset, tmp_path):
    ds = v2au_dataset()  # no geometry sources at all
    _run(v2au_engine, ds, tmp_path)
    ov = _read(ds / "v2au_patch_event_overlay_registry.csv")
    assert ov
    for r in ov:
        assert r["overlay_status"] == "BLOCKED_MISSING_PATCH_GEOMETRY"
        assert r["has_patch_overlay"] == "false"
        assert r["allowed_use"] == "blocked_missing_geometry"


def test_valid_intersection_computes_ratio(v2au_engine, v2au_dataset, tmp_path, v2au_make_package, v2au_make_geom):
    pkgs = [v2au_make_package("PKG_ov1", "E1", "P1")]
    geoms = _overlap_geoms(v2au_make_geom, "E1", "P1")
    ds = v2au_dataset(packages=pkgs, geometry_sources=geoms)
    _run(v2au_engine, ds, tmp_path)
    ov = _read(ds / "v2au_patch_event_overlay_registry.csv")[0]
    # patch 10x10=100, event 10x10 offset, intersection [5,5,10,10]=25 -> ratio_patch 0.25.
    assert ov["overlay_status"] == "OVERLAY_CONFIRMED"
    assert abs(float(ov["intersection_area_m2"]) - 25.0) < 1e-6
    assert abs(float(ov["intersection_ratio_patch"]) - 0.25) < 1e-3
    assert ov["has_patch_overlay"] == "true"
    assert ov["allowed_use"] == "c4_candidate_requires_human_review"


def test_no_intersection_when_disjoint(v2au_engine, v2au_dataset, tmp_path, v2au_make_package, v2au_make_geom):
    pkgs = [v2au_make_package("PKG_dis", "E1", "P1")]
    geoms = _overlap_geoms(v2au_make_geom, "E1", "P1", event_box="50,50,60,60")
    ds = v2au_dataset(packages=pkgs, geometry_sources=geoms)
    _run(v2au_engine, ds, tmp_path)
    ov = _read(ds / "v2au_patch_event_overlay_registry.csv")[0]
    assert ov["overlay_status"] == "NO_INTERSECTION"
    assert ov["has_patch_overlay"] == "false"


def test_unknown_crs_blocks(v2au_engine, v2au_dataset, tmp_path, v2au_make_package, v2au_make_geom):
    pkgs = [v2au_make_package("PKG_crs", "E1", "P1")]
    geoms = _overlap_geoms(v2au_make_geom, "E1", "P1", crs="EPSG:9999")
    ds = v2au_dataset(packages=pkgs, geometry_sources=geoms)
    _run(v2au_engine, ds, tmp_path)
    ov = _read(ds / "v2au_patch_event_overlay_registry.csv")[0]
    assert ov["overlay_status"] == "BLOCKED_UNKNOWN_CRS"
    assert ov["has_patch_overlay"] == "false"


def test_invalid_geometry_blocks(v2au_engine, v2au_dataset, tmp_path, v2au_make_package, v2au_make_geom):
    pkgs = [v2au_make_package("PKG_inv", "E1", "P1")]
    geoms = [
        v2au_make_geom("patch_boundary", "wkt", "POLYGON((0 0))", linked_patch_id="P1"),
        v2au_make_geom("event_observed_geometry", "bbox", "5,5,15,15", linked_event_id="E1"),
    ]
    ds = v2au_dataset(packages=pkgs, geometry_sources=geoms)
    _run(v2au_engine, ds, tmp_path)
    ov = _read(ds / "v2au_patch_event_overlay_registry.csv")[0]
    assert ov["overlay_status"] == "BLOCKED_INVALID_GEOMETRY"


def test_point_without_buffer_not_overlay(v2au_engine, v2au_dataset, tmp_path, v2au_make_package, v2au_make_geom):
    pkgs = [v2au_make_package("PKG_pt", "E1", "P1")]
    geoms = [
        v2au_make_geom("patch_boundary", "bbox", "0,0,10,10", linked_patch_id="P1"),
        v2au_make_geom("point_anchor", "wkt", "POINT(5 5)", linked_event_id="E1"),
    ]
    ds = v2au_dataset(packages=pkgs, geometry_sources=geoms)
    _run(v2au_engine, ds, tmp_path)
    ov = _read(ds / "v2au_patch_event_overlay_registry.csv")[0]
    assert ov["overlay_status"] == "BLOCKED_POINT_ONLY_NO_BUFFER"
    assert ov["has_patch_overlay"] == "false"


def test_context_geometry_does_not_promote_c4(v2au_engine, v2au_dataset, tmp_path, v2au_make_package, v2au_make_geom):
    pkgs = [v2au_make_package("PKG_ctx", "E1", "P1")]
    geoms = [
        v2au_make_geom("patch_boundary", "bbox", "0,0,10,10", linked_patch_id="P1"),
        v2au_make_geom("event_context_geometry", "bbox", "5,5,15,15", linked_event_id="E1"),
    ]
    ds = v2au_dataset(packages=pkgs, geometry_sources=geoms)
    _run(v2au_engine, ds, tmp_path)
    ov = _read(ds / "v2au_patch_event_overlay_registry.csv")[0]
    assert ov["overlay_status"] == "BLOCKED_CONTEXT_GEOMETRY_ONLY"
    assert ov["allowed_use"] == "blocked_context_only"


def test_confirmed_overlay_caps_at_c4_candidate(v2au_engine, v2au_dataset, tmp_path, v2au_make_package, v2au_make_geom):
    pkgs = [v2au_make_package("PKG_cap", "E1", "P1")]
    geoms = _overlap_geoms(v2au_make_geom, "E1", "P1")
    ds = v2au_dataset(packages=pkgs, geometry_sources=geoms)
    _run(v2au_engine, ds, tmp_path)
    upd = _read(ds / "v2au_event_patch_package_overlay_update.csv")[0]
    assert upd["new_promotion_decision"] == "C4_CANDIDATE_REQUIRES_HUMAN_REVIEW"
    assert upd["new_promotion_candidate_level"] == "C4_CANDIDATE"
    assert upd["has_patch_overlay_after"] == "true"
    assert upd["requires_human_review"] == "true"
    assert upd["can_create_operational_label"] == "false"


def test_v2at_registry_not_overwritten(v2au_engine, v2au_dataset, tmp_path, v2au_make_package, v2au_make_geom):
    pkgs = [v2au_make_package("PKG_ov1", "E1", "P1")]
    geoms = _overlap_geoms(v2au_make_geom, "E1", "P1")
    ds = v2au_dataset(packages=pkgs, geometry_sources=geoms)
    before = (ds / "v2at_event_patch_package_registry.csv").read_bytes()
    _run(v2au_engine, ds, tmp_path)
    after = (ds / "v2at_event_patch_package_registry.csv").read_bytes()
    assert before == after, "v2at registry must never be overwritten"


def test_determinism_and_stable_ids(v2au_engine, v2au_dataset, tmp_path, v2au_make_package, v2au_make_geom):
    pkgs = [v2au_make_package("PKG_det", "E1", "P1")]
    geoms = _overlap_geoms(v2au_make_geom, "E1", "P1")
    ds = v2au_dataset(packages=pkgs, geometry_sources=geoms)
    _run(v2au_engine, ds, tmp_path, sub="r1")
    first = {n: (ds / n).read_bytes() for n in ALL_OUTPUTS}
    _run(v2au_engine, ds, tmp_path, sub="r2")
    for n in ALL_OUTPUTS:
        assert (ds / n).read_bytes() == first[n], f"{n} not deterministic"
    ov = _read(ds / "v2au_patch_event_overlay_registry.csv")[0]
    assert ov["overlay_id"] == v2au_engine.stable_id("OVL_", ov["package_id"])


def test_rows_sorted_by_package_event_patch(v2au_engine, v2au_dataset, tmp_path, v2au_make_package):
    pkgs = [v2au_make_package("PKG_b", "E2", "P2"), v2au_make_package("PKG_a", "E1", "P1")]
    ds = v2au_dataset(packages=pkgs)
    _run(v2au_engine, ds, tmp_path)
    ov = _read(ds / "v2au_patch_event_overlay_registry.csv")
    keys = [(r["package_id"], r["event_id"], r["patch_id"]) for r in ov]
    assert keys == sorted(keys)
