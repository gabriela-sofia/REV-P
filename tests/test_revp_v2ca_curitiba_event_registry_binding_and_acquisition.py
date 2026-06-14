"""Tests for revp_v2ca_curitiba_event_registry_binding_and_acquisition.py.

Covers: registry repair from the real v1uv candidate registry (CUR_2022_01_15
and CUR_2022_01_05) without inventing an event when the registry is empty;
official vs unverified event classification; source classification (point as
point evidence, polygon as geometry source, risk-area as context only — never a
formal footprint); Curitiba patch auditing; boundary recovery from synthetic
raster-header bounds and the no-CRS block; patch-event binding without labels;
``ready_for_overlay`` only when patch boundary and event geometry both exist;
the no-label / no-formal-negative / no-training / no-negative-from-absence
invariants; full output generation; no heavy outputs; no private absolute paths;
safe report language and guardrail PASS.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "multimodal"
sys.path.insert(0, str(SCRIPTS_DIR))

from revp_v2ca_curitiba_event_registry_binding_and_acquisition import (  # noqa: E402
    METHODOLOGICAL_GUARDRAILS,
    PB_BLOCKED_CRS,
    PB_RECOVERED,
    build_artifacts,
    build_v1fs_bounds_index,
    classify_curitiba_source,
    recover_patch_boundary,
    repair_event_registry,
    write_artifacts,
)

FORBIDDEN_CLAIMS = [
    "operational flood detection",
    "validated prediction",
    "flood accuracy",
    "operational model",
    "detecção operacional",
    "predição validada",
    "acurácia de inundação",
    "modelo operacional",
]


# --------------------------------------------------------------------------- #
# Synthetic inputs
# --------------------------------------------------------------------------- #

def _candidates():
    return [
        {"candidate_event_id": "CE_0", "event_id_candidate": "CUR_2022_01_15", "city": "Curitiba", "uf": "PR",
         "start_date": "2022-01-15", "end_date": "2022-01-15", "hazard_scope": "urban_flooding|intense_rain",
         "official_source_status": "OFFICIAL_PUBLIC_SOURCE", "can_create_training_label": "false"},
        {"candidate_event_id": "CE_1", "event_id_candidate": "CUR_2022_01_05", "city": "Curitiba", "uf": "PR",
         "start_date": "2022-01-05", "end_date": "2022-01-05", "hazard_scope": "urban_flooding|intense_rain",
         "official_source_status": "OFFICIAL_PUBLIC_SOURCE", "can_create_training_label": "false"},
    ]


def _feature_rows():
    rows = []
    # Two Curitiba patches with embedding, two without; one Recife patch (ignored).
    rows.append({"canonical_patch_id": "CUR_00038", "region": "Curitiba", "source_asset_id": "a1",
                 "dino_input_id": "DINO_1", "dino_embedding_available": "True", "gis_feature_available": "False",
                 "split_group": "Curitiba__sentinel_tif__patch_curitiba_00038"})
    rows.append({"canonical_patch_id": "CUR_00249", "region": "Curitiba", "source_asset_id": "a2",
                 "dino_input_id": "DINO_2", "dino_embedding_available": "True", "gis_feature_available": "False",
                 "split_group": "Curitiba__sentinel_tif__patch_curitiba_00249"})
    rows.append({"canonical_patch_id": "CUR_00999", "region": "Curitiba", "source_asset_id": "a3",
                 "dino_input_id": "DINO_3", "dino_embedding_available": "False", "gis_feature_available": "False",
                 "split_group": "Curitiba__sentinel_tif__patch_curitiba_00999"})
    rows.append({"canonical_patch_id": "REC_00001", "region": "Recife", "source_asset_id": "r1",
                 "dino_input_id": "DINO_R", "dino_embedding_available": "True", "gis_feature_available": "True",
                 "split_group": "Recife__x"})
    return rows


def _bounds_index():
    # CUR_00038 has valid EPSG:32722 bounds; CUR_00249 has unknown CRS; CUR_00999 absent.
    return {
        "CUR_00038": {"crs": "EPSG:32722", "bounds": "661730.0,7168160.0,662680.0,7169100.0",
                      "asset_path": "data/cur.tif", "_conflict": "false"},
        "CUR_00249": {"crs": "UNKNOWN", "bounds": "1.0,2.0,3.0,4.0", "asset_path": "", "_conflict": "false"},
    }


def _art(tmp_path, candidates=None, feature_rows=None, sources=None, bounds=None):
    return build_artifacts(
        output_dir=tmp_path,
        curitiba_candidates_override=_candidates() if candidates is None else candidates,
        feature_rows_override=_feature_rows() if feature_rows is None else feature_rows,
        bounds_index_override=_bounds_index() if bounds is None else bounds,
        sources_override=[] if sources is None else sources,
    )


# --------------------------------------------------------------------------- #
# 1. Script runs with minimal synthetic inputs
# --------------------------------------------------------------------------- #

def test_runs_with_minimal_inputs(tmp_path):
    art = _art(tmp_path)
    assert art["summary"]["phase"] == "v2ca"
    assert art["guardrails"]["overall"] == "PASS"


# --------------------------------------------------------------------------- #
# 2-4. Reads scaffold, registers both real events
# --------------------------------------------------------------------------- #

def test_registers_both_real_events(tmp_path):
    art = _art(tmp_path)
    ids = {e["event_id"] for e in art["events"]}
    assert ids == {"CUR_2022_01_15", "CUR_2022_01_05"}
    assert art["summary"]["curitiba_events_repaired"] == 2
    assert art["summary"]["registry_repair_status"] == "CURITIBA_FLOOD_EVENT_REGISTRY_REPAIRED"


# --------------------------------------------------------------------------- #
# 5. Does not invent an event when the registry is empty
# --------------------------------------------------------------------------- #

def test_no_event_invented_when_registry_empty(tmp_path):
    art = _art(tmp_path, candidates=[])
    assert art["events"] == []
    assert art["summary"]["registry_repair_status"] == "CURITIBA_EVENT_REGISTRY_STILL_MISSING"
    assert art["guardrails"]["checks"]["no_event_invented"] == "PASS"


# --------------------------------------------------------------------------- #
# 6. Classifies official vs unverified events
# --------------------------------------------------------------------------- #

def test_official_vs_unverified_classification():
    official = repair_event_registry(_candidates(), Path("x"))
    assert all(e["is_official"] == "true" for e in official)
    assert all(e["source_family"] == "CURITIBA_OFFICIAL_PUBLIC_EVENT_CANDIDATE" for e in official)

    unverified = repair_event_registry([
        {"event_id_candidate": "CUR_2099_01_01", "city": "Curitiba", "start_date": "2099-01-01",
         "hazard_scope": "alagamento", "official_source_status": "rumor"},
    ], Path("x"))
    assert unverified[0]["source_family"] == "CURITIBA_EVENT_CANDIDATE_UNVERIFIED"
    assert unverified[0]["is_official"] == "false"


# --------------------------------------------------------------------------- #
# 7-9. Source classification: point / polygon / risk area
# --------------------------------------------------------------------------- #

def _write_geojson(path, gtype):
    if gtype == "point":
        geom = {"type": "Point", "coordinates": [-49.27, -25.43]}
    else:
        geom = {"type": "Polygon", "coordinates": [[[-49.3, -25.5], [-49.2, -25.5], [-49.2, -25.4], [-49.3, -25.4], [-49.3, -25.5]]]}
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": geom}]}), encoding="utf-8")


def test_point_classified_as_point_evidence(tmp_path):
    p = tmp_path / "curitiba_defesa_civil_points.geojson"
    _write_geojson(p, "point")
    info = classify_curitiba_source(p)
    assert info is not None
    assert info["is_point_source"] == "true"
    assert info["source_family"] == "POINT_EVIDENCE_SOURCE"


def test_polygon_classified_as_geometry_source_not_gt(tmp_path):
    p = tmp_path / "curitiba_event_footprint.geojson"
    _write_geojson(p, "polygon")
    info = classify_curitiba_source(p)
    assert info is not None
    assert info["is_polygon_source"] == "true"
    assert info["source_family"] == "POLYGON_GEOMETRY_SOURCE"
    assert "not_formal_gt" in info["recommended_use"]


def test_risk_area_polygon_is_context_only(tmp_path):
    p = tmp_path / "curitiba_areas_de_risco.geojson"
    _write_geojson(p, "polygon")
    info = classify_curitiba_source(p)
    assert info is not None
    assert info["is_risk_area_source"] == "true"
    assert info["source_family"] == "RISK_AREA_GEOMETRY_SOURCE"
    assert "context_only" in info["recommended_use"]


def test_risk_area_geometry_inventory_not_event_footprint(tmp_path):
    p = tmp_path / "curitiba_areas_de_risco.geojson"
    _write_geojson(p, "polygon")
    src = classify_curitiba_source(p)
    art = _art(tmp_path, sources=[src])
    risk_rows = [g for g in art["geometry"] if g["is_risk_area_general"] == "true"]
    assert risk_rows  # detected
    for g in risk_rows:
        assert g["is_event_specific"] == "false"
        assert g["can_support_formal_gt"] == "false"
        assert g["geometry_quality_status"] == "RISK_AREA_GEOMETRY_CONTEXT_ONLY"


# --------------------------------------------------------------------------- #
# 10. Audits Curitiba patches
# --------------------------------------------------------------------------- #

def test_audits_curitiba_patches(tmp_path):
    art = _art(tmp_path)
    pids = {p["canonical_patch_id"] for p in art["patches"]}
    assert pids == {"CUR_00038", "CUR_00249", "CUR_00999"}  # Recife excluded
    assert art["summary"]["patches_in_region"] == 3
    assert art["summary"]["patches_with_dino_embedding"] == 2


# --------------------------------------------------------------------------- #
# 11. Recovers boundary from synthetic bbox/header
# --------------------------------------------------------------------------- #

def test_recovers_boundary_from_synthetic_bounds(tmp_path):
    rec = recover_patch_boundary("CUR_00038", _bounds_index(), tmp_path / "rec")
    assert rec["boundary_recovered"] == "true"
    assert rec["boundary_recovery_status"] == PB_RECOVERED
    sidecar = tmp_path / "rec" / "patch_boundary_CUR_00038_recovered_v2ca.geojson"
    assert sidecar.exists()
    feat = json.loads(sidecar.read_text(encoding="utf-8"))
    assert feat["properties"]["can_be_ground_truth"] is False
    ring = feat["geometry"]["coordinates"][0]
    xs = [c[0] for c in ring]
    ys = [c[1] for c in ring]
    # Reprojected into the Curitiba window.
    assert -49.5 <= min(xs) and max(xs) <= -49.0
    assert -25.7 <= min(ys) and max(ys) <= -25.2


# --------------------------------------------------------------------------- #
# 12. Blocks boundary without CRS
# --------------------------------------------------------------------------- #

def test_blocks_boundary_without_crs(tmp_path):
    rec = recover_patch_boundary("CUR_00249", _bounds_index(), tmp_path / "rec")
    assert rec["boundary_recovered"] == "false"
    assert rec["boundary_recovery_status"] == PB_BLOCKED_CRS
    assert rec["blocked_reason"] == "CRS_UNKNOWN_OR_NO_REPROJECTION_BACKEND"


def test_blocks_boundary_when_no_recorded_bounds(tmp_path):
    rec = recover_patch_boundary("CUR_00999", _bounds_index(), tmp_path / "rec")
    assert rec["boundary_recovered"] == "false"
    assert rec["blocked_reason"] == "NO_RECORDED_HEADER_BOUNDS_FOR_PATCH_ID"


# --------------------------------------------------------------------------- #
# 13-14. Patch-event binding without label; ready_for_overlay only with geometry
# --------------------------------------------------------------------------- #

def test_binding_created_without_label(tmp_path):
    art = _art(tmp_path)
    # 2 events x 3 patches = 6 bindings.
    assert len(art["bindings"]) == 6
    for b in art["bindings"]:
        assert "gt_patch_flood_observed" not in b
        assert b["can_enter_overlay"] in {"true", "false"}


def test_ready_for_overlay_only_with_patch_boundary_and_event_geometry(tmp_path):
    # No event geometry -> no binding ready for overlay, even with recovered boundary.
    art = _art(tmp_path)
    assert art["summary"]["ready_for_overlay_count"] == 0
    assert all(b["can_enter_overlay"] == "false" for b in art["bindings"])

    # Add an event polygon source -> overlay becomes possible for patches with boundary.
    poly = tmp_path / "curitiba_event_footprint.geojson"
    _write_geojson(poly, "polygon")
    src = classify_curitiba_source(poly)
    art2 = _art(tmp_path, sources=[src])
    overlay_ready = [b for b in art2["bindings"] if b["can_enter_overlay"] == "true"]
    assert overlay_ready  # CUR_00038 has a recovered boundary
    for b in overlay_ready:
        assert b["patch_has_boundary"] == "true"
        assert b["event_has_polygon_geometry"] == "true"


def test_ready_for_adjudication_with_embedding_and_context(tmp_path):
    art = _art(tmp_path)
    # 2 patches with embedding x 2 events = 4 adjudication-ready bindings.
    assert art["summary"]["ready_for_adjudication_count"] == 4


# --------------------------------------------------------------------------- #
# 15-18. No positive, no negative, no training, no negative-from-absence
# --------------------------------------------------------------------------- #

def test_no_formal_positive_or_negative_or_training(tmp_path):
    art = _art(tmp_path)
    s = art["summary"]
    assert s["labels_created"] is False
    assert s["formal_negatives_created"] is False
    assert s["allowed_for_training_count"] == 0
    assert s["can_train_supervised_model"] is False
    assert art["gate"]["can_train_supervised_model"] is False


def test_no_negative_from_absence(tmp_path):
    art = _art(tmp_path)
    # No binding or patch row produces a negative label by absence of evidence.
    for b in art["bindings"]:
        assert "negative" not in b["binding_status"].lower()
    assert art["guardrails"]["checks"]["no_negative_from_absence"] == "PASS"


# --------------------------------------------------------------------------- #
# 19-20. Generates all expected outputs; no heavy outputs
# --------------------------------------------------------------------------- #

def test_generates_all_expected_outputs(tmp_path):
    art = _art(tmp_path)
    files = write_artifacts(tmp_path, art)
    expected = {
        "curitiba_event_registry_repaired_v2ca.csv",
        "curitiba_event_source_inventory_v2ca.csv",
        "curitiba_event_geometry_inventory_v2ca.csv",
        "curitiba_patch_readiness_inventory_v2ca.csv",
        "curitiba_patch_boundary_recovery_audit_v2ca.csv",
        "curitiba_patch_event_binding_candidates_v2ca.csv",
        "curitiba_evidence_acquisition_queue_v2ca.csv",
        "curitiba_next_chain_execution_plan_v2ca.csv",
        "curitiba_registry_binding_gate_v2ca.json",
        "curitiba_acquisition_guardrails_v2ca.json",
        "curitiba_acquisition_summary_v2ca.json",
        "curitiba_acquisition_report_v2ca.md",
    }
    assert expected.issubset(set(files))


def test_no_heavy_outputs(tmp_path):
    art = _art(tmp_path)
    write_artifacts(tmp_path, art)
    heavy = {".npz", ".npy", ".pt", ".pth", ".parquet", ".tif", ".tiff", ".shp"}
    for p in tmp_path.rglob("*"):
        if p.is_file():
            assert p.suffix.lower() not in heavy


# --------------------------------------------------------------------------- #
# 21. No private absolute paths
# --------------------------------------------------------------------------- #

def test_no_private_absolute_paths(tmp_path):
    art = _art(tmp_path)
    write_artifacts(tmp_path, art)
    needles = ["C:\\Users", "/Users/", "/home/", "gabriela"]
    for p in tmp_path.rglob("*"):
        if p.is_file() and p.suffix in {".csv", ".json", ".md", ".geojson"}:
            text = p.read_text(encoding="utf-8")
            for n in needles:
                assert n not in text, f"private path {n!r} leaked into {p.name}"


# --------------------------------------------------------------------------- #
# 22. Report has no forbidden claims
# --------------------------------------------------------------------------- #

def test_report_has_no_forbidden_claims(tmp_path):
    art = _art(tmp_path)
    write_artifacts(tmp_path, art)
    report = (tmp_path / "curitiba_acquisition_report_v2ca.md").read_text(encoding="utf-8").lower()
    # The standardized guardrail note negates these claims ("claims no ...");
    # the disclaimer is allowed. The body must make no affirmative claim.
    body = report.split("## guardrail note")[0]
    for claim in FORBIDDEN_CLAIMS:
        assert claim.lower() not in body


# --------------------------------------------------------------------------- #
# 23. Guardrails PASS
# --------------------------------------------------------------------------- #

def test_guardrails_pass(tmp_path):
    art = _art(tmp_path)
    g = art["guardrails"]
    assert g["overall"] == "PASS"
    for key, verdict in g["checks"].items():
        assert verdict in {"PASS", "BLOCKED_EXPECTED"}, f"{key}={verdict}"
    assert METHODOLOGICAL_GUARDRAILS["labels_created"] is False
    assert METHODOLOGICAL_GUARDRAILS["supervised_training"] is False


def test_v1fs_bounds_index_detects_conflict(tmp_path):
    audit = tmp_path / "audit.csv"
    audit.write_text(
        "candidate_id,crs_if_header_available,bounds_if_header_available,asset_path\n"
        "CUR_X,EPSG:32722,1.0,2.0,3.0,4.0,a\n",
        encoding="utf-8",
    )
    # Two distinct bounds for the same candidate -> conflict.
    audit.write_text(
        "candidate_id,crs_if_header_available,bounds_if_header_available,asset_path\n"
        "CUR_Y,EPSG:32722,\"1.0,2.0,3.0,4.0\",a\n"
        "CUR_Y,EPSG:32722,\"5.0,6.0,7.0,8.0\",b\n",
        encoding="utf-8",
    )
    idx = build_v1fs_bounds_index(audit)
    assert idx["CUR_Y"]["_conflict"] == "true"
    rec = recover_patch_boundary("CUR_Y", idx, tmp_path / "rec")
    assert rec["boundary_recovered"] == "false"
    assert rec["blocked_reason"] == "MULTIPLE_CONFLICTING_RECORDED_BOUNDS"
