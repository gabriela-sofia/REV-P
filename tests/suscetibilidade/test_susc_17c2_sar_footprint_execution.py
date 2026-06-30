"""SUSC-17C2 Sentinel-1/SAR Footprint Execution tests (review-only, fail-closed).

Verify the pipeline builds every artifact, follows the footprint/patch-link
schemas, stays review-only, never creates score v7 / ground truth / training,
keeps every prioritized canary in the execution registry, never makes a
candidate footprint eligible for strong evaluation / calibration / 17B before QA
accepted, requires a footprint behind every executed_candidate_created (and none
behind blocked executions), keeps every footprint in the QA queue, routes the
International Charter through the official-geometry review path, and that the
validator fails on every forbidden condition. IDs deterministic, build
byte-identical.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

EXEC_REGISTRY = OUT / "susc_17c2_canary_execution_registry.csv"
QUERY_PLAN = OUT / "susc_17c2_sentinel1_query_plan.csv"
PROCESSING_PLAN = OUT / "susc_17c2_sar_processing_plan.csv"
FOOTPRINT_REGISTRY = OUT / "susc_17c2_candidate_footprint_registry.csv"
FOOTPRINT_GEOJSON = OUT / "susc_17c2_candidate_footprints.geojson"
PATCH_LINKS = OUT / "susc_17c2_candidate_patch_links.csv"
QA_UPDATE = OUT / "susc_17c2_human_qa_update.csv"
SUMMARY = OUT / "susc_17c2_execution_summary.json"
FP_SCHEMA = SCHEMAS / "susc_17c2_candidate_footprint_schema_v1.json"
PL_SCHEMA = SCHEMAS / "susc_17c2_candidate_patch_link_schema_v1.json"
PRIORITY_17C = OUT / "susc_17c_priority_canary_events.csv"


def _load_common():
    path = SCRIPTS / "susc_17c2_sar_footprint_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17c2_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(name: str):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name)],
        cwd=ROOT, text=True, capture_output=True, timeout=600, check=False,
    )


CLEAN_FP = {
    "candidate_footprint_id": "S17C2_FP_x", "candidate_event_id": "S17C_E_x",
    "execution_id": "EXEC_x", "event_date": "2022-05-24", "geometry_type": "Polygon",
    "geometry_source": "susc_13a_official_international_charter_bbox",
    "evidence_class": "technical_remote_sensing_flood_footprint_candidate",
    "source_authority": "international_charter", "annotation_method": "official_charter_flood_extent_bbox",
    "uncertainty_m": "not_available", "spatial_resolution": "coarse_bbox_polygon",
    "temporal_resolution": "date_range_period", "pre_event_window": "2022-04-24/2022-05-17",
    "post_event_window": "2022-05-24/2022-05-31", "qa_status": "needs_review",
    "eligible_for_reference_registry": "false", "eligible_for_strong_evaluation": "false",
    "eligible_for_calibration": "false", "eligible_for_17b": "false",
    "review_only": "true", "trainable": "false", "ground_truth": "false",
    "not_ground_truth_reason": "candidate evidence",
}


def test_build_generates_all_artifacts():
    assert run_script("build_susc_17c2_sar_footprint_execution.py").returncode == 0
    for path in (EXEC_REGISTRY, QUERY_PLAN, PROCESSING_PLAN, FOOTPRINT_REGISTRY,
                 FOOTPRINT_GEOJSON, PATCH_LINKS, QA_UPDATE, SUMMARY):
        assert path.exists(), path
    assert run_script("validate_susc_17c2_sar_footprint_execution.py").returncode == 0


def test_schemas_valid():
    s = _load_common()
    fp_schema = json.loads(FP_SCHEMA.read_text(encoding="utf-8"))
    pl_schema = json.loads(PL_SCHEMA.read_text(encoding="utf-8"))
    for row in read_csv(FOOTPRINT_REGISTRY):
        assert s.schema_violations(row, fp_schema) == []
    for row in read_csv(PATCH_LINKS):
        assert s.schema_violations(row, pl_schema) == []


def test_ids_deterministic():
    for path, key in ((EXEC_REGISTRY, "execution_id"), (FOOTPRINT_REGISTRY, "candidate_footprint_id"),
                      (QA_UPDATE, "qa_item_id")):
        ids = [r[key] for r in read_csv(path)]
        assert len(ids) == len(set(ids))
        assert ids == sorted(ids)


def test_build_twice_byte_identical():
    run_script("build_susc_17c2_sar_footprint_execution.py")
    first = tuple(p.read_bytes() for p in (EXEC_REGISTRY, FOOTPRINT_REGISTRY, FOOTPRINT_GEOJSON, SUMMARY))
    run_script("build_susc_17c2_sar_footprint_execution.py")
    second = tuple(p.read_bytes() for p in (EXEC_REGISTRY, FOOTPRINT_REGISTRY, FOOTPRINT_GEOJSON, SUMMARY))
    assert first == second


def test_all_priority_canaries_in_exec_registry():
    priority = {r["candidate_event_id"] for r in read_csv(PRIORITY_17C)}
    execs = {r["candidate_event_id"] for r in read_csv(EXEC_REGISTRY)}
    assert priority <= execs


def test_no_score_v7():
    assert not (ROOT / "datasets" / "suscetibilidade" / "susc_score_v7_candidate_by_patch_v1.csv").exists()
    assert json.loads(SUMMARY.read_text(encoding="utf-8"))["score_v7_created"] is False


def test_no_score_v6_change():
    assert json.loads(SUMMARY.read_text(encoding="utf-8"))["score_v6_changed"] is False


def test_no_trainable():
    for path in (EXEC_REGISTRY, FOOTPRINT_REGISTRY, PATCH_LINKS, QA_UPDATE):
        for row in read_csv(path):
            assert row["trainable"] == "false"


def test_no_ground_truth():
    for path in (EXEC_REGISTRY, FOOTPRINT_REGISTRY, PATCH_LINKS, QA_UPDATE):
        for row in read_csv(path):
            assert row["ground_truth"] == "false"


def test_footprint_not_17b_eligible_before_qa():
    for row in read_csv(FOOTPRINT_REGISTRY):
        if row["qa_status"] != "accepted":
            assert row["eligible_for_17b"] == "false"


def test_footprint_not_calibration_eligible_before_qa():
    for row in read_csv(FOOTPRINT_REGISTRY):
        if row["qa_status"] != "accepted":
            assert row["eligible_for_calibration"] == "false"
            assert row["eligible_for_strong_evaluation"] == "false"


def test_empty_geojson_allowed_if_summary_blocks():
    geojson = json.loads(FOOTPRINT_GEOJSON.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if not geojson["features"]:
        assert summary["technical_footprint_created_count"] == 0
        assert summary["readiness_status"].startswith("BLOCKED") or summary["candidate_footprint_count"] == 0


def test_nonempty_geojson_requires_registry():
    geojson = json.loads(FOOTPRINT_GEOJSON.read_text(encoding="utf-8"))
    registry_ids = {r["candidate_footprint_id"] for r in read_csv(FOOTPRINT_REGISTRY)}
    for feat in geojson["features"]:
        assert feat["properties"]["candidate_footprint_id"] in registry_ids


def test_registry_does_not_point_to_missing_footprint():
    geojson_ids = {f["properties"]["candidate_footprint_id"] for f in json.loads(FOOTPRINT_GEOJSON.read_text(encoding="utf-8"))["features"]}
    for row in read_csv(FOOTPRINT_REGISTRY):
        assert row["candidate_footprint_id"] in geojson_ids


def test_every_footprint_has_qa_item():
    qa_events = {q["candidate_event_id"] for q in read_csv(QA_UPDATE)}
    for fp in read_csv(FOOTPRINT_REGISTRY):
        assert fp["candidate_event_id"] in qa_events


def test_every_block_has_reason():
    for row in read_csv(EXEC_REGISTRY):
        if row["execution_status"].startswith("blocked"):
            assert row["blocking_reason"] not in ("", "not_available")


def test_international_charter_official_geometry_path():
    s = _load_common()
    charter = next(r for r in read_csv(EXEC_REGISTRY) if r["candidate_event_id"] == s.CHARTER_EVENT_ID)
    assert charter["execution_mode"] in ("local_existing_artifact", "planned_not_executed")
    assert charter["execution_mode"] != "gee_task_spec"  # not forced SAR


def test_no_heavy_raster_in_outputs():
    for path in OUT.glob("susc_17c2_*"):
        assert path.suffix.lower() not in (".tif", ".tiff", ".img", ".jp2", ".nc", ".grd", ".npy", ".npz")


def test_failclosed_ground_truth():
    s = _load_common()
    assert s.footprint_row_violations({**CLEAN_FP, "ground_truth": "true"})
    assert s.footprint_row_violations({**CLEAN_FP, "official_score_changed": "true"})
    assert s.footprint_row_violations(CLEAN_FP) == []


def test_failclosed_score_v7_created():
    s = _load_common()
    assert s.footprint_row_violations({**CLEAN_FP, "score_v7_created": "true"})


def test_failclosed_17b_eligible_without_qa():
    s = _load_common()
    assert s.footprint_row_violations({**CLEAN_FP, "eligible_for_17b": "true"})
    assert s.footprint_row_violations({**CLEAN_FP, "eligible_for_calibration": "true"})
    assert s.footprint_row_violations({**CLEAN_FP, "eligible_for_strong_evaluation": "true"})
    # once accepted, strong eligibility is allowed by the guard
    assert s.footprint_row_violations({**CLEAN_FP, "qa_status": "accepted",
                                       "eligible_for_strong_evaluation": "true"}) == []


def test_blocked_execution_invents_no_geometry():
    exec_rows = read_csv(EXEC_REGISTRY)
    fp_events = {f["candidate_event_id"] for f in read_csv(FOOTPRINT_REGISTRY)}
    for ex in exec_rows:
        if ex["execution_status"].startswith("blocked"):
            assert ex["candidate_event_id"] not in fp_events
