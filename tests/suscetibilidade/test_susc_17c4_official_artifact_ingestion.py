"""SUSC-17C4 Official Artifact Ingestion tests (review-only, fail-closed).

Verify the pipeline builds every artifact, follows schemas, stays review-only,
never creates score v7 / ground truth / training, never treats PE3D as event,
never turns a PDF/ZIP without vector or a missing artifact into a reference,
never creates a patch-link for geometry outside the grid, keeps 17B ineligible
without QA, surfaces the P0 DRM-RJ / COMPDEC formal requests, and keeps the
GeoJSON / registry / patch-links consistent. IDs deterministic, build
byte-identical.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

INVENTORY = OUT / "susc_17c4_artifact_inventory.csv"
INGESTION = OUT / "susc_17c4_ingestion_attempts.csv"
CANDIDATES = OUT / "susc_17c4_extracted_reference_candidates.csv"
GEOJSON = OUT / "susc_17c4_candidate_geometries.geojson"
PATCH_LINKS = OUT / "susc_17c4_candidate_patch_links.csv"
FORMAL = OUT / "susc_17c4_formal_request_queue.csv"
SUMMARY = OUT / "susc_17c4_summary.json"
INV_SCHEMA = SCHEMAS / "susc_17c4_artifact_inventory_schema_v1.json"
CAND_SCHEMA = SCHEMAS / "susc_17c4_reference_candidate_schema_v1.json"


def _load_common():
    path = SCRIPTS / "susc_17c4_official_artifact_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17c4_common", path)
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


def test_build_generates_all_artifacts():
    assert run_script("build_susc_17c4_official_artifact_ingestion.py").returncode == 0
    for path in (INVENTORY, INGESTION, CANDIDATES, GEOJSON, PATCH_LINKS, FORMAL, SUMMARY):
        assert path.exists(), path
    assert run_script("validate_susc_17c4_official_artifact_ingestion.py").returncode == 0


def test_schemas_valid():
    s = _load_common()
    inv_schema = json.loads(INV_SCHEMA.read_text(encoding="utf-8"))
    cand_schema = json.loads(CAND_SCHEMA.read_text(encoding="utf-8"))
    for row in read_csv(INVENTORY):
        assert s.schema_violations(row, inv_schema) == []
    for row in read_csv(CANDIDATES):
        assert s.schema_violations(row, cand_schema) == []


def test_ids_deterministic():
    for path, key in ((INVENTORY, "artifact_id"), (INGESTION, "ingestion_attempt_id"),
                      (CANDIDATES, "reference_candidate_id"), (FORMAL, "request_id")):
        ids = [r[key] for r in read_csv(path)]
        assert len(ids) == len(set(ids))
        assert ids == sorted(ids)


def test_build_twice_byte_identical():
    run_script("build_susc_17c4_official_artifact_ingestion.py")
    first = tuple(p.read_bytes() for p in (INVENTORY, INGESTION, CANDIDATES, GEOJSON, SUMMARY))
    run_script("build_susc_17c4_official_artifact_ingestion.py")
    second = tuple(p.read_bytes() for p in (INVENTORY, INGESTION, CANDIDATES, GEOJSON, SUMMARY))
    assert first == second


def test_no_score_v7():
    assert not (ROOT / "datasets" / "suscetibilidade" / "susc_score_v7_candidate_by_patch_v1.csv").exists()
    assert json.loads(SUMMARY.read_text(encoding="utf-8"))["score_v7_created"] is False


def test_no_score_v6_change():
    assert json.loads(SUMMARY.read_text(encoding="utf-8"))["score_v6_changed"] is False


def test_no_trainable():
    for path in (INVENTORY, INGESTION, CANDIDATES, PATCH_LINKS, FORMAL):
        for row in read_csv(path):
            assert row["trainable"] == "false"


def test_no_ground_truth():
    for path in (INVENTORY, INGESTION, CANDIDATES, PATCH_LINKS, FORMAL):
        for row in read_csv(path):
            assert row["ground_truth"] == "false"


def test_pe3d_not_event():
    for row in read_csv(INVENTORY):
        if "pe3d" in row["artifact_name"].lower() or "pe3d" in row["source_name"].lower():
            assert row["artifact_local_status"] == "context_layer_only"
            assert row["expected_revp_use"] == "contextual_physical_layer"
    # and PE3D never produces a reference candidate
    cand_events = {c["event_id"] for c in read_csv(CANDIDATES)}
    assert "REC_CONTEXT_PE3D" not in cand_events


def test_pdf_without_vector_not_strong():
    inv = {r["artifact_id"]: r for r in read_csv(INVENTORY)}
    cand_artifacts = {c["artifact_id"] for c in read_csv(CANDIDATES)}
    for row in read_csv(INGESTION):
        if inv[row["artifact_id"]]["artifact_local_status"] == "pdf_only":
            assert row["attempt_status"] == "blocked_pdf_only_no_vector"
            assert row["artifact_id"] not in cand_artifacts


def test_zip_without_vector_not_strong():
    inv = {r["artifact_id"]: r for r in read_csv(INVENTORY)}
    cand_artifacts = {c["artifact_id"] for c in read_csv(CANDIDATES)}
    for row in read_csv(INGESTION):
        if inv[row["artifact_id"]]["artifact_local_status"] == "zip_only":
            assert row["attempt_status"] in ("blocked_unknown_format", "blocked_pdf_only_no_vector")
            assert row["artifact_id"] not in cand_artifacts


def test_missing_artifact_creates_no_reference():
    inv = {r["artifact_id"]: r for r in read_csv(INVENTORY)}
    cand_artifacts = {c["artifact_id"] for c in read_csv(CANDIDATES)}
    for aid, row in inv.items():
        if row["artifact_local_status"] == "missing_source_artifact":
            assert aid not in cand_artifacts


def test_outside_grid_no_patch_link():
    # the only candidate (Charter 758) falls outside the SUSC grid -> 0 patch links
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if s["candidate_geometry_outside_grid_count"] >= 1 and s["reference_candidate_count"] >= 1:
        assert s["candidate_patch_link_count"] == 0
    assert len(read_csv(PATCH_LINKS)) == s["candidate_patch_link_count"]


def test_17b_ineligible_without_qa():
    for row in read_csv(CANDIDATES):
        assert row["eligible_for_17b_now"] == "false"
        assert row["qa_status"] == "needs_review"
    assert json.loads(SUMMARY.read_text(encoding="utf-8"))["eligible_for_17b_now"] is False


def test_formal_queue_has_p0_drm_when_pkg_pet_absent():
    inv = read_csv(INVENTORY)
    missing_events = {a["event_id"] for a in inv if a["artifact_local_status"] == "missing_source_artifact"}
    priorities = {r["priority"] for r in read_csv(FORMAL)}
    if "PET_2022_02_15" in missing_events:
        assert "P0_DRM_RJ_PKG_FR_PET_001" in priorities


def test_formal_queue_has_p0_compdec_when_pkg_rec_absent():
    inv = read_csv(INVENTORY)
    missing_events = {a["event_id"] for a in inv if a["artifact_local_status"] == "missing_source_artifact"}
    priorities = {r["priority"] for r in read_csv(FORMAL)}
    if "REC_2022_05_24_30" in missing_events:
        assert "P0_COMPDEC_RECIFE_2022_POINTS_OR_ADDRESSES" in priorities


def test_empty_or_nonempty_geojson_consistent_with_registry():
    geojson = json.loads(GEOJSON.read_text(encoding="utf-8"))
    registry_ids = {c["reference_candidate_id"] for c in read_csv(CANDIDATES)}
    geojson_ids = {f["properties"]["reference_candidate_id"] for f in geojson["features"]}
    assert geojson_ids == registry_ids
    if not geojson["features"]:
        assert json.loads(SUMMARY.read_text(encoding="utf-8"))["candidate_geometry_count"] == 0


def test_patch_link_requires_candidate_geometry():
    registry_ids = {c["reference_candidate_id"] for c in read_csv(CANDIDATES)}
    for row in read_csv(PATCH_LINKS):
        assert row["reference_candidate_id"] in registry_ids


def test_summary_reflects_real_blocks():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["total_artifacts_inventory"] == len(read_csv(INVENTORY))
    assert s["reference_candidate_count"] == len(read_csv(CANDIDATES))
    assert s["candidate_patch_link_count"] == len(read_csv(PATCH_LINKS))
    assert s["p0_request_count"] == 2
    assert s["missing_source_artifact_count"] >= 2
    assert any("outside" in b or "missing" in b for b in s["blocks_17b"])
