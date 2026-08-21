"""SUSC-17C3 Official Source Acquisition & Patch Coverage Audit tests.

Verify the pipeline builds every artifact, follows schemas, stays review-only,
never creates score v7 / ground truth / training, never treats PE3D/MDE or risk
areas as observed events, never lets a PDF-without-vector or a neighborhood
centroid become strong geometry, never creates a patch-link for geometry that is
absent or outside the grid, keeps 17B ineligible without QA, surfaces the P0
DRM-RJ / COMPDEC packages, and reflects the real blocks. IDs deterministic,
build byte-identical.
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

SOURCE_TARGETS = OUT / "susc_17c3_official_source_acquisition_targets.csv"
EVIDENCE_LADDER = OUT / "susc_17c3_candidate_event_evidence_ladder.csv"
COVERAGE_AUDIT = OUT / "susc_17c3_patch_coverage_audit.csv"
MISSING_MANIFEST = OUT / "susc_17c3_missing_official_artifacts_manifest.csv"
DECISION_TABLE = OUT / "susc_17c3_next_action_decision_table.csv"
SUMMARY = OUT / "susc_17c3_summary.json"
SOURCE_SCHEMA = SCHEMAS / "susc_17c3_source_acquisition_target_schema_v1.json"
COVERAGE_SCHEMA = SCHEMAS / "susc_17c3_patch_coverage_audit_schema_v1.json"


def _load_common():
    path = SCRIPTS / "susc_17c3_official_source_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17c3_common", path)
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
    assert run_script("build_susc_17c3_official_source_acquisition.py").returncode == 0
    for path in (SOURCE_TARGETS, EVIDENCE_LADDER, COVERAGE_AUDIT, MISSING_MANIFEST,
                 DECISION_TABLE, SUMMARY):
        assert path.exists(), path
    assert run_script("validate_susc_17c3_official_source_acquisition.py").returncode == 0


def test_schemas_valid():
    s = _load_common()
    src_schema = json.loads(SOURCE_SCHEMA.read_text(encoding="utf-8"))
    cov_schema = json.loads(COVERAGE_SCHEMA.read_text(encoding="utf-8"))
    for row in read_csv(SOURCE_TARGETS):
        assert s.schema_violations(row, src_schema) == []
    for row in read_csv(COVERAGE_AUDIT):
        assert s.schema_violations(row, cov_schema) == []


def test_ids_deterministic():
    for path, key in ((SOURCE_TARGETS, "source_target_id"), (COVERAGE_AUDIT, "coverage_audit_id"),
                      (MISSING_MANIFEST, "artifact_request_id"), (DECISION_TABLE, "decision_id")):
        ids = [r[key] for r in read_csv(path)]
        assert len(ids) == len(set(ids))
        assert ids == sorted(ids)


def test_build_twice_byte_identical():
    run_script("build_susc_17c3_official_source_acquisition.py")
    first = tuple(p.read_bytes() for p in (SOURCE_TARGETS, COVERAGE_AUDIT, MISSING_MANIFEST, SUMMARY))
    run_script("build_susc_17c3_official_source_acquisition.py")
    second = tuple(p.read_bytes() for p in (SOURCE_TARGETS, COVERAGE_AUDIT, MISSING_MANIFEST, SUMMARY))
    assert first == second


def test_no_score_v7():
    assert not (ROOT / "datasets" / "suscetibilidade" / "susc_score_v7_candidate_by_patch_v1.csv").exists()
    assert json.loads(SUMMARY.read_text(encoding="utf-8"))["score_v7_created"] is False


def test_no_score_v6_change():
    assert json.loads(SUMMARY.read_text(encoding="utf-8"))["score_v6_changed"] is False


def test_no_trainable():
    for path in (SOURCE_TARGETS, EVIDENCE_LADDER, COVERAGE_AUDIT, MISSING_MANIFEST, DECISION_TABLE):
        for row in read_csv(path):
            assert row["trainable"] == "false"


def test_no_ground_truth():
    for path in (SOURCE_TARGETS, EVIDENCE_LADDER, COVERAGE_AUDIT, MISSING_MANIFEST, DECISION_TABLE):
        for row in read_csv(path):
            assert row["ground_truth"] == "false"


def test_pe3d_is_not_event():
    for row in read_csv(SOURCE_TARGETS):
        if "pe3d" in row["source_authority"].lower() or "pe3d" in row["source_name"].lower():
            assert row["expected_revp_class"] == "contextual_physical_layer"
            assert row["can_be_patch_level_if_acquired"] == "false"


def test_pdf_without_vector_not_strong():
    for row in read_csv(SOURCE_TARGETS):
        if row["source_artifact_status"] == "pdf_only":
            assert row["requires_pdf_extraction"] == "true"
            # pdf-only never asserts an event polygon as current evidence
    for row in read_csv(EVIDENCE_LADDER):
        if row["current_revp_class"] == "documentary_context":
            assert row["has_geometry"] == "false"


def test_risk_area_not_event():
    # no source target or ladder row may classify a risk area as an observed event
    for row in read_csv(SOURCE_TARGETS):
        assert "risk_area" not in row["expected_revp_class"]
        assert row["expected_revp_class"] != "official_observed_event_polygon" or \
            row["source_artifact_status"] != "not_applicable_context_only"


def test_neighborhood_centroid_not_strong_geometry():
    for row in read_csv(SOURCE_TARGETS):
        gt = row["expected_geometry_type"].lower()
        if "centroid" in gt or "neighborhood" in gt or "bairro" in gt:
            assert "polygon" not in row["expected_revp_class"].lower()


def test_no_geometry_no_patch_link():
    for row in read_csv(COVERAGE_AUDIT):
        if row["geometry_available"] == "false":
            assert str(row["intersecting_patch_count"]) == "0"
            assert row["geometry_bbox"] == "not_available"


def test_outside_grid_no_patch_link():
    for row in read_csv(COVERAGE_AUDIT):
        if row["coverage_status"] == "outside_patch_grid":
            assert str(row["intersecting_patch_count"]) == "0"


def test_17b_ineligible_without_qa():
    for row in read_csv(EVIDENCE_LADDER):
        assert row["eligible_for_17b_now"] == "false"
        assert row["has_qa"] == "false"
    assert json.loads(SUMMARY.read_text(encoding="utf-8"))["eligible_for_17b_now"] is False


def test_p0_includes_drm_rj_pkg_fr_pet_001():
    priorities = {r["priority"] for r in read_csv(MISSING_MANIFEST)}
    assert "P0_DRM_RJ_PKG_FR_PET_001" in priorities


def test_p0_includes_compdec_recife():
    priorities = {r["priority"] for r in read_csv(MISSING_MANIFEST)}
    assert "P0_COMPDEC_RECIFE_2022_POINTS_OR_ADDRESSES" in priorities


def test_summary_reflects_real_blocks():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["total_source_targets"] == len(read_csv(SOURCE_TARGETS))
    assert s["coverage_audits_count"] == len(read_csv(COVERAGE_AUDIT))
    assert s["intersects_patch_grid_count"] == 0
    assert s["outside_patch_grid_count"] >= 1  # International Charter outside grid
    assert "REC_2022_05_24_30" in s["events_outside_grid"]
    assert s["p0_targets"] == 2
    assert any("outside_patch_grid" in b for b in s["blocks_17b"])


def test_next_action_within_allowed_set():
    s = _load_common()
    for row in read_csv(DECISION_TABLE):
        assert row["next_action"] in s.NEXT_ACTIONS
