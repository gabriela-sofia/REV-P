"""SUSC-17A Reference Evidence Protocol tests (review-only, fail-closed).

Verify the protocol builds registry / tiers / policy / summary, follows the
schema, stays review-only, never creates training / ground truth / score v7,
restricts strong evaluation to the strong classes, blocks context/administrative
classes from patch validation, and that the validator fails on every forbidden
condition. IDs must be unique and the build deterministic.
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

REGISTRY = OUT / "susc_17a_reference_evidence_registry.csv"
QUALITY_TIERS = OUT / "susc_17a_reference_evidence_quality_tiers.csv"
CLASS_POLICY = OUT / "susc_17a_reference_evidence_class_policy.json"
SUMMARY = OUT / "susc_17a_reference_evidence_summary.json"
REPORT = OUT / "SUSC_17A_REFERENCE_EVIDENCE_PROTOCOL_REPORT.md"
SCHEMA = SCHEMAS / "susc_17a_reference_evidence_protocol_schema_v1.json"


def _load_common():
    path = SCRIPTS / "susc_17a_reference_evidence_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17a_common", path)
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


CLEAN_ROW = {
    "reference_evidence_id": "REF17A_x", "source_id": "S16AFP_0", "event_id": "E0",
    "geometry_id": "E0", "patch_link_id": "PL0", "patch_id": "p0",
    "event_date": "not_available", "pre_event_window": "not_available",
    "post_event_window": "not_available", "city": "Curitiba", "region": "curitiba",
    "phenomenon_type": "flood", "geometry_type": "Polygon",
    "spatial_resolution": "coarse_bbox_polygon", "temporal_resolution": "not_available",
    "source_authority": "local_project_official_artifact", "source_confidence": "moderate",
    "annotation_method": "local_geometry_parse", "uncertainty_m": "not_available",
    "link_quality": "high_spatial_overlap", "evidence_class": "official_observed_event_polygon",
    "quality_tier": "TIER_B_MODERATE_GEOMETRY_PATCH_LINKED",
    "reference_level": "score_evaluation_reference", "represents_observed_occurrence": "true",
    "eligible_for_evaluation": "true", "eligible_for_strong_evaluation": "true",
    "eligible_for_calibration": "true", "eligible_for_score_v7_future": "false",
    "review_only": "true", "trainable": "false", "ground_truth": "false",
    "not_ground_truth_reason": "x", "no_negative_control_reason": "y",
    "source_artifact": "a", "lineage_notes": "n",
}


def test_build_generates_all_artifacts():
    result = run_script("build_susc_17a_reference_evidence_protocol.py")
    assert result.returncode == 0, result.stdout + result.stderr
    for path in (REGISTRY, QUALITY_TIERS, CLASS_POLICY, SUMMARY, REPORT):
        assert path.exists(), path
    assert run_script("validate_susc_17a_reference_evidence_protocol.py").returncode == 0


def test_registry_follows_schema():
    s17a = _load_common()
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    rows = read_csv(REGISTRY)
    assert rows
    for row in rows:
        assert s17a.schema_violations(row, schema) == []


def test_all_rows_review_only():
    rows = read_csv(REGISTRY)
    assert all(row["review_only"] == "true" for row in rows)


def test_no_row_trainable_or_ground_truth():
    rows = read_csv(REGISTRY)
    assert all(row["trainable"] == "false" for row in rows)
    assert all(row["ground_truth"] == "false" for row in rows)
    assert all(row["eligible_for_score_v7_future"] == "false" for row in rows)
    assert all(row["not_ground_truth_reason"] for row in rows)


def test_strong_eval_only_for_strong_classes():
    s17a = _load_common()
    rows = read_csv(REGISTRY)
    strong = [r for r in rows if r["eligible_for_strong_evaluation"] == "true"]
    assert strong
    assert all(r["evidence_class"] in s17a.STRONG_CLASSES for r in strong)


def test_context_classes_do_not_validate_patch():
    s17a = _load_common()
    for cls in s17a.CONTEXT_CLASSES:
        bad = {**CLEAN_ROW, "evidence_class": cls, "eligible_for_evaluation": "true",
               "represents_observed_occurrence": "false"}
        assert s17a.row_governance_violations(bad)


def test_risk_area_not_event_is_not_occurrence():
    s17a = _load_common()
    bad = {**CLEAN_ROW, "evidence_class": "risk_area_not_event",
           "eligible_for_evaluation": "false", "eligible_for_strong_evaluation": "false",
           "eligible_for_calibration": "false", "represents_observed_occurrence": "true"}
    assert s17a.row_governance_violations(bad)


def test_alert_only_is_not_occurrence():
    s17a = _load_common()
    bad = {**CLEAN_ROW, "evidence_class": "alert_only",
           "eligible_for_evaluation": "false", "eligible_for_strong_evaluation": "false",
           "eligible_for_calibration": "false", "represents_observed_occurrence": "true"}
    assert s17a.row_governance_violations(bad)


def test_administrative_record_without_fine_geometry_does_not_validate_patch():
    s17a = _load_common()
    # administrative_disaster_record is a context class: must not validate patch.
    bad = {**CLEAN_ROW, "evidence_class": "administrative_disaster_record",
           "geometry_type": "not_available", "eligible_for_evaluation": "true"}
    assert s17a.row_governance_violations(bad)


def test_official_address_resolved_requires_uncertainty_and_never_strong():
    s17a = _load_common()
    no_unc = {**CLEAN_ROW, "evidence_class": "official_address_resolved",
              "eligible_for_strong_evaluation": "false", "eligible_for_calibration": "false",
              "uncertainty_m": ""}
    assert s17a.row_governance_violations(no_unc)
    strong = {**CLEAN_ROW, "evidence_class": "official_address_resolved",
              "uncertainty_m": "50", "eligible_for_strong_evaluation": "true"}
    assert s17a.row_governance_violations(strong)


def test_failclosed_negative_ground_truth():
    s17a = _load_common()
    assert s17a.row_governance_violations({**CLEAN_ROW, "negative_ground_truth": "true"})


def test_failclosed_eligible_score_v7_future():
    s17a = _load_common()
    assert s17a.row_governance_violations({**CLEAN_ROW, "eligible_for_score_v7_future": "true"})
    assert s17a.row_governance_violations({**CLEAN_ROW, "eligible_for_score_v7_future": "3"})


def test_failclosed_score_v7_created():
    s17a = _load_common()
    assert s17a.row_governance_violations({**CLEAN_ROW, "score_v7_created": "true"})


def test_failclosed_official_score_changed():
    s17a = _load_common()
    assert s17a.row_governance_violations({**CLEAN_ROW, "official_score_changed": "true"})


def test_ids_unique_and_deterministic():
    rows = read_csv(REGISTRY)
    ids = [r["reference_evidence_id"] for r in rows]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)


def test_build_twice_is_byte_identical():
    run_script("build_susc_17a_reference_evidence_protocol.py")
    first = REGISTRY.read_bytes(), SUMMARY.read_bytes(), CLASS_POLICY.read_bytes()
    run_script("build_susc_17a_reference_evidence_protocol.py")
    second = REGISTRY.read_bytes(), SUMMARY.read_bytes(), CLASS_POLICY.read_bytes()
    assert first == second


def test_clean_row_passes_all_guards():
    s17a = _load_common()
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert s17a.row_governance_violations(CLEAN_ROW) == []
    assert s17a.schema_violations(CLEAN_ROW, schema) == []
