"""SUSC-17C Strong Reference Acquisition Canary tests (review-only, fail-closed).

Verify the canary pipeline builds every artifact, follows schemas, stays
review-only, never creates score v7 / ground truth / training, never lets an
undated event become a SAR candidate, never lets month-only/unknown become
strong, never treats alerts as occurrences or risk areas as events, requires
date+geometry or date+SAR for strong candidates, keeps every priority canary in
the QA queue, and that the validator fails on every forbidden condition. IDs are
deterministic and the build byte-identical across runs.
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

TARGET_PACK = OUT / "susc_17c_candidate_event_target_pack.csv"
SAR_FEASIBILITY = OUT / "susc_17c_sar_feasibility_pack.csv"
PRIORITY = OUT / "susc_17c_priority_canary_events.csv"
QA_QUEUE = OUT / "susc_17c_human_qa_queue.csv"
SUMMARY = OUT / "susc_17c_strong_reference_acquisition_summary.json"
SOURCE_INV = OUT / "susc_17c_source_inventory.csv"
CANDIDATE_SCHEMA = SCHEMAS / "susc_17c_candidate_event_schema_v1.json"
SAR_SCHEMA = SCHEMAS / "susc_17c_sar_feasibility_schema_v1.json"


def _load_common():
    path = SCRIPTS / "susc_17c_strong_reference_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17c_common", path)
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


CLEAN_TARGET = {
    "candidate_event_id": "S17C_E_x", "source_id": "SRC_X", "city": "recife",
    "region": "recife", "phenomenon_type": "urban_flooding",
    "event_date_candidate": "2014-01-15", "event_date_precision": "exact_day",
    "date_resolution_status": "resolved", "geometry_status": "aoi_bbox_only",
    "geometry_type": "bbox", "has_official_geometry": "false", "has_patch_link": "false",
    "has_pre_event_window": "true", "has_post_event_window": "true",
    "occurrence_status": "observed_occurrence", "sar_candidate": "true",
    "strong_reference_candidate": "true",
    "expected_strong_class": "technical_remote_sensing_flood_footprint",
    "blocking_reason": "not_available", "priority_score": "0.500",
    "review_only": "true", "trainable": "false", "ground_truth": "false",
}


def test_build_generates_all_artifacts():
    assert run_script("build_susc_17c_strong_reference_acquisition.py").returncode == 0
    for path in (SOURCE_INV, TARGET_PACK, SAR_FEASIBILITY, PRIORITY, QA_QUEUE, SUMMARY):
        assert path.exists(), path
    assert run_script("validate_susc_17c_strong_reference_acquisition.py").returncode == 0


def test_target_pack_and_sar_follow_schema():
    s = _load_common()
    cand_schema = json.loads(CANDIDATE_SCHEMA.read_text(encoding="utf-8"))
    sar_schema = json.loads(SAR_SCHEMA.read_text(encoding="utf-8"))
    for row in read_csv(TARGET_PACK):
        assert s.schema_violations(row, cand_schema) == []
    for row in read_csv(SAR_FEASIBILITY):
        assert s.schema_violations(row, sar_schema) == []


def test_ids_unique_and_deterministic():
    ids = [r["candidate_event_id"] for r in read_csv(TARGET_PACK)]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)
    qa_ids = [r["qa_item_id"] for r in read_csv(QA_QUEUE)]
    assert len(qa_ids) == len(set(qa_ids))
    assert qa_ids == sorted(qa_ids)


def test_build_twice_byte_identical():
    run_script("build_susc_17c_strong_reference_acquisition.py")
    first = (TARGET_PACK.read_bytes(), SAR_FEASIBILITY.read_bytes(),
             PRIORITY.read_bytes(), SUMMARY.read_bytes())
    run_script("build_susc_17c_strong_reference_acquisition.py")
    second = (TARGET_PACK.read_bytes(), SAR_FEASIBILITY.read_bytes(),
              PRIORITY.read_bytes(), SUMMARY.read_bytes())
    assert first == second


def test_no_score_v7():
    assert not (ROOT / "datasets" / "suscetibilidade" / "susc_score_v7_candidate_by_patch_v1.csv").exists()
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["score_v7_created"] is False


def test_no_score_v6_change():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["score_v6_changed"] is False


def test_no_trainable_or_ground_truth():
    for row in read_csv(TARGET_PACK) + read_csv(SAR_FEASIBILITY) + read_csv(PRIORITY) + read_csv(QA_QUEUE):
        assert row["trainable"] == "false"
        assert row["ground_truth"] == "false"
        assert row["review_only"] == "true"


def test_undated_event_is_not_sar_candidate():
    for row in read_csv(TARGET_PACK):
        if row["event_date_precision"] not in ("exact_day", "date_range"):
            assert row["sar_candidate"] == "false"


def test_month_only_is_not_strong():
    s = _load_common()
    bad = {**CLEAN_TARGET, "event_date_precision": "month_only", "strong_reference_candidate": "true"}
    assert s.target_row_violations(bad)


def test_alert_is_not_occurrence():
    s = _load_common()
    bad = {**CLEAN_TARGET, "occurrence_status": "alert", "sar_candidate": "true"}
    assert s.target_row_violations(bad)
    # also enforced in real data
    for row in read_csv(TARGET_PACK):
        if row["occurrence_status"] == "alert":
            assert row["sar_candidate"] == "false"
            assert row["strong_reference_candidate"] == "false"


def test_risk_area_is_not_event():
    s = _load_common()
    bad = {**CLEAN_TARGET, "occurrence_status": "risk_area", "strong_reference_candidate": "true"}
    assert s.target_row_violations(bad)
    for row in read_csv(TARGET_PACK):
        if row["occurrence_status"] == "risk_area":
            assert row["strong_reference_candidate"] == "false"


def test_admin_without_fine_geometry_not_strong():
    s = _load_common()
    bad = {**CLEAN_TARGET, "occurrence_status": "administrative",
           "has_official_geometry": "false", "strong_reference_candidate": "true"}
    assert s.target_row_violations(bad)


def test_strong_requires_date_plus_geometry_or_sar():
    for row in read_csv(TARGET_PACK):
        if row["strong_reference_candidate"] == "true":
            dated = row["event_date_precision"] in ("exact_day", "date_range")
            assert dated
            assert row["has_official_geometry"] == "true" or row["sar_candidate"] == "true"


def test_priority_canary_viable_or_blocked_with_reason():
    targets = {r["candidate_event_id"]: r for r in read_csv(TARGET_PACK)}
    priority = read_csv(PRIORITY)
    assert 3 <= len(priority) <= 5
    for row in priority:
        tgt = targets[row["candidate_event_id"]]
        assert tgt["sar_candidate"] == "true" or row["blocking_risk"]


def test_every_priority_canary_has_qa_item():
    qa_ids = {r["candidate_event_id"] for r in read_csv(QA_QUEUE)}
    for row in read_csv(PRIORITY):
        assert row["candidate_event_id"] in qa_ids
        assert row["qa_required"] == "true"


def test_summary_reflects_real_counts():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    targets = read_csv(TARGET_PACK)
    assert s["total_candidate_events"] == len(targets)
    assert s["total_sources"] == len(read_csv(SOURCE_INV))
    assert s["sar_candidate_count"] == sum(1 for t in targets if t["sar_candidate"] == "true")
    assert s["priority_canary_count"] == len(read_csv(PRIORITY))
    assert s["human_qa_items_count"] == len(read_csv(QA_QUEUE))
    assert s["strong_reference_created_count"] == 0
    assert s["technical_footprint_created_count"] == 0


def test_failclosed_strong_without_date():
    s = _load_common()
    bad = {**CLEAN_TARGET, "event_date_precision": "not_available",
           "strong_reference_candidate": "true", "sar_candidate": "false"}
    assert s.target_row_violations(bad)


def test_failclosed_technical_footprint_without_qa():
    # The QA queue must keep technical-footprint canaries under human review.
    qa = read_csv(QA_QUEUE)
    tech = [q for q in qa if q["evidence_type"].startswith("technical_remote_sensing")]
    for q in tech:
        assert q["default_status"] == "needs_review"
        assert q["blocking_if_unreviewed"] == "true"
        assert q["ground_truth"] == "false"


def test_failclosed_score_v7_created():
    s = _load_common()
    assert s.target_row_violations({**CLEAN_TARGET, "score_v7_created": "true"})


def test_failclosed_ground_truth():
    s = _load_common()
    assert s.target_row_violations({**CLEAN_TARGET, "ground_truth": "true"})
    assert s.target_row_violations({**CLEAN_TARGET, "official_score_changed": "true"})
    assert s.target_row_violations(CLEAN_TARGET) == []
