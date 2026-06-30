"""SUSC-16D review-only calibration candidate tests.

Verify that the SUSC-16D package reproduces the SUSC-16C diagnostic, stays
review-only and fail-closed, never creates score v7 / ground truth / training
data, never treats footprints as ground truth or controls as true negatives,
and that the validator actually fails on each forbidden condition.
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

MATRIX = OUT / "susc_16d_calibration_candidate_matrix.csv"
HYPOTHESES = OUT / "susc_16d_calibration_hypotheses.csv"
FAILURE_TO_WEIGHT = OUT / "susc_16d_failure_mode_to_weight_hypothesis.csv"
FEATURE_POLICY = OUT / "susc_16d_feature_direction_policy.json"
SENSITIVITY_SUMMARY = OUT / "susc_16d_sensitivity_summary.json"
REPORT = OUT / "SUSC_16D_CALIBRATION_CANDIDATE_REVIEW_ONLY_REPORT.md"
PROTOCOL_STUB = OUT / "susc_17a_reference_evidence_protocol_registry_stub.csv"
PROTOCOL_POLICY = OUT / "susc_17a_reference_evidence_protocol_class_policy.json"


def _load_common():
    path = SCRIPTS / "susc_16d_calibration_candidate_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s16d_common", path)
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


def test_pipeline_and_validator_pass():
    for name in (
        "build_susc_16d_calibration_candidate_matrix.py",
        "validate_susc_16d_calibration_candidate.py",
    ):
        result = run_script(name)
        assert result.returncode == 0, name + "\n" + result.stdout + result.stderr


def test_matrix_counts_and_governance():
    rows = read_csv(MATRIX)
    assert len(rows) == 65
    assert len({row["patch_id"] for row in rows}) == 62
    for row in rows:
        assert row["review_only"] == "true"
        assert row["trainable"] == "false"
        assert row["ground_truth"] == "false"
        assert row["score_v6_unchanged"] == "true"
        assert row["score_v7_created"] == "false"
        assert row["eligible_for_score_v7_future"] == "false"
        assert row["has_observed_footprint"] == "true"
        assert row["not_ground_truth_reason"]
        assert row["candidate_adjustment_family"] in {
            "decouple_documentary_component_from_susceptibility",
            "increase_urban_flash_flood_weight",
            "increase_rainfall_trigger_weight",
            "increase_spectral_water_weight",
            "guard_low_documentary_high_physical_signal",
        }


def test_reproduces_16c_diagnostic():
    summary = json.loads(SENSITIVITY_SUMMARY.read_text(encoding="utf-8"))
    assert summary["footprint_patch_links"] == 65
    assert summary["unique_event_patches"] == 62
    assert summary["low_or_medium_event_links"] == 57
    assert summary["documentary_lowest_component_count"] == 65
    assert summary["urban_flash_flood_underrepresented"] == 60
    assert summary["rainfall_trigger_underweighted"] == 5
    assert summary["best_sensitivity_scenario_16c"] == "increase_spectral_water_weight"
    assert summary["best_hit_rate_top_30_simulated_16c"] == "0.366667"


def test_summary_blocked_review_only():
    summary = json.loads(SENSITIVITY_SUMMARY.read_text(encoding="utf-8"))
    assert summary["calibration_candidate_created"] is True
    assert summary["official_score_changed"] is False
    assert summary["score_v7_created"] is False
    assert summary["trainable"] is False
    assert summary["ground_truth"] is False
    assert summary["eligible_for_score_v7_future"] == 0
    assert summary["readiness_status"] == "BLOCKED_REVIEW_ONLY"
    assert summary["footprint_treated_as_ground_truth"] is False
    assert summary["control_treated_as_negative"] is False


def test_five_calibration_hypotheses():
    rows = read_csv(HYPOTHESES)
    ids = {row["hypothesis_id"] for row in rows}
    assert ids == {
        "decouple_documentary_component_from_susceptibility",
        "increase_urban_flash_flood_weight",
        "increase_rainfall_trigger_weight",
        "increase_spectral_water_weight",
        "guard_low_documentary_high_physical_signal",
    }
    for row in rows:
        assert row["justification"]
        assert row["features_involved"]
        assert row["methodological_risk"]
        assert row["eligible_for_score_v7_future"] == "false"


def test_documentary_component_is_evidence_confidence():
    policy = json.loads(FEATURE_POLICY.read_text(encoding="utf-8"))
    assert policy["documentary_component_is_susceptibility_signal"] is False
    assert policy["documentary_absence_is_true_negative"] is False
    assert policy["footprint_is_ground_truth"] is False
    doc = policy["features"]["documentary_component"]
    assert doc["signal_domain"] == "evidence_confidence"
    assert doc["can_change_official_score_v6"] is False
    # physical features stay susceptibility_signal
    assert policy["features"]["urban_prop"]["signal_domain"] == "susceptibility_signal"


def test_reference_protocol_foundation_is_fail_closed():
    assert read_csv(PROTOCOL_STUB) == []  # no invented records
    policy = json.loads(PROTOCOL_POLICY.read_text(encoding="utf-8"))
    strong = {c["geometry_type"] for c in policy["classes"] if c["can_be_strong_evaluation"]}
    assert strong == {
        "official_observed_event_polygon",
        "official_observed_event_point",
        "technical_remote_sensing_flood_footprint",
    }
    assert all(c["can_be_ground_truth"] is False for c in policy["classes"])
    assert all(c["can_be_true_negative"] is False for c in policy["classes"])
    assert len(policy["classes"]) == 8


def test_report_mandatory_statement():
    text = REPORT.read_text(encoding="utf-8")
    assert "nao altera" in text and "score v6 oficial" in text
    assert "nao cria score v7" in text
    assert "BLOCKED_REVIEW_ONLY" in text


def test_validator_fails_on_forbidden_rows():
    s16d = _load_common()
    clean = {
        "review_only": "true", "trainable": "false", "ground_truth": "false",
        "score_v6_unchanged": "true", "score_v7_created": "false",
        "eligible_for_score_v7_future": "false", "not_ground_truth_reason": "x",
        "candidate_adjustment_family": "increase_urban_flash_flood_weight",
        "has_observed_footprint": "true",
    }
    assert s16d.matrix_row_violations(clean) == []
    assert s16d.matrix_row_violations({**clean, "score_v7_created": "true"})
    assert s16d.matrix_row_violations({**clean, "ground_truth": "true"})
    assert s16d.matrix_row_violations({**clean, "trainable": "true"})
    assert s16d.matrix_row_violations({**clean, "eligible_for_score_v7_future": "true"})
    assert s16d.matrix_row_violations({**clean, "eligible_for_score_v7_future": "3"})
    assert s16d.matrix_row_violations({**clean, "not_ground_truth_reason": ""})
    assert s16d.matrix_row_violations({**clean, "score_v6_unchanged": "false"})
