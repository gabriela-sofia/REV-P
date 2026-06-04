"""Tests for v1ue — Event Evidence Scorecard."""

import csv
import os
import subprocess
import sys

import pytest

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1ue_event_evidence_scorecard.py")
EVENTS = os.path.join("datasets", "protocolo_c", "event_candidate_registry.csv")
WINDOWS = os.path.join("datasets", "protocolo_c", "v1ue_event_temporal_window_registry.csv")
STATIONS = os.path.join("datasets", "protocolo_c", "v1ue_station_candidate_registry.csv")
OBSERVATIONS = os.path.join("datasets", "protocolo_c", "v1ue_observation_series_registry.csv")
RESOLUTIONS = os.path.join("datasets", "protocolo_c", "v1ue_official_dataset_resolution_registry.csv")

SCORECARD_COLUMNS = [
    "scorecard_id", "event_id", "region", "city", "hazard_scope",
    "temporal_evidence_score", "hydrometeorological_score",
    "phenomenon_typing_score", "locality_score", "geometry_score",
    "source_authority_score", "independence_score", "review_readiness_score",
    "aggregate_score", "classification", "can_create_ground_reference",
    "can_create_training_label", "ground_truth_operational",
    "supervisor_review_completed", "blocking_summary",
]


@pytest.fixture
def scorecard_output(tmp_path):
    out = str(tmp_path / "scorecard.csv")
    result = subprocess.run(
        [sys.executable, SCRIPT, "--events", EVENTS, "--windows", WINDOWS,
         "--stations", STATIONS, "--observations", OBSERVATIONS,
         "--resolutions", RESOLUTIONS, "--out", out],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    return out


class TestScorecard:
    def test_runs(self, scorecard_output):
        assert os.path.exists(scorecard_output)

    def test_required_columns(self, scorecard_output):
        with open(scorecard_output, "r", encoding="utf-8") as f:
            cols = csv.DictReader(f).fieldnames
        for col in SCORECARD_COLUMNS:
            assert col in cols

    def test_no_ground_truth_promoted(self, scorecard_output):
        with open(scorecard_output, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["can_create_ground_reference"] == "false"
            assert r["can_create_training_label"] == "false"
            assert r["ground_truth_operational"] == "false"
            assert r["supervisor_review_completed"] == "false"

    def test_mixed_event_blocked_phenomenon(self, scorecard_output):
        with open(scorecard_output, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # PET events are mixed -> phenomenon separation block
        pet = [r for r in rows if r["event_id"].startswith("PET")]
        for r in pet:
            assert "PHENOMENON_SEPARATION" in r["classification"] or "PHENOMENON_SEPARATION" in r["blocking_summary"]

    def test_geometry_missing_blocks(self, scorecard_output):
        with open(scorecard_output, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # No event has geometry in this stage -> geometry_score 0
        for r in rows:
            assert float(r["geometry_score"]) == 0.0

    def test_high_score_does_not_promote(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1ue_event_evidence_scorecard import classify

        # Even with high scores, mixed without phenomenon separation is blocked
        scores = {
            "temporal_evidence_score": 1.0,
            "hydrometeorological_score": 1.0,
            "phenomenon_typing_score": 0.2,
            "locality_score": 1.0,
            "geometry_score": 0.0,
            "source_authority_score": 1.0,
        }
        classification, blocking = classify(scores, "mixed", False)
        assert classification == "BLOCKED_PHENOMENON_SEPARATION_REQUIRED"

    def test_no_geometry_high_temporal_not_ready(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1ue_event_evidence_scorecard import classify

        scores = {
            "temporal_evidence_score": 1.0,
            "hydrometeorological_score": 0.5,
            "phenomenon_typing_score": 0.5,
            "locality_score": 0.0,
            "geometry_score": 0.0,
            "source_authority_score": 1.0,
        }
        classification, blocking = classify(scores, "urban_flooding", False)
        assert classification != "READY_FOR_HUMAN_REVIEW"

    def test_aggregate_in_range(self, scorecard_output):
        with open(scorecard_output, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            agg = float(r["aggregate_score"])
            assert 0.0 <= agg <= 1.0
