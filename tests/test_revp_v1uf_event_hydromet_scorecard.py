"""Tests for v1uf — Event Hydromet Scorecard."""

import csv
import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1uf_event_hydromet_scorecard.py")
EVENTS = os.path.join("datasets", "protocolo_c", "event_candidate_registry.csv")
ASSETS = os.path.join("datasets", "protocolo_c", "v1uf_station_series_asset_registry.csv")
CATALOG = os.path.join("datasets", "protocolo_c", "v1uf_official_station_catalog_registry.csv")
BINDING = os.path.join("datasets", "protocolo_c", "v1uf_station_binding_registry.csv")
METRICS = os.path.join("datasets", "protocolo_c", "v1uf_hydromet_window_metrics_registry.csv")
V1UE_SC = os.path.join("datasets", "protocolo_c", "v1ue_event_evidence_scorecard.csv")

SCORECARD_COLUMNS = [
    "event_id", "has_official_station_series", "has_precipitation_during_event",
    "has_pre_event_precipitation", "has_temporal_anchor", "has_station_coordinates",
    "has_spatial_event_geometry", "has_phenomenon_separation",
    "hydromet_evidence_level", "hydromet_summary", "remaining_blocker",
    "can_support_ground_reference_future", "can_create_ground_reference",
    "can_create_training_label", "next_best_action",
]


def _run(tmp_path):
    out = str(tmp_path / "out")
    report = str(tmp_path / "report.md")
    result = subprocess.run(
        [sys.executable, SCRIPT, "--events", EVENTS, "--assets", ASSETS,
         "--catalog", CATALOG, "--binding", BINDING, "--metrics", METRICS,
         "--v1ue-scorecard", V1UE_SC, "--out-dir", out, "--out-report", report],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    return out, report


class TestScorecard:
    def test_runs_and_columns(self, tmp_path):
        out, _ = _run(tmp_path)
        sc = os.path.join(out, "v1uf_event_hydromet_scorecard.csv")
        assert os.path.exists(sc)
        with open(sc, "r", encoding="utf-8") as f:
            cols = csv.DictReader(f).fieldnames
        for col in SCORECARD_COLUMNS:
            assert col in cols

    def test_no_ground_reference_or_label(self, tmp_path):
        out, _ = _run(tmp_path)
        sc = os.path.join(out, "v1uf_event_hydromet_scorecard.csv")
        with open(sc, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["can_create_ground_reference"] == "false"
            assert r["can_create_training_label"] == "false"
            assert r["has_spatial_event_geometry"] == "false"

    def test_mixed_event_blocked(self, tmp_path):
        out, _ = _run(tmp_path)
        sc = os.path.join(out, "v1uf_event_hydromet_scorecard.csv")
        with open(sc, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        pet = [r for r in rows if r["event_id"].startswith("PET")]
        for r in pet:
            assert r["hydromet_evidence_level"] == "BLOCKED_PHENOMENON_SEPARATION_REQUIRED"

    def test_rec_blocked_no_geometry_or_coverage(self, tmp_path):
        out, _ = _run(tmp_path)
        sc = os.path.join(out, "v1uf_event_hydromet_scorecard.csv")
        with open(sc, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        rec = [r for r in rows if r["event_id"].startswith("REC")]
        for r in rec:
            assert r["hydromet_evidence_level"].startswith("BLOCKED") or \
                   r["remaining_blocker"] in ("EVENT_GEOMETRY_MISSING", "INSUFFICIENT_COVERAGE", "STATION_COORDINATES_MISSING")

    def test_strong_rain_does_not_promote(self):
        """classify_hydromet: even with everything except geometry, no ground reference."""
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1uf_event_hydromet_scorecard import classify_hydromet
        # full hydromet, single hazard, but no geometry
        level, blocker = classify_hydromet(
            has_series=True, has_event_precip=True, has_pre_precip=True,
            has_coord=True, has_geometry=False, has_phenomenon_sep=True,
            hazard_scope="urban_flooding", coverage_ok=True)
        assert level == "TEMPORAL_HYDROMET_ANCHOR_CONFIRMED"
        assert blocker == "EVENT_GEOMETRY_MISSING"

    def test_mixed_blocks_before_hydromet(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1uf_event_hydromet_scorecard import classify_hydromet
        level, blocker = classify_hydromet(
            has_series=True, has_event_precip=True, has_pre_precip=True,
            has_coord=True, has_geometry=False, has_phenomenon_sep=False,
            hazard_scope="mixed", coverage_ok=True)
        assert level == "BLOCKED_PHENOMENON_SEPARATION_REQUIRED"

    def test_report_has_guardrails(self, tmp_path):
        _, report = _run(tmp_path)
        with open(report, "r", encoding="utf-8") as f:
            content = f.read()
        assert "ground_truth_operational" in content
        assert "can_create_ground_reference" in content
        assert "no_coordinates_invented" in content
        assert "estação oficial não é geometria de inundação" in content
