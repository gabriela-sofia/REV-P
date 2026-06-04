"""Tests for v1uf — Hydromet Window Metrics."""

import csv
import os
import sys
from datetime import datetime

POLICY = os.path.join("configs", "protocolo_c", "v1uf_hydromet_metrics_policy.yaml")
METRICS_CSV = os.path.join("datasets", "protocolo_c", "v1uf_hydromet_window_metrics_registry.csv")

METRIC_COLUMNS = [
    "metric_id", "event_id", "station_candidate_id", "source_id",
    "window_type", "window_start", "window_end", "observed_variable",
    "precipitation_total_mm", "precipitation_max_hourly_mm",
    "precipitation_max_daily_mm", "valid_observation_count",
    "missing_observation_count", "coverage_ratio", "temporal_overlap_status",
    "metric_status", "evidence_role", "can_support_temporal_gate",
    "can_create_ground_reference", "can_create_training_label", "limitations",
]


def _import():
    sys.path.insert(0, os.path.abspath("."))
    from scripts.protocolo_c import revp_v1uf_hydromet_window_metrics as m
    return m


class TestParsing:
    def test_parse_value_br_decimal(self):
        m = _import()
        assert m.parse_value_br("1,4", ["", "-9999"]) == 1.4
        assert m.parse_value_br("", ["", "-9999"]) is None
        assert m.parse_value_br("-9999", ["", "-9999"]) is None

    def test_parse_datetime_inmet_format(self):
        m = _import()
        dt = m.parse_datetime_cell("2022/01/01", "0000 UTC")
        assert dt == datetime(2022, 1, 1, 0, 0)
        dt2 = m.parse_datetime_cell("2022/02/15", "1300 UTC")
        assert dt2 == datetime(2022, 2, 15, 13, 0)

    def test_compute_window_metrics_basic(self):
        m = _import()
        series = [
            (datetime(2022, 2, 15, 0, 0), 2.0),
            (datetime(2022, 2, 15, 1, 0), 3.0),
            (datetime(2022, 2, 15, 2, 0), None),
        ]
        result = m.compute_window_metrics(
            series, datetime(2022, 2, 15, 0, 0), datetime(2022, 2, 15, 23, 59, 59))
        assert result["total"] == "5.0"
        assert result["max_hourly"] == "3.0"
        assert result["valid_count"] == "2"
        assert result["missing_count"] == "1"

    def test_insufficient_coverage(self):
        m = _import()
        # empty series over a window -> insufficient
        result = m.compute_window_metrics(
            [], datetime(2022, 5, 24, 0, 0), datetime(2022, 5, 30, 23, 59, 59))
        assert result["status"] in ("INSUFFICIENT_COVERAGE", "NO_DATA")


class TestMetricsRegistry:
    def test_registry_columns(self):
        if not os.path.exists(METRICS_CSV):
            return
        with open(METRICS_CSV, "r", encoding="utf-8") as f:
            cols = csv.DictReader(f).fieldnames
        for col in METRIC_COLUMNS:
            assert col in cols

    def test_no_ground_reference_or_label(self):
        if not os.path.exists(METRICS_CSV):
            return
        with open(METRICS_CSV, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["can_create_ground_reference"] == "false"
            assert r["can_create_training_label"] == "false"

    def test_precipitation_does_not_become_truth(self):
        """Even with computed precipitation, evidence_role stays temporal_anchor."""
        if not os.path.exists(METRICS_CSV):
            return
        with open(METRICS_CSV, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["evidence_role"] == "temporal_anchor"
            assert "temporal" in r["limitations"].lower() or "not patch" in r["limitations"].lower()

    def test_insufficient_coverage_blocks_temporal_gate(self):
        if not os.path.exists(METRICS_CSV):
            return
        with open(METRICS_CSV, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if r["metric_status"] == "INSUFFICIENT_COVERAGE":
                assert r["can_support_temporal_gate"] == "false"
