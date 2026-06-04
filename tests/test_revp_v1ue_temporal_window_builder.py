"""Tests for v1ue — Temporal Window Builder."""

import csv
import os
import subprocess
import sys
from datetime import datetime

import pytest

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1ue_temporal_window_builder.py")
EVENTS = os.path.join("datasets", "protocolo_c", "event_candidate_registry.csv")
WINDOWS_CONFIG = os.path.join("configs", "protocolo_c", "v1ue_event_temporal_windows.yaml")

WINDOW_COLUMNS = [
    "window_id", "event_id", "region", "city", "start_date", "end_date",
    "window_type", "window_start", "window_end", "purpose",
    "can_support_temporal_gate", "can_create_label",
]


@pytest.fixture
def built_windows(tmp_path):
    out = str(tmp_path / "windows.csv")
    result = subprocess.run(
        [sys.executable, SCRIPT, "--events", EVENTS,
         "--windows-config", WINDOWS_CONFIG, "--out", out],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    return out


class TestWindowBuilder:
    def test_dry_run(self, tmp_path):
        result = subprocess.run(
            [sys.executable, SCRIPT, "--events", EVENTS,
             "--windows-config", WINDOWS_CONFIG, "--dry-run"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0
        assert "DRY RUN" in result.stdout

    def test_required_columns(self, built_windows):
        with open(built_windows, "r", encoding="utf-8") as f:
            cols = csv.DictReader(f).fieldnames
        for col in WINDOW_COLUMNS:
            assert col in cols

    def test_five_windows_per_event(self, built_windows):
        with open(built_windows, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        events = set(r["event_id"] for r in rows)
        for ev in events:
            ev_rows = [r for r in rows if r["event_id"] == ev]
            assert len(ev_rows) == 5, f"{ev} has {len(ev_rows)} windows"

    def test_core_window_matches_event_dates(self, built_windows):
        with open(built_windows, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if r["window_type"] == "event_core_window":
                assert r["window_start"] == r["start_date"]
                assert r["window_end"] == r["end_date"]

    def test_pre_event_3d_is_3_days_before(self, built_windows):
        with open(built_windows, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if r["window_type"] == "pre_event_window_3d":
                start = datetime.strptime(r["start_date"], "%Y-%m-%d").date()
                wstart = datetime.strptime(r["window_start"], "%Y-%m-%d").date()
                assert (start - wstart).days == 3

    def test_pre_event_7d_is_7_days_before(self, built_windows):
        with open(built_windows, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if r["window_type"] == "pre_event_window_7d":
                start = datetime.strptime(r["start_date"], "%Y-%m-%d").date()
                wstart = datetime.strptime(r["window_start"], "%Y-%m-%d").date()
                assert (start - wstart).days == 7

    def test_no_can_create_label_true(self, built_windows):
        with open(built_windows, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["can_create_label"] == "false"

    def test_sentinel_window_no_temporal_gate(self, built_windows):
        with open(built_windows, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if r["window_type"] == "sentinel_link_window":
                assert r["can_support_temporal_gate"] == "false"
