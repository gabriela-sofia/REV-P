"""v2az - replay plan, log and readiness certificate tests."""

from __future__ import annotations

import csv
from pathlib import Path

import scripts.v2az_turning_point_replay_orchestrator as engine
from v2az_test_helpers import make_dataset


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_canonical_dry_run_executes_no_subprocess():
    logs = read(ROOT / "datasets" / "v2az_replay_execution_log.csv")
    assert len(logs) == 5
    assert all(row["mode"] == "dry_run" and row["executed"] == "false" for row in logs)


def test_replay_without_valid_geometry_executes_nothing(tmp_path):
    ds, out, cfg, docs = make_dataset(tmp_path)
    code, summary = engine.run("replay", str(ds), str(out), str(cfg), str(docs))
    assert code == 0 and summary["can_attempt_replay"] is False
    logs = read(ds / "v2az_replay_execution_log.csv")
    assert all(row["executed"] == "false" for row in logs)


def test_certificate_records_current_fail_closed_state():
    cert = read(ROOT / "datasets" / "v2az_replay_readiness_certificate.csv")[0]
    assert cert["valid_patch_boundaries"] == cert["valid_event_polygons"] == "0"
    assert cert["can_replay_v2au"] == cert["can_attempt_tp4"] == "false"
    assert cert["turning_point_level"] == "TP0_DOCUMENTED_ABSENCE"
