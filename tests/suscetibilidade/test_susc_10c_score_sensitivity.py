"""SUSC-10C score sensitivity tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCEN = OUT / "SUSC_10C_score_sensitivity_scenarios.csv"
PATCH = OUT / "SUSC_10C_score_sensitivity_by_patch.csv"
SUMMARY = OUT / "SUSC_10C_score_sensitivity_summary.json"
REPORT = OUT / "SUSC_10C_score_sensitivity_report.md"

REQUIRED = {"baseline_v6", "no_rainfall_trigger", "no_urban_spectral", "no_topography_hydrology",
            "no_evidence_support", "equal_group_weights", "hydrology_heavy", "urban_heavy",
            "rainfall_heavy", "strict_high_threshold", "loose_high_threshold"}


def _csv(p):
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_artifacts_exist():
    for p in (SCEN, PATCH, SUMMARY, REPORT):
        assert p.exists(), p


def test_all_scenarios_present():
    assert {r["scenario"] for r in _csv(SCEN)} >= REQUIRED


def test_baseline_self_spearman_one():
    base = next(r for r in _csv(SCEN) if r["scenario"] == "baseline_v6")
    assert abs(float(base["spearman_vs_baseline"]) - 1.0) < 1e-6


def test_by_patch_300():
    assert len(_csv(PATCH)) == 300


def test_stability_flags_valid():
    for r in _csv(PATCH):
        assert r["stability_flag"] in {"robust", "moderate", "unstable"}


def test_governance():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["can_be_used_as_ground_truth"] is False
    assert s["allowed_for_training"] is False
    assert s["trained_model"] is False
    assert s["model_persisted"] is False


def test_report_mandatory():
    assert "não constitui treinamento" in REPORT.read_text(encoding="utf-8")
