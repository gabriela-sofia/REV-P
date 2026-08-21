"""SUSC-10B score comparison tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
OUT = ROOT / "outputs_public" / "suscetibilidade"
RANK = OUT / "SUSC_10B_score_rank_shift_by_patch.csv"
METRICS = OUT / "SUSC_10B_score_comparison_metrics.csv"
SUMMARY = OUT / "SUSC_10B_score_comparison_summary.json"
REPORT = OUT / "SUSC_10B_score_comparison_report.md"
HIGHSPATIAL = OUT / "SUSC_10B_high_score_spatial_evidence_cases.csv"


def _csv(p):
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_artifacts_exist():
    for p in (RANK, METRICS, SUMMARY, REPORT, HIGHSPATIAL):
        assert p.exists(), p


def test_rank_has_300_patches():
    assert len(_csv(RANK)) == 300


def test_score_v6_in_unit_interval():
    for r in _csv(RANK):
        assert 0.0 <= float(r["score_v6"]) <= 1.0


def test_summary_governance():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["can_be_used_as_ground_truth"] is False
    assert s["allowed_for_training"] is False
    assert s["review_only"] is True
    assert s["uses_proxy_v5_as_score_feature"] is False
    assert s["interpretation_guard"] == "correlation_is_not_real_event_validation"


def test_report_mandatory_statement():
    assert "não constitui" in REPORT.read_text(encoding="utf-8")


def test_metrics_have_correlations():
    metrics = {r["metric"] for r in _csv(METRICS)}
    assert "pearson_v6_v5" in metrics
    assert "spearman_v6_v5" in metrics
