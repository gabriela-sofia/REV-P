"""
SUSC-06B tests — proxy baseline diagnostics, circularity, no event validation.

Assert the diagnostics exist, circularity is recorded, the target proxy is not
ground truth, R2 is framed as recoverability (not predictive validation),
divergences are present, governance is preserved, and no model is persisted.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())

SCHEMA = ROOT / "schemas" / "suscetibilidade" / "susc_06b_baseline_diagnostics_schema_v1.json"
OUT = ROOT / "outputs_public" / "suscetibilidade"
CIRC = OUT / "SUSC_06B_baseline_circularity_report.csv"
DIR_AUDIT = OUT / "SUSC_06B_partial_effect_direction_audit.csv"
REGION_DIAG = OUT / "SUSC_06B_region_stability_diagnostics.csv"
ACTION = OUT / "SUSC_06B_feature_action_matrix.csv"
OVERLAY = OUT / "SUSC_06B_event_overlay_readiness_matrix.csv"
SUMMARY = OUT / "SUSC_06B_diagnostic_summary.json"
REPORT = OUT / "SUSC_06B_proxy_baseline_diagnostics_report.md"

MODEL_EXTS = {".pt", ".pth", ".pkl", ".joblib", ".h5", ".onnx", ".ckpt", ".sav", ".model"}


@pytest.fixture(scope="module")
def summary():
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def _csv(p):
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_artifacts_exist():
    for p in (SCHEMA, CIRC, DIR_AUDIT, REGION_DIAG, ACTION, OVERLAY, SUMMARY, REPORT):
        assert p.exists(), f"missing: {p}"


def test_circularity_recorded(summary):
    assert summary["circularity_risk"] in {"high", "medium", "low"}
    assert summary["circularity_risk"] == "high"
    rows = _csv(CIRC)
    assert rows and rows[0]["circularity_risk"] == "high"


def test_target_proxy_not_ground_truth(summary):
    rows = _csv(CIRC)
    assert rows[0]["target_is_proxy"] == "true"
    assert rows[0]["target_is_ground_truth"] == "false"
    assert summary["can_be_used_as_ground_truth"] is False


def test_r2_not_predictive_validation(summary):
    assert summary["r2_interpretation"] == "heuristic_recoverability"
    assert summary["predictive_power_claim"] is False
    rows = _csv(CIRC)
    assert rows[0]["predictive_power_claim"].lower() == "false"


def test_divergences_present(summary):
    audited = {r["feature"] for r in _csv(DIR_AUDIT)}
    for feat in ("curvature_laplacian_mean", "rain_3d_7d_ratio", "flow_acc_log_p75"):
        assert feat in audited
    assert summary["n_divergent"] >= 3


def test_divergent_not_auto_advanced():
    flagged = {
        r["feature"] for r in _csv(DIR_AUDIT)
        if r["partial_effect_class"] in {"direction_diverges", "near_zero_effect"}
    }
    actions = {r["feature"]: r["feature_action"] for r in _csv(ACTION)}
    for feat in flagged:
        assert actions.get(feat) != "advance_to_score_v6_candidate", feat


def test_governance_preserved(summary):
    assert summary["allowed_for_training"] is False
    assert summary["can_be_used_as_ground_truth"] is False
    assert summary["review_only"] is True
    assert summary["model_persisted"] is False


def test_mandatory_statement_in_report():
    text = REPORT.read_text(encoding="utf-8")
    assert "não valida ocorrência real de enchente" in text


def test_no_persisted_model():
    found = []
    for base in (OUT, ROOT / "scripts" / "suscetibilidade", ROOT / "datasets" / "suscetibilidade"):
        if base.exists():
            found += [p.name for p in base.rglob("*") if p.suffix.lower() in MODEL_EXTS]
    assert not found, f"unexpected model artifact(s): {found}"


def test_overlay_readiness_gaps(summary):
    assert "future_documentary_evidence" in summary["overlay_gaps_for_susc07"]
