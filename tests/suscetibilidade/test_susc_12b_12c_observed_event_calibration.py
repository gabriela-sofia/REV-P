"""SUSC-12B/12C observed-event calibration tests."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
VALIDATOR = ROOT / "scripts" / "suscetibilidade" / "validate_susc_12b_12c_observed_event_calibration.py"
DATASET = ROOT / "datasets" / "suscetibilidade" / "susc_12b_event_feature_contrast_dataset_v1.csv"
RECS = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12C_proxy_calibration_recommendations.csv"
CAL_SUMMARY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12C_proxy_calibration_summary.json"
NOT_READY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12C_score_v7_not_ready_report.md"
INTEGRATED = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12B_12C_observed_event_contrast_and_calibration_report.md"

MANDATORY = (
    "O SUSC-12B/12C compara métricas de suscetibilidade com evidências observacionais "
    "de alagamento/inundação para calibração review-only do proxy. A análise não cria "
    "ground truth, não treina modelo supervisionado e não transforma ausência de registro "
    "em negativo verdadeiro."
)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_validator_passes():
    result = subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_dataset_governance_and_controls():
    rows = read_csv(DATASET)
    assert rows
    assert all(r["can_be_ground_truth"] == "false" for r in rows)
    assert all(r["allowed_for_training"] == "false" for r in rows)
    assert all(r["review_only"] == "true" for r in rows)
    controls = [r for r in rows if r["contrast_group"] == "no_documented_event_control"]
    assert controls
    assert all("no_documented_event_control" in r["control_selection_reason"] for r in controls)


def test_context_not_promoted_to_observed():
    for row in read_csv(DATASET):
        if "risk_area_context" in row["evidence_levels"] or "administrative_record_only" in row["evidence_levels"]:
            assert row["contrast_group"] == "weak_context_patch"


def test_recommendations_review_only():
    rows = read_csv(RECS)
    assert rows
    assert {r["recommendation"] for r in rows} == {"needs_more_observed_events"}
    assert all(r["review_only"] == "true" for r in rows)


def test_score_v7_not_created_and_report_statement():
    import json

    summary = json.loads(CAL_SUMMARY.read_text(encoding="utf-8"))
    assert summary["score_v7_created"] is False
    assert NOT_READY.exists()
    assert not (ROOT / "datasets" / "suscetibilidade" / "susc_score_v7_candidate_by_patch_v1.csv").exists()
    assert MANDATORY in INTEGRATED.read_text(encoding="utf-8")
