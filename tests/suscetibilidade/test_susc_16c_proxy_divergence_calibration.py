"""SUSC-16C proxy divergence and calibration review-only tests."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"

UNIFIED = DAT / "susc_16c_unified_observational_analysis_table_v1.csv"
CASE_AUDIT = OUT / "SUSC_16C_individual_case_audit.csv"
CASE_DIR = OUT / "SUSC_16C_individual_cases"
WEIGHT_SUMMARY = OUT / "SUSC_16C_weight_sensitivity_summary.json"
REPORT = OUT / "SUSC_16C_proxy_divergence_and_calibration_review_report.md"
MANDATORY = (
    "O SUSC-16C transforma a divergencia entre score v6 e footprints "
    "observacionais em diagnostico auditavel de calibracao. A etapa permanece "
    "review-only, nao cria ground truth, nao libera treino supervisionado, nao "
    "altera o score v6 oficial e nao cria score v7 automatico."
)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(name: str):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )


def test_pipeline_and_validator_pass():
    for name in (
        "build_susc_16c_unified_observational_analysis_table.py",
        "decompose_susc_16c_score_v6_components.py",
        "build_susc_16c_individual_footprint_patch_case_files.py",
        "run_susc_16c_feature_direction_stability.py",
        "run_susc_16c_weight_sensitivity_review_only.py",
        "classify_susc_16c_proxy_failure_modes.py",
        "build_susc_16c_calibration_design_matrix.py",
        "validate_susc_16c_proxy_divergence_calibration.py",
    ):
        result = run_script(name)
        assert result.returncode == 0, name + "\n" + result.stdout + result.stderr


def test_unified_table_governance_and_counts():
    rows = read_csv(UNIFIED)
    assert len(rows) == 103
    event_rows = [row for row in rows if row["is_event_patch"] == "true"]
    assert len(event_rows) == 65
    assert len({row["patch_id"] for row in event_rows}) == 62
    assert all(row["allowed_for_training"] == "false" for row in rows)
    assert all(row["can_be_ground_truth"] == "false" for row in rows)
    assert all(row["can_be_used_as_ground_truth"] == "false" for row in rows)
    assert all(row["review_only"] == "true" for row in rows)
    assert all(row["score_v7_created"] == "false" for row in rows)


def test_case_files_and_controls_are_not_ground_truth():
    case_rows = read_csv(CASE_AUDIT)
    assert len(case_rows) == 65
    assert list(CASE_DIR.glob("S16C_CASE_*.md"))
    controls = [row for row in read_csv(UNIFIED) if row["is_control_patch"] == "true"]
    assert controls
    assert all(row["evidence_strength"] != "negative" for row in controls)
    assert not (DAT / "susc_score_v7_candidate_by_patch_v1.csv").exists()


def test_weight_sensitivity_does_not_create_score_v7():
    summary = json.loads(WEIGHT_SUMMARY.read_text(encoding="utf-8"))
    assert summary["score_v7_created"] is False
    assert summary["susc_score_v6_candidate_modified"] is False
    assert summary["scenario_count"] == 8


def test_report_mandatory_statement():
    text = REPORT.read_text(encoding="utf-8")
    assert MANDATORY in text
    assert "nao cria score v7" in text
