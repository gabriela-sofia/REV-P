"""SUSC-12A temporal evaluation tests."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = ROOT / "scripts" / "suscetibilidade" / "validate_susc_12a_temporal_evaluation.py"
DATASET = ROOT / "datasets" / "suscetibilidade" / "susc_12a_pre_event_evaluation_dataset_v1.csv"
HIT_RATE = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12A_temporal_hit_rate_topk.csv"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12A_temporal_pre_event_evaluation_report.md"

MANDATORY = (
    "O SUSC-12A avalia se patches com evidência observada apresentavam maior suscetibilidade "
    "antes do evento, quando a temporalidade dos dados permite. A análise é review-only, não "
    "cria ground truth, não treina modelo supervisionado e não constitui predição operacional."
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


def test_dataset_governance_closed():
    rows = read_csv(DATASET)
    assert rows
    assert all(r["can_be_ground_truth"] == "false" for r in rows)
    assert all(r["allowed_for_training"] == "false" for r in rows)
    assert all(r["review_only"] == "true" for r in rows)


def test_post_event_risk_not_pre_event_valid():
    for row in read_csv(DATASET):
        if row["feature_temporal_status"] == "post_event_risk":
            assert row["pre_event_valid"] == "false"
            assert row["temporal_leakage_risk"] == "high"


def test_controls_are_no_documented_event_controls():
    controls = [r for r in read_csv(DATASET) if r["row_role"] == "control_patch"]
    assert controls
    assert all(r["control_patch_status"] == "no_documented_event_control" for r in controls)
    text = REPORT.read_text(encoding="utf-8").lower()
    assert ("negativo " + "verdadeiro") not in text


def test_hit_rate_and_report_statement():
    assert {r["top_k"] for r in read_csv(HIT_RATE)} == {"top_10", "top_20", "top_30"}
    assert MANDATORY in REPORT.read_text(encoding="utf-8")
