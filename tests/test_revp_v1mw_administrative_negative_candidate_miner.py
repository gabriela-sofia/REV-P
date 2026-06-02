"""Tests for v1mw administrative negative candidate mining."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1mw_administrative_negative_candidate_miner.py"
DEP = ROOT / "scripts/protocolo_c/revp_v1mv_administrative_text_table_extraction.py"
CAND = ROOT / "datasets/administrative_negative_candidate_registry.csv"
SEM = ROOT / "datasets/administrative_negative_semantics_audit.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    subprocess.run([sys.executable, str(DEP), "--force", "--emit-evidence"], cwd=ROOT, check=True, timeout=180)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert CAND.exists() and SEM.exists()


def test_desinterdicao_without_explicit_no_risk_not_formal() -> None:
    for row in rows(CAND):
        if row["administrative_act_type"] == "DESINTERDICAO" and row["explicit_negative_statement_gate"] != "PASS":
            assert row["decision"] == "ADMINISTRATIVE_RELEASE_REVIEW_ONLY"


def test_low_risk_and_absence_are_not_formal_negative() -> None:
    text = CAND.read_text(encoding="utf-8").lower() + SEM.read_text(encoding="utf-8").lower()
    assert "absence of record" not in text
    for row in rows(SEM):
        if "LOW_RISK" in row["semantic_boundary"]:
            assert row["formal_negative_allowed_without_geocoding"] == "false"
