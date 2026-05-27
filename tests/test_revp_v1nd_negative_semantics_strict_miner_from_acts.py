"""Tests for v1nd strict negative semantics from gazette acts."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
sys.path.insert(0, str(ROOT / "scripts/protocolo_c"))
from revp_v1na_v1nh_common import contains_release_context  # noqa: E402

DEP = ROOT / "scripts/protocolo_c/revp_v1nc_administrative_act_segmenter.py"
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1nd_negative_semantics_strict_miner_from_acts.py"
CAND = ROOT / "datasets/gazette_negative_semantics_candidate_registry.csv"
REJ = ROOT / "datasets/gazette_negative_semantics_rejection_registry.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    subprocess.run([sys.executable, str(DEP), "--force", "--emit-evidence"], cwd=ROOT, check=True, timeout=180)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr + result.stdout
    assert CAND.exists() and REJ.exists()


def test_desinterdicao_or_release_without_explicit_no_risk_is_not_formal() -> None:
    for row in rows(REJ):
        if row["decision"] == "ADMINISTRATIVE_RELEASE_REVIEW_ONLY":
            assert row["explicit_negative_statement_gate"] == "FAIL"


def test_baixo_risco_and_deliberados_do_not_create_formal_candidate() -> None:
    assert not contains_release_context("onde serao deliberados os seguintes assuntos")
    text = (CAND.read_text(encoding="utf-8") + REJ.read_text(encoding="utf-8")).lower()
    assert "baixo risco,pass" not in text
