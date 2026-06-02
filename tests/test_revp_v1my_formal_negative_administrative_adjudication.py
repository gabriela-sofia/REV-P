"""Tests for v1my formal administrative negative adjudication."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1my_formal_negative_administrative_adjudication.py"
DEP = ROOT / "scripts/protocolo_c/revp_v1mx_geocode_administrative_negative_candidates.py"
REG = ROOT / "datasets/formal_negative_administrative_candidate_registry.csv"
MATRIX = ROOT / "datasets/formal_negative_administrative_gate_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    subprocess.run([sys.executable, str(DEP), "--force", "--emit-evidence"], cwd=ROOT, check=True, timeout=180)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert REG.exists() and MATRIX.exists()


def test_formal_negative_requires_all_hard_gates() -> None:
    for row in rows(REG):
        if row["decision"] == "FORMAL_NEGATIVE_CANDIDATE":
            for gate in ["official_source_gate", "administrative_document_gate", "explicit_negative_statement_gate", "phenomenon_specific_gate", "date_gate", "location_gate", "coordinate_or_precise_address_gate", "positive_buffer_exclusion_gate", "patch_extractability_gate", "leakage_precheck_gate"]:
                assert row[gate] == "PASS"


def test_no_training_or_operational_label_here() -> None:
    assert rows(MATRIX)[0]["can_train_model"] == "false"
    assert rows(MATRIX)[0]["can_create_operational_label"] == "false"
