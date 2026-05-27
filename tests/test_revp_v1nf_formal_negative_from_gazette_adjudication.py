"""Tests for v1nf formal negative adjudication from gazette acts."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1nf_formal_negative_from_gazette_adjudication.py"
DEP = ROOT / "scripts/protocolo_c/revp_v1ne_address_locality_extraction_geocoding_acts.py"
REG = ROOT / "datasets/formal_negative_gazette_candidate_registry.csv"
MATRIX = ROOT / "datasets/formal_negative_gazette_gate_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    subprocess.run([sys.executable, str(DEP), "--force", "--emit-evidence"], cwd=ROOT, check=True, timeout=120)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert REG.exists() and MATRIX.exists()


def test_formal_negative_requires_all_gates() -> None:
    hard_gates = ["official_gazette_gate", "administrative_act_gate", "explicit_negative_statement_gate", "phenomenon_specific_gate", "date_gate", "precise_location_gate", "coordinate_or_geocodable_address_gate", "positive_buffer_exclusion_gate", "patch_extractability_gate", "leakage_precheck_gate"]
    for row in rows(REG):
        if row["decision"] == "FORMAL_NEGATIVE_CANDIDATE":
            assert all(row[gate] == "PASS" for gate in hard_gates)


def test_no_training_or_operational_label_here() -> None:
    row = rows(MATRIX)[0]
    assert row["can_train_model"] == "false"
    assert row["can_create_operational_label"] == "false"
