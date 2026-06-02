"""Tests for v1lt inventory completeness adjudication."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1lt_inventory_completeness_adjudication.py"
MATRIX = ROOT / "datasets/inventory_completeness_adjudication_matrix.csv"
REG = ROOT / "datasets/inventory_event_scope_adjudication_registry.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert MATRIX.exists() and REG.exists()


def test_complete_inventory_gate_requires_supporting_citation() -> None:
    matrix = rows(MATRIX)[0]
    if matrix["complete_inventory_gate"] != "PASS":
        assert matrix["decision"] != "COMPLETE_EVENT_INVENTORY_CONFIRMED"
        assert matrix["can_create_operational_label"] == "false"
    for row in rows(REG):
        if row["supports_gate"] == "true":
            assert row["phrase_id"] != "none" and row["document_id"] != "none" and row["page"] != ""


def test_can_train_model_false() -> None:
    assert rows(MATRIX)[0]["can_train_model"] == "false"
