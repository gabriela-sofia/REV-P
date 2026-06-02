"""Tests for v1ll inventory completeness proof miner."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1ll_inventory_completeness_event_link_proof_miner.py"
MATRIX = ROOT / "datasets/inventory_event_link_proof_matrix.csv"
REG = ROOT / "datasets/inventory_completeness_proof_candidate_registry.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert MATRIX.exists() and REG.exists()


def test_negated_proof_phrase_is_not_complete_inventory_support() -> None:
    sys.path.insert(0, str(ROOT / "scripts/protocolo_c"))
    from revp_v1lj_v1lq_common import is_negated_context

    assert is_negated_context("inventario completo nao provado")


def test_complete_inventory_gate_requires_explicit_support() -> None:
    matrix = rows(MATRIX)[0]
    if matrix["complete_inventory_gate"] != "PASS":
        assert matrix["decision"] != "COMPLETE_EVENT_INVENTORY_CONFIRMED"
        assert matrix["can_create_operational_label"] == "false"
    assert matrix["can_train_model"] == "false"
