"""Tests for v1lb inventory completeness evidence recovery."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1lb_official_inventory_completeness_evidence_recovery.py"
OUT = ROOT / "datasets/official_inventory_completeness_evidence_registry.csv"
PHRASES = ROOT / "datasets/official_inventory_completeness_phrase_audit.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists() and PHRASES.exists()


def test_no_negated_completeness_statement_is_promoted() -> None:
    assert sum(1 for r in rows(OUT) if r["can_support_complete_inventory_gate"] == "true") == 0


def test_public_outputs_have_no_private_paths() -> None:
    text = OUT.read_text(encoding="utf-8") + PHRASES.read_text(encoding="utf-8")
    assert "C:\\" not in text and "C:/" not in text
    assert "gabriela" not in text.lower()
