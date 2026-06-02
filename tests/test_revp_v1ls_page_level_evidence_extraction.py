"""Tests for v1ls page-level evidence extraction."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1ls_page_level_evidence_extraction.py"
DOCS = ROOT / "datasets/official_page_level_evidence_registry.csv"
PHRASES = ROOT / "datasets/official_evidence_phrase_context_registry.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr + result.stdout
    assert DOCS.exists() and PHRASES.exists()


def test_evidence_has_document_page_phrase_identity() -> None:
    phrase_rows = rows(PHRASES)
    assert phrase_rows
    for row in phrase_rows:
        assert row["phrase_id"]
        assert row["document_id"]
        assert row["page"] != ""
        assert row["official_source_gate"] in {"PASS", "FAIL"}


def test_negated_or_interface_completeness_is_not_gate_support() -> None:
    for row in rows(PHRASES):
        if "mapeamento completo" in row["phrase"].lower() and ("ser" in row["phrase"].lower() or "notifications" in row["phrase"].lower()):
            assert row["polarity"] != "SUPPORTS_GATE"
