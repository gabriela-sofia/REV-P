"""Tests for v1mv administrative text/table extraction."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1mv_administrative_text_table_extraction.py"
DEP = ROOT / "scripts/protocolo_c/revp_v1mu_administrative_negative_source_discovery.py"
DOC = ROOT / "datasets/administrative_text_table_extraction_registry.csv"
PHRASE = ROOT / "datasets/administrative_phrase_context_registry.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    subprocess.run([sys.executable, str(DEP), "--force", "--emit-evidence"], cwd=ROOT, check=True, timeout=240)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr + result.stdout
    assert DOC.exists() and PHRASE.exists()


def test_phrase_registry_has_required_context_columns() -> None:
    row = rows(PHRASE)[0]
    for col in ["has_date", "has_location", "has_precise_address", "has_coordinate", "has_phenomenon"]:
        assert col in row


def test_public_outputs_have_no_private_paths() -> None:
    low = DOC.read_text(encoding="utf-8").lower() + PHRASE.read_text(encoding="utf-8").lower()
    assert "c:\\" not in low and "c:/" not in low and "gabriela" not in low
