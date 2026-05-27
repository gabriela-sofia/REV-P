"""Tests for v1nb gazette full text extraction."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
DEP = ROOT / "scripts/protocolo_c/revp_v1na_petropolis_official_gazette_temporal_crawler.py"
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1nb_gazette_full_text_ocr_extraction.py"
DOCS = ROOT / "datasets/petropolis_gazette_text_extraction_registry.csv"
PAGES = ROOT / "datasets/petropolis_gazette_page_text_inventory.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    subprocess.run([sys.executable, str(DEP), "--force", "--emit-evidence", "--no-extended", "--max-issues", "2"], cwd=ROOT, check=True, timeout=180)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr + result.stdout
    assert DOCS.exists() and PAGES.exists()


def test_extraction_registers_method_or_ocr_blocker() -> None:
    for row in rows(DOCS):
        assert row["extraction_method"]
        assert row["ocr_status"] in {"OCR_NOT_NEEDED", "OCR_NOT_AVAILABLE_OR_NOT_RUN", "OCR_NOT_RUN"}


def test_public_page_inventory_has_no_private_paths() -> None:
    low = PAGES.read_text(encoding="utf-8").lower()
    assert "c:\\" not in low and "c:/" not in low and "gabriela" not in low
