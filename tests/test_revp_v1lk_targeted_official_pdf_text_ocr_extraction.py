"""Tests for v1lk official document text extraction."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1lk_targeted_official_pdf_text_ocr_extraction.py"
REG = ROOT / "datasets/official_document_text_extraction_registry.csv"
OCR = ROOT / "datasets/official_document_ocr_need_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr + result.stdout
    assert REG.exists() and OCR.exists()


def test_text_extraction_registry_is_metadata_only() -> None:
    data = rows(REG)
    assert data
    assert all(r["private_path_removed"] == "true" for r in data)
    assert all(r["can_create_operational_label"] == "false" and r["can_train_model"] == "false" for r in data)
    assert {"has_date_terms", "has_completeness_terms", "has_no_occurrence_terms"}.issubset(data[0])


def test_public_outputs_have_no_private_paths() -> None:
    text = REG.read_text(encoding="utf-8") + OCR.read_text(encoding="utf-8")
    low = text.lower()
    assert "c:\\" not in low and "c:/" not in low and "gabriela" not in low
    assert ".shp" not in low and ".tif" not in low and ".npy" not in low and ".npz" not in low
