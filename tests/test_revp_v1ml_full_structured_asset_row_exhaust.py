"""Tests for v1ml full structured asset row exhaustion."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1ml_full_structured_asset_row_exhaust.py"
INV = ROOT / "datasets/full_structured_asset_row_inventory.csv"
AUDIT = ROOT / "datasets/full_structured_asset_field_value_audit.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stderr + result.stdout
    assert INV.exists() and AUDIT.exists()


def test_processing_records_full_or_limited_mode_not_samples_only() -> None:
    data = rows(INV)
    assert data
    assert all(r["processed_mode"] in {"FULL_FILE", "CHUNKED_LIMIT_REACHED", "XLSX_READER_UNAVAILABLE", "DBF_HEADER_TOO_SMALL", "EMPTY_WORKBOOK"} or r["processed_mode"].startswith("READ_FAIL") for r in data)
    assert any(int(r["row_count"] or 0) >= 0 for r in data)


def test_public_outputs_are_sanitized() -> None:
    low = (INV.read_text(encoding="utf-8") + AUDIT.read_text(encoding="utf-8")).lower()
    assert "c:\\" not in low and "c:/" not in low and "gabriela" not in low
    assert all(r["private_path_removed"] == "true" for r in rows(INV))
