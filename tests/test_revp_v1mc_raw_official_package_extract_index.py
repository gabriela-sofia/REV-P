"""Tests for v1mc raw official package extraction/indexing."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1mc_raw_official_package_extract_index.py"
REG = ROOT / "datasets/official_raw_package_index_registry.csv"
ASSETS = ROOT / "datasets/official_extracted_asset_inventory.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stderr + result.stdout
    assert REG.exists() and ASSETS.exists()


def test_raw_assets_are_indexed_sanitized() -> None:
    assert rows(ASSETS)
    assert all(r["private_path_removed"] == "true" for r in rows(ASSETS))
    assert any(r["asset_kind"] in {"PDF", "SHP", "DBF", "PRJ", "GEOJSON", "JSON", "HTML"} for r in rows(ASSETS))


def test_public_outputs_have_no_private_paths() -> None:
    low = (REG.read_text(encoding="utf-8") + ASSETS.read_text(encoding="utf-8")).lower()
    assert "c:\\" not in low and "c:/" not in low and "gabriela" not in low
