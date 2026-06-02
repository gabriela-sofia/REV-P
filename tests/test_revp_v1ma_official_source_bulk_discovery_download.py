"""Tests for v1ma bulk official source discovery/download."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1ma_official_source_bulk_discovery_download.py"
REG = ROOT / "datasets/official_bulk_source_discovery_registry.csv"
MANIFEST = ROOT / "datasets/official_bulk_download_manifest.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stderr + result.stdout
    assert REG.exists() and MANIFEST.exists()


def test_bulk_manifest_is_public_metadata_only() -> None:
    data = rows(MANIFEST)
    assert data
    assert all(r["raw_storage_policy"] == "RAW_ONLY_LOCAL_RUNS" for r in data)
    assert all(r["private_path_removed"] == "true" for r in data)
    assert any(r["acquisition_status"] == "DOWNLOAD_OK" for r in data)


def test_public_outputs_have_no_private_paths_or_raw_extensions() -> None:
    low = (REG.read_text(encoding="utf-8") + MANIFEST.read_text(encoding="utf-8")).lower()
    assert "c:\\" not in low and "c:/" not in low and "gabriela" not in low
    assert ".pdf" not in low and ".zip" not in low and ".shp" not in low and ".npy" not in low and ".npz" not in low
