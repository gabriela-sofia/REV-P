"""Tests for v1mu administrative negative source discovery."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1mu_administrative_negative_source_discovery.py"
SRC = ROOT / "datasets/administrative_negative_source_registry.csv"
DL = ROOT / "datasets/administrative_negative_download_manifest.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stderr + result.stdout
    assert SRC.exists() and DL.exists()


def test_download_manifest_is_metadata_only() -> None:
    assert all(r["raw_storage_policy"] == "RAW_ONLY_LOCAL_RUNS" for r in rows(DL))
    low = SRC.read_text(encoding="utf-8").lower() + DL.read_text(encoding="utf-8").lower()
    assert "c:\\" not in low and "c:/" not in low and "gabriela" not in low


def test_non_official_sources_cannot_be_c4_proof() -> None:
    for row in rows(SRC):
        if row["discovery_status"] == "SKIPPED_NON_OFFICIAL":
            assert row["candidate_relevance"] != "FORMAL_NEGATIVE_CANDIDATE"
