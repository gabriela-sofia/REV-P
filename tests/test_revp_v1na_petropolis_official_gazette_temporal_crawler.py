"""Tests for v1na Petrópolis official gazette temporal crawler."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1na_petropolis_official_gazette_temporal_crawler.py"
ISSUES = ROOT / "datasets/petropolis_official_gazette_issue_registry.csv"
MANIFEST = ROOT / "datasets/petropolis_official_gazette_download_manifest.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence", "--no-extended", "--max-issues", "2"], cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr + result.stdout
    assert ISSUES.exists() and MANIFEST.exists()


def test_listing_page_does_not_count_as_real_issue() -> None:
    for row in rows(ISSUES):
        if row["issue_kind"] in {"CATEGORY_INDEX", "INDEX_HTML", "NONE"}:
            assert row["is_real_issue"] == "false"
        if row["is_real_issue"] == "true":
            assert row["issue_kind"] == "ISSUE_DOWNLOAD_LINK"
            assert row["issue_date"].startswith("2022-")


def test_public_manifest_is_metadata_only() -> None:
    text = ISSUES.read_text(encoding="utf-8") + MANIFEST.read_text(encoding="utf-8")
    low = text.lower()
    assert "c:\\" not in low and "c:/" not in low and "gabriela" not in low
    assert all(r["raw_storage_policy"] == "RAW_ONLY_LOCAL_RUNS" for r in rows(MANIFEST))
