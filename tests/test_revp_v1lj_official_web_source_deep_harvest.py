"""Tests for v1lj official web source harvest."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1lj_official_web_source_deep_harvest.py"
REG = ROOT / "datasets/official_web_ground_truth_source_harvest_registry.csv"
AUDIT = ROOT / "datasets/official_web_download_audit_registry.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr + result.stdout
    assert REG.exists() and AUDIT.exists()


def test_harvest_uses_official_domains_and_public_hashes_only() -> None:
    data = rows(REG)
    assert len(data) >= 10
    assert all(r["official_domain"].endswith((".gov.br", ".rj.gov.br")) or "sgb.gov.br" in r["official_domain"] for r in data)
    assert all(r["url_hash"] and "http" not in r["url_hash"].lower() for r in data)
    assert all(r["private_path_removed"] == "true" for r in data)


def test_public_outputs_have_no_private_paths_or_raw_filenames() -> None:
    text = REG.read_text(encoding="utf-8") + AUDIT.read_text(encoding="utf-8")
    low = text.lower()
    assert "c:\\" not in low and "c:/" not in low and "gabriela" not in low
    assert ".pdf" not in low and ".zip" not in low and ".shp" not in low and ".npy" not in low and ".npz" not in low
    assert all(r["can_train_model"] == "false" for r in rows(REG))
