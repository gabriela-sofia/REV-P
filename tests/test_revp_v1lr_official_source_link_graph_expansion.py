"""Tests for v1lr official source link graph expansion."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1lr_official_source_link_graph_expansion.py"
REG = ROOT / "datasets/official_source_link_graph_registry.csv"
ATTACH = ROOT / "datasets/official_source_deep_attachment_registry.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr + result.stdout
    assert REG.exists() and ATTACH.exists()


def test_link_graph_is_official_and_metadata_only() -> None:
    data = rows(REG)
    assert data
    assert all(r["private_path_removed"] == "true" for r in data)
    assert all("http" not in r["url_hash"].lower() for r in data)
    assert all(r["domain"] == "none" or r["domain"].endswith((".gov.br", ".rj.gov.br")) or "sgb.gov.br" in r["domain"] for r in data)


def test_public_outputs_have_no_private_paths_or_raw_names() -> None:
    low = (REG.read_text(encoding="utf-8") + ATTACH.read_text(encoding="utf-8")).lower()
    assert "c:\\" not in low and "c:/" not in low and "gabriela" not in low
    assert ".pdf" not in low and ".zip" not in low and ".shp" not in low and ".npy" not in low and ".npz" not in low
