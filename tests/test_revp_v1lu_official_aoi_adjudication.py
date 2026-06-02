"""Tests for v1lu official AOI adjudication."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1lu_official_aoi_adjudication.py"
MATRIX = ROOT / "datasets/official_aoi_adjudication_matrix.csv"
REG = ROOT / "datasets/official_aoi_candidate_registry.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert MATRIX.exists() and REG.exists()


def test_aoi_not_promoted_from_bounds_or_generic_polygons_only() -> None:
    matrix = rows(MATRIX)[0]
    if matrix["official_aoi_gate"] != "PASS":
        assert matrix["decision"] != "OFFICIAL_AOI_CONFIRMED"
    for row in rows(REG):
        if "poligonos demarcados" in row["phrase"].lower() or "polígonos demarcados" in row["phrase"].lower():
            assert row["supports_official_aoi_gate"] == "false"


def test_public_outputs_have_no_private_paths() -> None:
    low = (MATRIX.read_text(encoding="utf-8") + REG.read_text(encoding="utf-8")).lower()
    assert "c:\\" not in low and "c:/" not in low and "gabriela" not in low
