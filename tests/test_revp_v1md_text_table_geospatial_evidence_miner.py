"""Tests for v1md official evidence miner."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1md_text_table_geospatial_evidence_miner.py"
TEXT = ROOT / "datasets/official_evidence_phrase_table_registry.csv"
STRUCT = ROOT / "datasets/official_structured_feature_evidence_registry.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stderr + result.stdout
    assert TEXT.exists() and STRUCT.exists()


def test_mandatory_query_log_is_not_mined_as_negative() -> None:
    low = TEXT.read_text(encoding="utf-8").lower()
    assert "petropolis 2022 sem ocorrencia deslizamento" not in low


def test_evidence_has_required_specificity_fields() -> None:
    row = rows(TEXT)[0]
    assert {"evidence_type", "strength", "spatial_specificity", "temporal_specificity"}.issubset(row)
