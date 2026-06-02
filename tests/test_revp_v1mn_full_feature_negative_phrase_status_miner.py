"""Tests for v1mn feature-level negative phrase/status mining."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1mn_full_feature_negative_phrase_status_miner.py"
DEP1 = ROOT / "scripts/protocolo_c/revp_v1ml_full_structured_asset_row_exhaust.py"
DEP2 = ROOT / "scripts/protocolo_c/revp_v1mm_arcgis_full_feature_pagination_domain_decode.py"
CAND = ROOT / "datasets/full_feature_negative_phrase_candidate_registry.csv"
SEM = ROOT / "datasets/full_feature_status_semantics_registry.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    subprocess.run([sys.executable, str(DEP1), "--force", "--emit-evidence"], cwd=ROOT, check=True, timeout=240)
    subprocess.run([sys.executable, str(DEP2), "--force", "--emit-evidence"], cwd=ROOT, check=True, timeout=180)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert CAND.exists() and SEM.exists()


def test_low_risk_is_not_formal_negative_semantics() -> None:
    for row in rows(SEM):
        if "LOW_RISK" in row["semantic_class"]:
            assert row["formal_negative_allowed"] == "false"


def test_absence_of_feature_is_not_candidate() -> None:
    low = CAND.read_text(encoding="utf-8").lower()
    assert "absence of feature" not in low
    assert "stable_control" not in low
