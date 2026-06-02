"""Tests for v1lp C4 real reevaluation after official harvest."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1lp_c4_real_reevaluation_after_official_harvest.py"
OUT = ROOT / "datasets/c4_real_reevaluation_after_official_harvest.csv"
LABELS = ROOT / "datasets/c4_real_label_readiness_after_harvest.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists() and LABELS.exists()


def test_c4_only_opens_with_formal_negative_and_gates() -> None:
    row = rows(OUT)[0]
    if row["formal_negative_count"] == "0":
        assert row["decision"] == "C4_BLOCKED_NO_FORMAL_NEGATIVES"
        assert row["can_create_operational_label"] == "false"
    assert row["can_train_model"] == "false"


def test_no_detection_or_prediction_claims_in_label_boundary() -> None:
    low = LABELS.read_text(encoding="utf-8").lower()
    assert "detection" not in low and "prediction" not in low
