"""Tests for v1mj C4 recheck after bulk acquisition."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1mj_c4_operational_recheck_after_bulk_acquisition.py"
OUT = ROOT / "datasets/c4_operational_recheck_after_bulk_acquisition.csv"
LABELS = ROOT / "datasets/c4_operational_label_readiness_after_bulk_acquisition.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists() and LABELS.exists()


def test_c4_only_opens_with_negative_patch_and_split() -> None:
    row = rows(OUT)[0]
    if row["formal_negative_count"] == "0":
        assert row["decision"] == "C4_BLOCKED_NO_FORMAL_NEGATIVES"
        assert row["can_create_operational_label"] == "false"
    if row["decision"] == "C4_OPERATIONAL_READY":
        assert row["patch_qa_gate"] == "PASS" and row["split_leakage_gate"] == "PASS"


def test_can_train_model_false() -> None:
    assert rows(OUT)[0]["can_train_model"] == "false"
