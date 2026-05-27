"""Tests for v1ng C4 recheck after gazette negatives."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1ng_c4_recheck_after_gazette_negatives.py"
DEP = ROOT / "scripts/protocolo_c/revp_v1nf_formal_negative_from_gazette_adjudication.py"
C4 = ROOT / "datasets/c4_recheck_after_gazette_negatives.csv"
READY = ROOT / "datasets/c4_label_readiness_after_gazette_negatives.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    subprocess.run([sys.executable, str(DEP), "--force", "--emit-evidence"], cwd=ROOT, check=True, timeout=120)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert C4.exists() and READY.exists()


def test_c4_only_opens_with_formal_negative() -> None:
    row = rows(C4)[0]
    if row["formal_negative_count"] == "0":
        assert row["decision"] == "C4_BLOCKED_NO_FORMAL_NEGATIVES"
        assert row["can_create_operational_label"] == "false"
    assert row["can_train_model"] == "false"


def test_readiness_keeps_training_false() -> None:
    assert rows(READY)[0]["can_train_model"] == "false"
