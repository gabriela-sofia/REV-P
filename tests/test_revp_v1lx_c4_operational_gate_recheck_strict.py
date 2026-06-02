"""Tests for v1lx strict C4 recheck."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1lx_c4_operational_gate_recheck_strict.py"
OUT = ROOT / "datasets/c4_operational_gate_recheck_strict.csv"
LABELS = ROOT / "datasets/c4_operational_label_readiness_strict.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists() and LABELS.exists()


def test_c4_only_opens_with_formal_negative() -> None:
    row = rows(OUT)[0]
    if row["formal_negative_count"] == "0":
        assert row["decision"] == "C4_BLOCKED_NO_FORMAL_NEGATIVES"
        assert row["can_create_operational_label"] == "false"
    assert row["can_train_model"] == "false"


def test_benchmark_not_used_as_ground_truth() -> None:
    low = LABELS.read_text(encoding="utf-8").lower()
    assert "benchmark" not in low
