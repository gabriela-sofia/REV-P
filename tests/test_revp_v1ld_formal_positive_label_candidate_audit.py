"""Tests for v1ld formal positive candidate audit."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1ld_formal_positive_label_candidate_audit.py"
OUT = ROOT / "datasets/formal_positive_label_candidate_registry.csv"
GATES = ROOT / "datasets/formal_positive_patch_gate_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-positive"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists() and GATES.exists()


def test_nine_positive_patch_candidates_exist_but_do_not_open_training() -> None:
    positives = [r for r in rows(OUT) if r["decision"] == "FORMAL_POSITIVE_PATCH_CANDIDATE"]
    assert len(positives) == 9
    assert all(r["can_create_operational_label"] == "false" for r in positives)
    assert all(r["can_train_model"] == "false" for r in positives)
