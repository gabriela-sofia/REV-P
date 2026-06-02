"""Tests for v1mh patch readiness for new formal candidates."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1mh_patch_extraction_new_formal_candidates.py"
REG = ROOT / "datasets/new_formal_candidate_patch_readiness_registry.csv"
MATRIX = ROOT / "datasets/new_formal_candidate_multimodal_qa_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert REG.exists() and MATRIX.exists()


def test_dino_frozen_and_no_training() -> None:
    assert all(r["dino_frozen_gate"] == "PASS" for r in rows(REG))
    assert rows(MATRIX)[0]["can_train_model"] == "false"


def test_no_operational_label_from_patch_readiness_stage() -> None:
    assert rows(MATRIX)[0]["can_create_operational_label"] == "false"
