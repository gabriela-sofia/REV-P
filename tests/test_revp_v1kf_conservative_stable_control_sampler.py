"""Tests for v1kf conservative stable controls."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1kf_conservative_stable_control_sampler.py"
OUT_REG = ROOT / "datasets/conservative_stable_control_candidate_registry.csv"
OUT_GATE = ROOT / "datasets/conservative_stable_control_gate_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-controls"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert OUT_REG.exists()
    assert OUT_GATE.exists()


def test_conservative_stable_control_is_not_formal_negative() -> None:
    data = rows(OUT_REG)
    assert data
    assert all(row["control_class"] != "FORMAL_NEGATIVE" for row in data)
    assert all(row["is_formal_negative"] == "false" for row in data)
    assert all(row["can_create_operational_label"] == "false" for row in data)


def test_absence_of_registry_is_not_used_as_negative() -> None:
    text = OUT_REG.read_text(encoding="utf-8")
    assert "absence of registry is not used as a negative" in text
    assert "negative_label" not in text.lower()


def test_dino_frozen_and_local_runs_ignored() -> None:
    data = rows(OUT_REG)
    assert all(row["dino_status"] == "FROZEN_REVIEW_ONLY" for row in data)
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "local_runs" in gitignore
