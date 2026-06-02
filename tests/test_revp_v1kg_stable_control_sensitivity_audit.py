"""Tests for v1kg stable control sensitivity audit."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT_KF = ROOT / "scripts/protocolo_c/revp_v1kf_conservative_stable_control_sampler.py"
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1kg_stable_control_sensitivity_audit.py"
OUT_READY = ROOT / "datasets/hard_stable_control_readiness_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    subprocess.run([sys.executable, str(SCRIPT_KF), "--force", "--emit-controls"], cwd=ROOT, check=True, capture_output=True, text=True, timeout=120)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-sensitivity"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert OUT_READY.exists()


def test_hard_control_requires_numeric_spectral_and_terrain_gates() -> None:
    data = rows(OUT_READY)
    assert data
    assert all(row["can_be_formal_negative"] == "false" for row in data)
    hard = [row for row in data if row["can_be_hard_stable_control"] == "true"]
    assert hard == []
    assert all(row["can_train_operational_model"] == "false" for row in data)


def test_review_controls_remain_distinct_from_hard_controls() -> None:
    data = rows(OUT_READY)
    assert any(row["readiness_decision"] == "CONSERVATIVE_STABLE_CONTROL_REVIEW" for row in data)
    assert all(row["readiness_decision"] != "FORMAL_NEGATIVE" for row in data)
