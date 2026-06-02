"""Tests for v1kh positive vs stable-control sandbox probe."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPTS = ROOT / "scripts/protocolo_c"
OUT = ROOT / "datasets/positive_stable_control_sandbox_probe_registry.csv"
READY = ROOT / "datasets/positive_stable_control_feature_readiness_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    for name, args in [
        ("revp_v1kf_conservative_stable_control_sampler.py", ["--force", "--emit-controls"]),
        ("revp_v1kg_stable_control_sensitivity_audit.py", ["--force", "--emit-sensitivity"]),
        ("revp_v1kh_positive_stable_control_sandbox_probe.py", ["--force", "--emit-probe"]),
    ]:
        result = subprocess.run([sys.executable, str(SCRIPTS / name), *args], cwd=ROOT, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists()
    assert READY.exists()


def test_no_operational_training_or_detection_claim() -> None:
    row = rows(OUT)[0]
    assert row["can_create_operational_label"] == "false"
    assert row["can_train_operational_model"] == "false"
    assert row["can_claim_detection"] == "false"
    assert row["dino_status"] == "FROZEN"


def test_feature_table_insufficient_without_numeric_control_features() -> None:
    row = rows(READY)[0]
    assert row["feature_table_sufficient_for_sandbox"] == "false"
    assert row["blocking_reason"] == "NO_NUMERIC_STABLE_CONTROL_S2_DEM_FEATURES"


def test_no_weights_or_heavy_artifacts_in_public_output() -> None:
    text = OUT.read_text(encoding="utf-8") + READY.read_text(encoding="utf-8")
    for token in [".tif", ".npy", ".npz", "model.pkl", "weights"]:
        assert token not in text.lower()
