"""Tests for v1ku GEE extractability probe."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1ku_gee_control_patch_extractability_probe.py"
OUT = ROOT / "datasets/control_gee_extractability_probe_registry.csv"
FAIL = ROOT / "datasets/control_extractability_failure_reason_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_outputs_exist_or_run_existing() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists()
    assert FAIL.exists()


def test_extractability_requires_gee_s2_pre_post_and_dem() -> None:
    probe = rows(OUT)
    assert probe
    for row in probe:
        if row["extractability_status"] == "PASS":
            assert row["gee_auth_ok"] == "true"
            assert row["s2_pre_available"] == "true"
            assert row["s2_post_available"] == "true"
            assert row["dem_available"] == "true"
            assert row["patch_size_ok"] == "true"
        assert row["can_create_operational_label"] == "false"
        assert row["can_train_model"] == "false"
