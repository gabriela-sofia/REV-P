"""Tests for v1kv real control patch acquisition."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1kv_real_control_patch_acquisition_minimum_batch.py"
OUT = ROOT / "datasets/real_control_patch_acquisition_registry.csv"
QA = ROOT / "datasets/real_control_patch_qa_matrix.csv"
PROBE = ROOT / "datasets/control_gee_extractability_probe_registry.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_outputs_exist_or_run_existing() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists()
    assert QA.exists()


def test_real_patch_qa_requires_s2_pre_post_and_dem() -> None:
    qa = rows(QA)
    passed = [r for r in qa if r["real_patch_qa_status"] == "QA_PASS"]
    assert len(passed) >= 9
    assert all(r["s2_pre_qa_gate"] == "PASS" and r["s2_post_qa_gate"] == "PASS" and r["dem_qa_gate"] == "PASS" for r in passed)
    assert all(r["can_create_operational_label"] == "false" for r in qa)
    assert all(r["can_train_model"] == "false" for r in qa)


def test_acquisition_only_uses_probe_pass_controls() -> None:
    pass_ids = {r["control_candidate_id"] for r in rows(PROBE) if r["extractability_status"] == "PASS"}
    acquired = {r["control_candidate_id"] for r in rows(QA) if r["real_patch_qa_status"] == "QA_PASS"}
    assert acquired <= pass_ids
