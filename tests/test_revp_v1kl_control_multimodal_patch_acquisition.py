"""Tests for v1kl control multimodal patch acquisition."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPTS = ROOT / "scripts/protocolo_c"
OUT = ROOT / "datasets/control_multimodal_patch_registry.csv"
QA = ROOT / "datasets/control_patch_qa_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_chain() -> None:
    chain = [
        ("revp_v1kk_control_candidate_pool_expansion.py", ["--force", "--limit", "50", "--emit-pool"]),
        ("revp_v1kl_control_multimodal_patch_acquisition.py", ["--force", "--emit-patch-qa"]),
    ]
    for name, args in chain:
        result = subprocess.run([sys.executable, str(SCRIPTS / name), *args], cwd=ROOT, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stderr + result.stdout


def test_script_runs_with_force() -> None:
    run_chain()
    assert OUT.exists()
    assert QA.exists()


def test_dino_is_frozen_and_no_operational_label_is_created() -> None:
    qa = rows(QA)
    assert qa
    assert all(r["dino_frozen_gate"] == "PASS" for r in qa)
    assert all(r["can_create_operational_label"] == "false" for r in qa)
    assert all(r["can_train_model"] == "false" for r in qa)


def test_no_raw_public_path_or_heavy_artifact_reference() -> None:
    text = OUT.read_text(encoding="utf-8") + QA.read_text(encoding="utf-8")
    assert "C:\\" not in text and "C:/" not in text
    assert "gabriela" not in text.lower()
    assert ".npy" not in text.lower() and ".npz" not in text.lower()
