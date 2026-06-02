"""Tests for v1lq official ground-truth final report."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1lq_ground_truth_breakthrough_or_final_blocker_report.py"
OUT = ROOT / "datasets/protocol_c_ground_truth_breakthrough_or_blocker_summary.csv"
DOCS = [
    ROOT / "docs/metodologia_cientifica/protocolo_c_busca_oficial_ground_truth_v1lj_v1lq.md",
    ROOT / "docs/metodologia_cientifica/protocolo_c_relatorio_busca_oficial_ground_truth_v1lj_v1lq.md",
]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-summary"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists()
    assert all(path.exists() for path in DOCS)


def test_summary_preserves_operational_boundaries() -> None:
    row = rows(OUT)[0]
    if row["c4_operational_status"] != "C4_OPERATIONAL_READY":
        assert row["can_create_operational_label"] == "false"
    assert row["can_train_model"] == "false"
    assert "stable" not in row["next_single_technical_action"].lower()


def test_public_outputs_have_no_private_paths_or_heavy_artifact_refs() -> None:
    text = OUT.read_text(encoding="utf-8") + "\n".join(path.read_text(encoding="utf-8") for path in DOCS)
    low = text.lower()
    assert "c:\\" not in low and "c:/" not in low and "gabriela" not in low
    assert ".pdf" not in low and ".zip" not in low and ".shp" not in low and ".npy" not in low and ".npz" not in low
