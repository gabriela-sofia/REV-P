"""Tests for v1li ground truth real resolution summary."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1li_ground_truth_real_resolution_summary.py"
OUT = ROOT / "datasets/protocol_c_ground_truth_real_resolution_summary.csv"
DOCS = [
    ROOT / "docs/metodologia_cientifica/protocolo_c_resolucao_ground_truth_real_v1la_v1li.md",
    ROOT / "docs/metodologia_cientifica/protocolo_c_relatorio_resolucao_ground_truth_real_v1la_v1li.md",
]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-summary"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists()
    assert all(path.exists() for path in DOCS)


def test_summary_keeps_c4_blocked_without_formal_negatives() -> None:
    summary = rows(OUT)[0]
    assert summary["formal_positive_candidate_count"] == "9"
    assert summary["formal_negative_inventory_count"] == "0"
    assert summary["formal_negative_no_occurrence_count"] == "0"
    assert summary["c4_operational_status"] == "C4_BLOCKED_NO_FORMAL_NEGATIVES"
    assert summary["can_create_operational_label"] == "false"
    assert summary["can_train_model"] == "false"


def test_public_outputs_have_no_private_paths_or_heavy_artifact_refs() -> None:
    text = OUT.read_text(encoding="utf-8") + "\n".join(path.read_text(encoding="utf-8") for path in DOCS)
    lowered = text.lower()
    assert "c:\\" not in lowered and "c:/" not in lowered
    assert "gabriela" not in lowered
    assert ".tif" not in lowered and ".npy" not in lowered and ".npz" not in lowered and ".pdf" not in lowered and ".shp" not in lowered
