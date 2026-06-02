"""Tests for v1kz numeric blocker final report."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1kz_numeric_blocker_final_report.py"
OUT = ROOT / "datasets/control_numeric_resolution_summary_v1kz.csv"
DOCS = [
    ROOT / "docs/metodologia_cientifica/protocolo_c_resolucao_features_controles_v1ks_v1kz.md",
    ROOT / "docs/metodologia_cientifica/protocolo_c_relatorio_resolucao_features_controles_v1ks_v1kz.md",
]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-report"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists()
    assert all(path.exists() for path in DOCS)


def test_summary_resolves_numeric_blocker_without_operational_c4() -> None:
    summary = rows(OUT)[0]
    assert int(summary["valid_coordinate_count"]) == 50
    assert int(summary["gee_extractable_count"]) >= 9
    assert int(summary["real_patch_qa_pass_count"]) >= 9
    assert int(summary["numeric_feature_pass_count"]) >= 9
    assert summary["control_experiment_ready"] == "true"
    assert summary["c4_operational_status"] == "BLOCKED"
    assert summary["can_create_operational_label"] == "false"
    assert summary["can_train_model"] == "false"


def test_public_outputs_have_no_private_paths_or_heavy_artifacts() -> None:
    text = OUT.read_text(encoding="utf-8") + "\n".join(path.read_text(encoding="utf-8") for path in DOCS)
    lowered = text.lower()
    assert "c:\\" not in lowered and "c:/" not in lowered
    assert "gabriela" not in lowered
    assert ".tif" not in lowered and ".npy" not in lowered and ".npz" not in lowered
