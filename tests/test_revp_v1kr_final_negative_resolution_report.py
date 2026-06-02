"""Tests for v1kr final negative/control resolution report."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPTS = ROOT / "scripts/protocolo_c"
OUT = ROOT / "datasets/protocol_c_negative_resolution_summary_v1kr.csv"
DOCS = [
    ROOT / "docs/metodologia_cientifica/protocolo_c_resolucao_final_negativos_controles_v1kk_v1kr.md",
    ROOT / "docs/metodologia_cientifica/protocolo_c_relatorio_resolucao_final_negativos_controles_v1kk_v1kr.md",
]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    chain = [
        ("revp_v1kk_control_candidate_pool_expansion.py", ["--force", "--limit", "50", "--emit-pool"]),
        ("revp_v1kl_control_multimodal_patch_acquisition.py", ["--force", "--emit-patch-qa"]),
        ("revp_v1km_positive_control_numeric_feature_table.py", ["--force", "--emit-feature-table"]),
        ("revp_v1kn_hard_stable_control_numeric_gate.py", ["--force", "--emit-gates"]),
        ("revp_v1ko_control_experiment_split_leakage_protocol.py", ["--force", "--emit-split"]),
        ("revp_v1kp_control_experiment_sandbox_probe.py", ["--force", "--emit-probe"]),
        ("revp_v1kq_operational_vs_control_resolution_decision.py", ["--force", "--emit-decision"]),
        ("revp_v1kr_final_negative_resolution_report.py", ["--force", "--emit-report"]),
    ]
    for name, args in chain:
        result = subprocess.run([sys.executable, str(SCRIPTS / name), *args], cwd=ROOT, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists()
    assert all(path.exists() for path in DOCS)


def test_summary_keeps_operational_boundaries_closed() -> None:
    summary = rows(OUT)[0]
    assert summary["formal_negative_count"] == "0"
    assert summary["can_create_operational_label"] == "false"
    assert summary["can_train_model"] == "false"


def test_docs_do_not_turn_controls_into_negatives() -> None:
    text = "\n".join(path.read_text(encoding="utf-8") for path in DOCS)
    assert "nao sao negativos formais" in text
    assert "ausencia de registro" in text


def test_public_outputs_have_no_private_paths_or_heavy_artifacts() -> None:
    text = OUT.read_text(encoding="utf-8") + "\n".join(path.read_text(encoding="utf-8") for path in DOCS)
    assert "C:\\" not in text and "C:/" not in text
    assert "gabriela" not in text.lower()
    lowered = text.lower()
    assert ".npy" not in lowered and ".npz" not in lowered and ".tif" not in lowered
