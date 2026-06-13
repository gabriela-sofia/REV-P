"""v2ax - methodological guardrail tests."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_summary_blocks_training_and_operational_labels():
    summary = json.loads((ROOT / "outputs_public" / "execution_reports" /
                          "v2ax_recife_geometry_intake_pack_summary.json").read_text(encoding="utf-8"))
    assert summary["can_train_model"] is False
    assert summary["can_create_operational_labels"] is False
    assert summary["methodological_status"] == "RECIFE_GEOMETRY_INTAKE_PACK_READY_NOT_FOR_TRAINING"


def test_no_model_or_forbidden_label_artifact_created():
    model_suffixes = {".pt", ".pth", ".onnx", ".h5", ".ckpt", ".pkl", ".joblib", ".safetensors"}
    forbidden = ("C4_OPERATIONAL_LABEL", "TRAINING_LABEL", "GROUND_TRUTH_FINAL")
    paths = [
        ROOT / "datasets" / "manual_intake" / "recife_p1",
        ROOT / "datasets" / "v2ax_recife_manual_intake_manifest.csv",
        ROOT / "outputs_public" / "execution_reports" / "v2ax_recife_geometry_intake_pack_report.md",
    ]
    for base in paths:
        candidates = list(base.rglob("*")) if base.is_dir() else [base]
        for path in candidates:
            if not path.is_file():
                continue
            assert path.suffix.lower() not in model_suffixes
            text = path.read_text(encoding="utf-8")
            assert all(term not in text for term in forbidden)
