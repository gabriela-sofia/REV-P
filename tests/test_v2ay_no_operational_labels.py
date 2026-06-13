"""v2ay - no operational labels, models, training or automatic C4 tests."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_summary_blocks_model_and_operational_labels():
    summary = json.loads((ROOT / "outputs_public" / "execution_reports" /
                          "v2ay_event_scope_reconciliation_turning_point_summary.json").read_text(encoding="utf-8"))
    assert summary["can_train_model"] is False
    assert summary["can_create_operational_labels"] is False
    assert summary["methodological_status"] == (
        "EVENT_SCOPE_RECONCILED_GEOMETRY_ACQUISITION_REQUIRED_NOT_FOR_TRAINING")


def test_no_model_artifact_created_in_v2ay_scope():
    suffixes = {".pt", ".pth", ".onnx", ".h5", ".ckpt", ".pkl", ".joblib", ".safetensors"}
    for base in (ROOT / "datasets", ROOT / "outputs_public"):
        for path in base.rglob("*v2ay*"):
            if path.is_file():
                assert path.suffix.lower() not in suffixes
