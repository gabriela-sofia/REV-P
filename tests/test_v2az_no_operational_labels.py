"""v2az - methodological safety tests."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_summary_blocks_training_and_operational_labels():
    summary = json.loads((ROOT / "outputs_public" / "execution_reports" /
                          "v2az_turning_point_replay_orchestrator_summary.json").read_text(encoding="utf-8"))
    assert summary["can_train_model"] is False
    assert summary["can_create_operational_labels"] is False
    assert summary["methodological_status"] == (
        "TURNING_POINT_REPLAY_READY_WAITING_FOR_REAL_GEOMETRY_NOT_FOR_TRAINING")


def test_no_model_artifacts_in_v2az_scope():
    suffixes = {".pt", ".pth", ".onnx", ".h5", ".ckpt", ".pkl", ".joblib", ".safetensors"}
    for base in (ROOT / "datasets", ROOT / "outputs_public"):
        for path in base.rglob("*v2az*"):
            if path.is_file():
                assert path.suffix.lower() not in suffixes
