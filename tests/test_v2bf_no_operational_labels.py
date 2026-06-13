"""v2bf scientific guardrail tests."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_labels_models_truth_or_license_blockers():
    summary = json.loads((ROOT / "outputs_public/execution_reports/v2bf_recife_observed_event_polygon_tp2_summary.json").read_text(encoding="utf-8"))
    assert summary["can_train_model"] is False
    assert summary["can_create_operational_labels"] is False
    paths = list((ROOT / "datasets").glob("v2bf_*.csv")) + list((ROOT / "docs").glob("v2bf_*.md"))
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    for forbidden in ("license_status", "license_requirement", "C4_OPERATIONAL_LABEL", "TRAINING_LABEL",
                      "GROUND_TRUTH_FINAL", "can_train_model=true", "can_create_operational_labels=true"):
        assert forbidden not in text
