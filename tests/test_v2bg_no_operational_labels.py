"""v2bg scientific guardrail tests."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_labels_models_truth_or_license_blockers():
    summary = json.loads((ROOT / "outputs_public/execution_reports/v2bg_charter758_deep_product_mining_tp2_recovery_summary.json").read_text(encoding="utf-8"))
    assert summary["can_train_model"] is False
    assert summary["can_create_operational_labels"] is False
    text = "\n".join(path.read_text(encoding="utf-8") for path in
                     list((ROOT / "datasets").glob("v2bg_*.csv")) + list((ROOT / "docs").glob("v2bg_*.md")))
    for forbidden in ("license_status", "license_requirement", "C4_OPERATIONAL_LABEL", "TRAINING_LABEL",
                      "GROUND_TRUTH_FINAL", "can_train_model=true", "can_create_operational_labels=true"):
        assert forbidden not in text
