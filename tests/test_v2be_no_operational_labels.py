"""v2be scientific guardrail tests."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_operational_promotion_or_event_polygon_created():
    summary = json.loads((ROOT / "outputs_public/execution_reports/v2be_tp1_patch_boundary_integration_summary.json").read_text(encoding="utf-8"))
    assert summary["can_train_model"] is False
    assert summary["can_create_operational_labels"] is False
    paths = list((ROOT / "datasets").glob("v2be_*.csv"))
    paths += list((ROOT / "docs").glob("v2be_*.md"))
    paths += list((ROOT / "outputs_public").rglob("v2be_*"))
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths if path.is_file())
    for forbidden in ("C4_OPERATIONAL_LABEL", "TRAINING_LABEL", "GROUND_TRUTH_FINAL",
                      "can_train_model=true", "can_create_operational_labels=true", "C:\\Users\\gabriela"):
        assert forbidden not in text
