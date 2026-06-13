"""v2ba scientific guardrail tests."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_summary_forbids_training_and_operational_labels():
    summary = json.loads((ROOT / "outputs_public" / "execution_reports" /
                          "v2ba_minimal_real_geometry_acquisition_workbench_summary.json").read_text(encoding="utf-8"))
    assert summary["can_train_model"] is False
    assert summary["can_create_operational_labels"] is False


def test_generated_artifacts_contain_no_forbidden_promotions():
    paths = list((ROOT / "datasets").glob("v2ba_*.csv"))
    paths += list((ROOT / "docs").glob("v2ba_*.md"))
    paths += list((ROOT / "outputs_public").rglob("v2ba_*"))
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths if path.is_file())
    for forbidden in ("C4_OPERATIONAL_LABEL", "TRAINING_LABEL", "GROUND_TRUTH_FINAL",
                      "can_train_model=true", "can_create_operational_labels=true"):
        assert forbidden not in text
