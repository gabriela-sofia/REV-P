"""v2bb scientific guardrail tests."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_license_fields_or_forbidden_promotions():
    paths = list((ROOT/"datasets").glob("v2bb_*.csv")) + list((ROOT/"docs").glob("v2bb_*.md"))
    paths += list((ROOT/"outputs_public").rglob("v2bb_*"))
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths if path.is_file())
    for forbidden in ("license_status", "license_requirement", "C4_OPERATIONAL_LABEL", "TRAINING_LABEL",
                      "GROUND_TRUTH_FINAL", "can_train_model=true", "can_create_operational_labels=true"):
        assert forbidden not in text
