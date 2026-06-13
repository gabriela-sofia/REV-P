"""v2bd scientific guardrail tests."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_forbidden_promotions_or_private_paths():
    paths = [path for path in (ROOT / "datasets").glob("v2bd_*.csv")
             if path.name != "v2bd_REC_00019_reference_inventory.csv"]
    paths += list((ROOT / "docs").glob("v2bd_*.md"))
    paths += list((ROOT / "outputs_public").rglob("v2bd_*"))
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths if path.is_file())
    for forbidden in ("C4_OPERATIONAL_LABEL", "TRAINING_LABEL", "GROUND_TRUTH_FINAL",
                      "can_train_model=true", "can_create_operational_labels=true", "C:\\Users\\gabriela"):
        assert forbidden not in text
