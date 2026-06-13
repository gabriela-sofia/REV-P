"""v2bc scientific guardrail tests."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_quickview_is_visual_support_not_geometry():
    text = (ROOT / "datasets" / "v2bc_quickview_visual_support_registry.csv").read_text(encoding="utf-8")
    assert "visual_support_only" in text
    assert ",false,false,false," in text


def test_no_forbidden_promotions_or_private_paths():
    paths = list((ROOT / "datasets").glob("v2bc_*.csv")) + list((ROOT / "docs").glob("v2bc_*.md"))
    paths += list((ROOT / "outputs_public").rglob("v2bc_*"))
    text = "\n".join(p.read_text(encoding="utf-8") for p in paths if p.is_file())
    for forbidden in ("C4_OPERATIONAL_LABEL", "TRAINING_LABEL", "GROUND_TRUTH_FINAL",
                      "can_train_model=true", "can_create_operational_labels=true", "C:\\Users\\gabriela"):
        assert forbidden not in text
