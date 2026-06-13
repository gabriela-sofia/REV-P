"""v2bh safety tests."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_model_label_training_or_ground_truth():
    summary = json.loads((ROOT / "outputs_public/execution_reports/v2bh_charter758_recife_product_georeferencing_digitization_summary.json").read_text(encoding="utf-8"))
    assert summary["can_train_model"] is False
    assert summary["can_create_operational_labels"] is False
    with (ROOT / "datasets/v2bh_safety_and_context_audit.csv").open(encoding="utf-8", newline="") as handle:
        audit = {r["rule"]: r for r in csv.DictReader(handle)}
    for rule in ("patch boundary not used as event polygon", "context not promoted", "no label", "no model", "no ground truth"):
        assert audit[rule]["passed"] == "true"
