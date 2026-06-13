"""v2bg TP2 gate and TP3 precheck tests."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_digitization_ready_without_tp2_or_tp3():
    summary = json.loads((ROOT / "outputs_public/execution_reports/v2bg_charter758_deep_product_mining_tp2_recovery_summary.json").read_text(encoding="utf-8"))
    assert summary["tp1_available"] is True
    assert summary["digitization_ready_from_public_product"] is True
    assert summary["valid_event_polygons"] == 0
    assert summary["tp2_feeds_ready"] == 0
    assert summary["tp3_precheck_ready"] is False
    assert summary["turning_point_level"] == "TP2_DIGITIZATION_READY_FROM_PUBLIC_CHARTER_PRODUCT"
    with (ROOT / "datasets/v2bg_charter758_tp3_precheck.csv").open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["tp2_digitization_ready"] == "true"
    assert row["ready_for_v2au_overlay"] == "false"
