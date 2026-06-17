from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "multimodal"))

import revp_v2cs_to_v2cw_common as common  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def prepared(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    common.run_seeding(root, force=True)
    return root


def test_checklist_has_one_row_per_source(tmp_path: Path) -> None:
    rows = common.build_checklist(prepared(tmp_path))
    assert len(rows) == 7


def test_charter_751_targets_petropolis_products(tmp_path: Path) -> None:
    row = next(row for row in common.build_checklist(prepared(tmp_path)) if "CHARTER_751" in row["source_id"])
    assert "Petropolis" in row["target_product_name"]
    assert "PRODUCT_DISCOVERY_REQUIRED" == row["expected_blocker"]


def test_charter_758_rejects_olinda_transfer_and_flood_inference(tmp_path: Path) -> None:
    row = next(row for row in common.build_checklist(prepared(tmp_path)) if "CHARTER_758" in row["source_id"])
    assert "Olinda" in row["rejection_criteria"]
    assert "flood extent" in row["rejection_criteria"]


def test_copernicus_ems_requires_brazil_compatible_activation(tmp_path: Path) -> None:
    row = next(row for row in common.build_checklist(prepared(tmp_path)) if row["source_id"].endswith("EMS_ON_DEMAND"))
    assert "Brazil" in row["manual_steps"]


def test_gfm_requires_aoi_and_time_window(tmp_path: Path) -> None:
    row = next(row for row in common.build_checklist(prepared(tmp_path)) if row["source_id"].endswith("GFM"))
    assert "AOI" in row["minimum_required_metadata"] or "AOI" in row["manual_steps"]


def test_sgb_contextual_layer_not_tp2(tmp_path: Path) -> None:
    row = next(row for row in common.build_checklist(prepared(tmp_path)) if "SGB_CPRM" in row["source_id"])
    assert "observed event" in row["rejection_criteria"]


def test_curitiba_does_not_invent_event_dataset(tmp_path: Path) -> None:
    row = next(row for row in common.build_checklist(prepared(tmp_path)) if row["region"] == "Curitiba")
    assert "register gap" in row["manual_steps"]


def test_checklist_output_generated(tmp_path: Path) -> None:
    root = prepared(tmp_path)
    common.run_checklist(root, force=True)
    assert common.checklist_path(root).exists()


def test_checklist_report_generated(tmp_path: Path) -> None:
    root = prepared(tmp_path)
    common.run_checklist(root, force=True)
    assert "criterios de aceite" in (root / "outputs_public/execution_reports/revp_external_product_discovery_checklist_report_v2cv.md").read_text(encoding="utf-8")


def test_checklist_forbids_ground_truth_claims(tmp_path: Path) -> None:
    rows = common.build_checklist(prepared(tmp_path))
    assert all("ground_truth_operacional" in row["forbidden_claim"] for row in rows)
