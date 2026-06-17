from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "multimodal"))

import revp_v2cs_to_v2cw_common as common  # noqa: E402


FORBIDDEN = ["GROUND_TRUTH_READY", "LABEL_READY", "TP2_CLOSED"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def prepared(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    common.run_seeding(root, force=True)
    return root


def test_regional_report_has_three_regions(tmp_path: Path) -> None:
    rows = common.build_regional_readiness(prepared(tmp_path))
    assert {row["region"] for row in rows} == {"Recife", "Petropolis", "Curitiba"}


def test_petropolis_has_documentary_evidence(tmp_path: Path) -> None:
    row = next(row for row in common.build_regional_readiness(prepared(tmp_path)) if row["region"] == "Petropolis")
    assert row["documentary_evidence"] == "available"


def test_recife_has_documentary_evidence_from_charter_758(tmp_path: Path) -> None:
    row = next(row for row in common.build_regional_readiness(prepared(tmp_path)) if row["region"] == "Recife")
    assert row["documentary_evidence"] == "available"


def test_curitiba_stays_not_ready_for_tp2(tmp_path: Path) -> None:
    row = next(row for row in common.build_regional_readiness(prepared(tmp_path)) if row["region"] == "Curitiba")
    assert "NOT_READY_FOR_TP2" in row["regional_status"]


def test_all_regions_keep_license_geometry_crs_hash_gaps(tmp_path: Path) -> None:
    rows = common.build_regional_readiness(prepared(tmp_path))
    assert all(row["license_gap"] == row["geometry_gap"] == row["crs_gap"] == row["hash_gap"] == "true" for row in rows)


def test_no_region_ready_for_replay(tmp_path: Path) -> None:
    rows = common.build_regional_readiness(prepared(tmp_path))
    assert all(row["replay_readiness"].startswith("blocked") for row in rows)


def test_regional_output_and_guardrails_generated(tmp_path: Path) -> None:
    root = prepared(tmp_path)
    common.run_regional_report(root, force=True)
    assert common.readiness_path(root).exists()
    assert (root / "outputs_public/logs_summary/revp_external_evidence_regional_guardrails_v2cw.csv").exists()


def test_regional_report_generated(tmp_path: Path) -> None:
    root = prepared(tmp_path)
    common.run_regional_report(root, force=True)
    assert "NOT_READY_FOR_TP2" in (root / "outputs_public/execution_reports/revp_external_evidence_regional_report_v2cw.md").read_text(encoding="utf-8")


def test_forbidden_status_tokens_absent_from_regional_outputs(tmp_path: Path) -> None:
    root = prepared(tmp_path)
    common.run_regional_report(root, force=True)
    text = common.readiness_path(root).read_text(encoding="utf-8")
    assert all(token not in text for token in FORBIDDEN)


def test_regional_forbidden_claim_blocks_labels(tmp_path: Path) -> None:
    rows = common.build_regional_readiness(prepared(tmp_path))
    assert all("label_binario" in row["forbidden_claim"] for row in rows)
