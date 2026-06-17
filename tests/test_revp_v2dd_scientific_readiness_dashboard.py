from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "multimodal"))

import revp_v2cx_to_v2dd_common as common  # noqa: E402
from revp_v2dd_scientific_readiness_dashboard import main  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def prepared(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    common.write_csv(common.readiness_path(root), [
        {"readiness_id": "R1", "region": "Recife", "source_id": "SRC", "product_candidate_id": "P", "patch_id": "", "source_available": "true", "product_candidate_available": "false", "license_ready": "false", "download_ready": "false", "local_file_available": "false", "sha256_available": "false", "geospatial_qa_ready": "false", "patch_boundary_ready": "false", "pairing_ready": "false", "replay_ready": "false", "tp2_candidate_readiness": "READY_FOR_MANUAL_PRODUCT_DISCOVERY", "next_blocking_step": "PRODUCT_DISCOVERY_REQUIRED", "recommended_next_action": "manual", "allowed_claim": common.ALLOWED_CLAIM, "forbidden_claim": common.FORBIDDEN_CLAIM}
    ], common.READINESS_FIELDS)
    return root


def test_dashboard_exists_after_cli(tmp_path: Path) -> None:
    root = prepared(tmp_path)
    assert main(["--repo-root", str(root), "--offline", "--force"]) == 0
    assert common.dashboard_path(root).exists()


def test_ground_truth_operational_absent(tmp_path: Path) -> None:
    rows = common.build_dashboard(prepared(tmp_path))
    assert all(row["ground_truth_operational_status"] == "ABSENT" for row in rows)


def test_regions_present(tmp_path: Path) -> None:
    rows = common.build_dashboard(prepared(tmp_path))
    assert {row["region"] for row in rows} == {"Recife", "Petropolis", "Curitiba"}


def test_summary_contains_allowed_and_forbidden_claims(tmp_path: Path) -> None:
    root = prepared(tmp_path)
    common.run_dashboard(root, force=True)
    text = (root / "outputs_public/execution_reports/revp_v2cx_to_v2dd_scientific_summary.md").read_text(encoding="utf-8")
    assert "Allowed claims" in text
    assert "Forbidden claims" in text


def test_next_action_by_region_is_explicit(tmp_path: Path) -> None:
    rows = common.build_dashboard(prepared(tmp_path))
    assert all(row["best_next_action"] for row in rows)


def test_guardrail_log_generated(tmp_path: Path) -> None:
    root = prepared(tmp_path)
    common.run_dashboard(root, force=True)
    rows = read_csv(root / "outputs_public/logs_summary/revp_scientific_readiness_guardrails_v2dd.csv")
    assert any(row["guardrail"] == "ground_truth_operational_status" for row in rows)


@pytest.mark.parametrize("field", common.DASHBOARD_FIELDS)
def test_dashboard_has_required_fields(tmp_path: Path, field: str) -> None:
    row = common.build_dashboard(prepared(tmp_path))[0]
    assert field in row


def test_tp2_and_tp3_not_ready(tmp_path: Path) -> None:
    rows = common.build_dashboard(prepared(tmp_path))
    assert all(row["tp2_readiness_status"] == "NOT_READY_FOR_TP2" for row in rows)
    assert all(row["tp3_readiness_status"] == "NOT_READY_FOR_TP3" for row in rows)
