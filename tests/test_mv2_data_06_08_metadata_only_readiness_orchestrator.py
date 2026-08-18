from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import mv2_data_06_08_metadata_only_readiness_orchestrator as orch


def test_evaluate_data06_blocked_no_filled_template() -> None:
    result = orch.evaluate_data06()
    assert result["status"] == "BLOCKED_NO_FILLED_TEMPLATE"
    assert result["promoted_metadata_ready"] == 0
    assert result["filled_template_found"] is False


def test_evaluate_data07_unknown_blocked() -> None:
    result = orch.evaluate_data07()
    assert result["status"] == "UNKNOWN_BLOCKED"
    assert result["sentinel_2_eligible"] == 0


def test_evaluate_data08_blocked_no_config() -> None:
    result = orch.evaluate_data08()
    assert result["status"] == "BLOCKED_NO_CONFIG"
    assert result["calls_allowed"] is False


def test_evaluate_mv2_16_dry_run_and_day10_blocked() -> None:
    result = orch.evaluate_mv2_16()
    assert result["readiness"] == "READY_FOR_MV2_16_DRY_RUN"
    assert result["day10_status"] == "BLOCKED"


def test_consolidation_reports_zero_side_effects() -> None:
    consolidation = orch.build_consolidation(
        orch.evaluate_data06(),
        orch.evaluate_data07(),
        orch.evaluate_data08(),
        orch.evaluate_mv2_16(),
    )
    assert consolidation["fail_closed"] is True
    assert consolidation["api_calls"] == 0
    assert consolidation["downloads"] == 0
    assert consolidation["rasters"] == 0
    assert consolidation["crops"] == 0
    assert consolidation["data_06_status"] == "BLOCKED_NO_FILLED_TEMPLATE"
    assert consolidation["data_07_status"] == "UNKNOWN_BLOCKED"
    assert consolidation["data_08_status"] == "BLOCKED_NO_CONFIG"
    assert consolidation["mv2_16_readiness"] == "READY_FOR_MV2_16_DRY_RUN"
    assert consolidation["day10_status"] == "BLOCKED"
