from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import mv2_data_06_08_local_promotion_orchestrator as orch


def _readiness(d06: str, d07: str, d08: str) -> dict:
    return {
        "data_06_status": d06,
        "data_07_status": d07,
        "data_08_status": d08,
        "local_input_counts": {
            "data_06_templates": 0,
            "data_07_templates": 0,
            "data_08_configs": 0,
        },
        "data_08": {
            "safe_flags": {
                "allow_network": False,
                "allow_metadata_calls": False,
                "allow_raster_download": False,
                "allow_canary_download": False,
            }
        },
    }


def test_no_local_inputs_preserve_expected_blockers() -> None:
    summary = orch.build_local_promotion(
        _readiness("NO_LOCAL_INPUT_FOUND", "NO_LOCAL_INPUT_FOUND", "BLOCKED_NO_CONFIG")
    )
    assert summary["data_06_status"] == "BLOCKED_NO_FILLED_TEMPLATE"
    assert summary["data_07_status"] == "UNKNOWN_BLOCKED"
    assert summary["data_08_status"] == "BLOCKED_NO_CONFIG"
    assert summary["mv2_16_status"] == "READY_FOR_MV2_16_DRY_RUN"
    assert summary["day10_status"] == "BLOCKED"
    assert summary["live_calls"] == 0
    assert summary["downloads"] == 0
    assert summary["rasters"] == 0
    assert summary["crops"] == 0


def test_metadata_only_ready_requires_all_three_gates() -> None:
    ready = _readiness(
        "PROMOTED_METADATA_READY",
        "SENTINEL_2_ELIGIBLE_FOUND",
        "READY_METADATA_ONLY_PREFLIGHT",
    )
    ready["data_08"]["safe_flags"] = {
        "allow_network": True,
        "allow_metadata_calls": True,
        "allow_raster_download": False,
        "allow_canary_download": False,
    }
    summary = orch.build_local_promotion(ready, allow_live_metadata=True)
    assert summary["mv2_16_status"] == "READY_FOR_MV2_16_METADATA_ONLY"
    assert summary["ready_for_real_metadata_only"] is True
    assert summary["live_calls"] == 0


def test_live_metadata_flag_does_not_override_missing_config() -> None:
    summary = orch.build_local_promotion(
        _readiness("PROMOTED_METADATA_READY", "SENTINEL_2_ELIGIBLE_FOUND", "BLOCKED_NO_CONFIG"),
        allow_live_metadata=True,
    )
    assert summary["mv2_16_status"] == "READY_FOR_MV2_16_DRY_RUN"
    assert summary["ready_for_real_metadata_only"] is False


def test_raster_download_flag_blocks_live_metadata() -> None:
    ready = _readiness(
        "PROMOTED_METADATA_READY",
        "SENTINEL_2_ELIGIBLE_FOUND",
        "READY_METADATA_ONLY_PREFLIGHT",
    )
    ready["data_08"]["safe_flags"] = {
        "allow_network": True,
        "allow_metadata_calls": True,
        "allow_raster_download": True,
        "allow_canary_download": False,
    }
    summary = orch.build_local_promotion(ready, allow_live_metadata=True)
    assert summary["ready_for_real_metadata_only"] is False
    assert summary["downloads"] == 0
