from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import mv2_16_unified_sentinel_execution_core as core


def test_gate_a_blocks_without_temporal_window() -> None:
    assert core.compute_gate_a({}) == "BLOCKED_NO_TEMPORAL_WINDOW"


def test_gate_a_blocks_without_sensor() -> None:
    assert core.compute_gate_a({"temporal_status": "PROMOTED_METADATA_READY"}) == "BLOCKED_NO_SENSOR_LINEAGE"


def test_gate_a_blocks_without_config() -> None:
    row = {"temporal_status": "PROMOTED_METADATA_READY", "lineage_status": "SENTINEL_2_ELIGIBLE", "metadata_status": "BLOCKED_NO_CONFIG"}
    assert core.compute_gate_a(row) == "BLOCKED_NO_CONFIG"


def test_dry_run_readiness_with_blocked_rows() -> None:
    rows = [{"gate_a": "BLOCKED_NO_TEMPORAL_WINDOW"}]
    assert core.compute_mv2_16_readiness(rows) == "READY_FOR_MV2_16_DRY_RUN"
