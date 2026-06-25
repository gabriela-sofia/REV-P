from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import mv2_data_08_09_metadata_execution_orchestrator as orch
from mv2_data_08_metadata_provider_contracts import MetadataConsensusRecord, MetadataQueryTarget


def test_resolve_providers() -> None:
    assert orch.resolve_providers("ALL") == ["GEE", "CDSE_STAC", "CDSE_ODATA", "TRACEABILITY"]
    assert orch.resolve_providers("GEE") == ["GEE"]
    assert orch.resolve_providers("BOGUS") == []


def test_metadata_execution_status_no_targets_is_no_call() -> None:
    assert orch.metadata_execution_status([], live=False, replay_mode=True, live_calls=0) == "NO_CALL"


def test_metadata_execution_status_replay_only_with_targets() -> None:
    targets = [MetadataQueryTarget(patch_id="P1", asset_id="A1")]
    assert orch.metadata_execution_status(targets, live=False, replay_mode=True, live_calls=0) == "REPLAY_ONLY"


def test_metadata_execution_status_live_executed() -> None:
    targets = [MetadataQueryTarget(patch_id="P1", asset_id="A1")]
    assert orch.metadata_execution_status(targets, live=True, replay_mode=False, live_calls=3) == "METADATA_ONLY_EXECUTED"


def test_lineage_consensus_status_empty_is_no_call() -> None:
    assert orch.lineage_consensus_status([]) == "NO_CALL"


def test_lineage_consensus_status_prioritizes_conflict() -> None:
    records = [
        MetadataConsensusRecord(patch_id="P1", asset_id="A1", consensus_status="STRONG"),
        MetadataConsensusRecord(patch_id="P2", asset_id="A2", consensus_status="CONFLICT"),
    ]
    assert orch.lineage_consensus_status(records) == "CONFLICT"


def test_load_config_missing_returns_none(tmp_path: Path) -> None:
    assert orch.load_config(tmp_path / "absent.json") is None


def test_default_run_is_fail_closed() -> None:
    rc = orch.main(["--strict", "--replay-only"])
    assert rc == 0
    summary = json.loads(
        (orch.OUT_DIR / "mv2_data_08_09_metadata_execution_summary.json").read_text(encoding="utf-8")
    )
    assert summary["data_06_status"] == "BLOCKED_NO_FILLED_TEMPLATE"
    assert summary["data_07_status"] == "UNKNOWN_BLOCKED"
    assert summary["data_08_status"] == "BLOCKED_NO_CONFIG"
    assert summary["metadata_execution_status"] == "NO_CALL"
    assert summary["lineage_consensus_status"] == "NO_CALL"
    assert summary["mv2_16_readiness"] == "READY_FOR_MV2_16_DRY_RUN"
    assert summary["day10_status"] == "BLOCKED"
    assert summary["live_calls"] == 0
    assert summary["downloads"] == 0
    assert summary["rasters"] == 0
    assert summary["crops"] == 0


def test_default_run_emits_timestamped_report() -> None:
    orch.main(["--strict", "--replay-only", "--timestamp", "20260623T213111"])
    report = orch.EXEC_REPORTS / "revp_data_08_09_metadata_engine_report_20260623T213111.md"
    summary = orch.EXEC_REPORTS / "revp_data_08_09_metadata_engine_summary_20260623T213111.json"
    assert report.exists()
    assert summary.exists()
