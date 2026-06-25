from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import mv2_data_06_09_real_metadata_evolution_orchestrator as orch
import mv2_data_08_lineage_consensus_engine as consensus
from mv2_data_08_metadata_clients import build_clients
from mv2_data_08_metadata_provider_contracts import MetadataQueryTarget

GOOD_CONFIG = {
    "allow_network": True,
    "allow_metadata_calls": True,
    "allow_raster_download": False,
    "allow_canary_download": False,
}
GEOM = {"type": "Polygon", "coordinates": [[[0, 0]]]}


def _target() -> MetadataQueryTarget:
    return MetadataQueryTarget(
        patch_id="P1",
        asset_id="A1",
        sensor_family="SENTINEL_2",
        temporal_window_start="2022-05-24",
        temporal_window_end="2022-05-30",
        aoi_wgs84=[-34.99, -8.23, -34.98, -8.22],
    )


def test_final_status_mappers() -> None:
    assert orch.final_data06_status("DATA_06_PROMOTABLE") == "PROMOTED_METADATA_READY"
    assert orch.final_data06_status("DATA_06_NO_REAL_INPUT") == "BLOCKED_NO_REAL_TEMPORAL_WINDOW"
    assert orch.final_data07_status("DATA_07_S2_ELIGIBLE") == "SENTINEL_2_ELIGIBLE_FOUND"
    assert orch.final_data07_status("DATA_07_INVALID") == "BLOCKED_NO_REAL_SENSOR_LINEAGE"
    assert orch.final_data08_status("DATA_08_READY_METADATA_ONLY") == "READY_METADATA_ONLY_PREFLIGHT"
    assert orch.final_data08_status("DATA_08_NO_CONFIG") == "BLOCKED_NO_CONFIG"


def test_data06_valid_but_data07_missing_blocks_metadata() -> None:
    ready = orch.all_gates_ready("PROMOTED_METADATA_READY", "BLOCKED_NO_REAL_SENSOR_LINEAGE", "READY_METADATA_ONLY_PREFLIGHT")
    assert ready is False
    assert orch.metadata_execution_status(ready, live=True, live_calls=0) == "NO_CALL"


def test_data07_valid_but_data06_missing_blocks_metadata() -> None:
    ready = orch.all_gates_ready("BLOCKED_NO_REAL_TEMPORAL_WINDOW", "SENTINEL_2_ELIGIBLE_FOUND", "READY_METADATA_ONLY_PREFLIGHT")
    assert ready is False
    assert orch.metadata_execution_status(ready, live=False, live_calls=0) == "NO_CALL"


def test_all_gates_ready_true_only_when_three_green() -> None:
    assert orch.all_gates_ready("PROMOTED_METADATA_READY", "SENTINEL_2_ELIGIBLE_FOUND", "READY_METADATA_ONLY_PREFLIGHT") is True


def test_ready_but_not_executed_without_live() -> None:
    assert orch.metadata_execution_status(True, live=False, live_calls=0) == "READY_BUT_NOT_EXECUTED"


def test_metadata_only_done_when_live_executed() -> None:
    assert orch.metadata_execution_status(True, live=True, live_calls=2) == "METADATA_ONLY_DONE"


def test_consensus_status_no_call_without_live() -> None:
    assert orch.lineage_consensus_status([], live=False, live_calls=0) == "NO_CALL"


def test_mv2_16_readiness_metadata_only_when_ready() -> None:
    assert orch.mv2_16_readiness(True) == "READY_FOR_MV2_16_METADATA_ONLY"
    assert orch.mv2_16_readiness(False) == "READY_FOR_MV2_16_DRY_RUN"


def test_no_call_when_not_live() -> None:
    result = orch.run_metadata([_target()], ["GEE"], live=False, replay_mode=True, config=None)
    assert result["live_calls"] == 0


def test_synthetic_fixtures_readiness_metadata_only_no_network() -> None:
    # Replay-mode clients read fixtures and never touch the network.
    result = orch.run_metadata([_target()], ["GEE", "CDSE_STAC"], live=False, replay_mode=True, config=None)
    assert result["live_calls"] == 0
    # Empty fixtures -> NO_MATCH per provider, consensus NO_MATCH.
    assert all(r.consensus_status == "NO_MATCH" for r in result["records"])


def test_strong_consensus_with_equal_product_id_in_fixtures() -> None:
    raw = {
        "features": [
            {
                "id": "S2_SCENE",
                "geometry": GEOM,
                "properties": {"PRODUCT_ID": "S2A_X", "datetime": "2022-05-24T13:00:00Z", "MGRS_TILE": "25MGR"},
            }
        ]
    }
    stac_raw = {
        "features": [
            {
                "id": "S2_SCENE",
                "collection": "S2_L2A",
                "geometry": GEOM,
                "bbox": [-34.99, -8.23, -34.98, -8.22],
                "properties": {"s2:product_uri": "S2A_X", "datetime": "2022-05-24T13:00:00Z", "grid:code": "25MGR"},
            }
        ]
    }
    clients = build_clients(
        ["GEE", "CDSE_STAC"],
        live=True,
        config=GOOD_CONFIG,
        transports={"GEE": lambda t: raw, "CDSE_STAC": lambda t: stac_raw},
    )
    target = _target()
    results = []
    for client in clients.values():
        results.extend(client.query(target))
    record = consensus.compute_consensus("P1", "A1", results)
    assert record.consensus_status == "STRONG"
    assert record.product_id == "S2A_X"


def test_conflict_consensus_with_divergent_product_id() -> None:
    gee_raw = {"features": [{"id": "S", "geometry": GEOM, "properties": {"PRODUCT_ID": "S2A_X", "datetime": "2022-05-24T13:00:00Z"}}]}
    stac_raw = {"features": [{"id": "S", "geometry": GEOM, "properties": {"s2:product_uri": "S2A_Y", "datetime": "2022-05-24T13:00:00Z"}}]}
    clients = build_clients(
        ["GEE", "CDSE_STAC"],
        live=True,
        config=GOOD_CONFIG,
        transports={"GEE": lambda t: gee_raw, "CDSE_STAC": lambda t: stac_raw},
    )
    target = _target()
    results = []
    for client in clients.values():
        results.extend(client.query(target))
    record = consensus.compute_consensus("P1", "A1", results)
    assert record.consensus_status == "CONFLICT"
    assert record.conflict_reason == "divergent_product_id"


def test_default_run_is_fail_closed_invariants() -> None:
    # Invariants that hold regardless of whether real local inputs were acquired
    # (inputs_local/ is git-ignored and may legitimately contain real candidates).
    rc = orch.main(["--strict", "--replay-only"])
    assert rc == 0
    summary = json.loads(
        (orch.OUT_DIR / "mv2_data_06_09_real_metadata_evolution_summary.json").read_text(encoding="utf-8")
    )
    assert summary["mode"] == "replay-only"
    assert summary["strict"] is True
    # DATA-08 stays blocked unless a local config is created (none here).
    assert summary["data_08_status"] == "BLOCKED_NO_CONFIG"
    # No live config => no metadata call and no consensus call.
    assert summary["metadata_execution_status"] == "NO_CALL"
    assert summary["lineage_consensus_status"] == "NO_CALL"
    assert summary["mv2_16_readiness"] in {"READY_FOR_MV2_16_DRY_RUN", "READY_FOR_MV2_16_METADATA_ONLY"}
    # DATA-06 status is input-dependent (git-ignored local candidate may exist).
    assert summary["data_06_status"] in {"BLOCKED_NO_REAL_TEMPORAL_WINDOW", "PROMOTED_METADATA_READY"}
    assert summary["day10_status"] == "BLOCKED"
    assert summary["live_calls"] == 0
    assert summary["downloads"] == 0
    assert summary["rasters"] == 0
    assert summary["crops"] == 0
