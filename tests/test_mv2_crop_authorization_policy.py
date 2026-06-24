from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import mv2_crop_authorization_policy as crop


def test_no_temporal_window_blocks_crop() -> None:
    assert crop.evaluate_crop_authorization({}) == "NOT_AUTHORIZED_NO_TEMPORAL_WINDOW"


def test_no_sensor_blocks_crop() -> None:
    assert crop.evaluate_crop_authorization({"temporal_status": "PROMOTED_METADATA_READY"}) == "NOT_AUTHORIZED_NO_SENSOR"


def test_full_metadata_authorizes_metadata_only() -> None:
    status = crop.evaluate_crop_authorization(
        {
            "temporal_status": "PROMOTED_METADATA_READY",
            "lineage_classification": "SENTINEL_2_ELIGIBLE",
            "product_id": "P",
            "scene_id": "S",
            "bbox": "0,0,1,1",
            "consensus_status": "STRONG",
        }
    )
    assert status == "AUTHORIZED_METADATA_ONLY"
