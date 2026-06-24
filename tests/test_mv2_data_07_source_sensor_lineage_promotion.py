from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import mv2_data_07_source_sensor_lineage_promotion as d07


def test_unknown_blocks() -> None:
    assert d07.classify_sensor_lineage("UNKNOWN") == ("UNKNOWN_BLOCKED", "sensor_lineage_ausente")


def test_dino_blocks() -> None:
    status, _ = d07.classify_sensor_lineage("DINO_DERIVED")
    assert status == "DINO_DERIVED_BLOCKED"


def test_sentinel_2_requires_traceable_source() -> None:
    status, reason = d07.classify_sensor_lineage("SENTINEL_2", "")
    assert status == "UNKNOWN_BLOCKED"
    assert reason == "sem_source_asset_ref_rastreavel"


def test_sentinel_2_with_source_is_eligible() -> None:
    assert d07.classify_sensor_lineage("SENTINEL_2", "S2_PRODUCT") == ("SENTINEL_2_ELIGIBLE", "")
