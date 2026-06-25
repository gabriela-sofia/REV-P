from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import mv2_data_07_prepare_sensor_lineage_human_pack as pack


def test_template_defaults_to_unknown_blocked() -> None:
    promotion = [{"patch_id": "REC_00019", "asset_id": "abc", "slot_id": "2"}]
    rows = pack.build_template_rows(promotion)
    assert len(rows) == 1
    row = rows[0]
    assert row["sensor_family"] == "UNKNOWN"
    assert row["spectral_eligible"] == "false"
    assert row["sensor_source_ref"] == ""
    assert row["blocked_reason"] == "UNKNOWN_BLOCKED"


def test_eligibility_for_family_rules() -> None:
    assert pack.eligibility_for_family("SENTINEL_2") == (True, False)
    assert pack.eligibility_for_family("SENTINEL_1") == (False, True)
    assert pack.eligibility_for_family("DINO_DERIVED") == (False, False)
    assert pack.eligibility_for_family("PNG_RENDER") == (False, False)
    assert pack.eligibility_for_family("NPZ_EMBEDDING") == (False, False)
    assert pack.eligibility_for_family("UNKNOWN") == (False, False)
    assert pack.eligibility_for_family("CONFLICT") == (False, False)


def test_assert_no_inferred_sensor_raises_on_eligible_without_ref() -> None:
    rows = [{"patch_id": "X", "sensor_family": "SENTINEL_2", "spectral_eligible": "true", "sensor_source_ref": ""}]
    with pytest.raises(ValueError):
        pack.assert_no_inferred_sensor(rows)


def test_allowed_sensor_families_are_exact() -> None:
    assert pack.ALLOWED_SENSOR_FAMILIES == [
        "SENTINEL_2",
        "SENTINEL_1",
        "DINO_DERIVED",
        "PNG_RENDER",
        "NPZ_EMBEDDING",
        "UNKNOWN",
        "CONFLICT",
    ]


def test_summary_blocked_zero_side_effects() -> None:
    rows = pack.build_template_rows([{"patch_id": "X", "asset_id": "y"}])
    summary = pack.summarize(rows)
    assert summary["promotion_status"] == "UNKNOWN_BLOCKED"
    assert summary["spectral_eligible"] == 0
    assert summary["api_calls"] == 0
    assert summary["downloads"] == 0
    assert summary["rasters"] == 0
    assert summary["crops"] == 0
