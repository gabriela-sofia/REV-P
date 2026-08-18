from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import mv2_data_06_prepare_temporal_window_human_pack as pack


def test_template_rows_never_fill_dates() -> None:
    promotion = [{"patch_id": "REC_00019", "asset_id": "abc"}]
    rows = pack.build_template_rows(promotion, [])
    assert len(rows) == 1
    row = rows[0]
    assert row["temporal_window_start"] == ""
    assert row["temporal_window_end"] == ""
    assert row["review_status"] == "PENDING_HUMAN_FILL"
    assert row["blocked_reason"] == "BLOCKED_NO_FILLED_TEMPLATE"


def test_build_template_rows_uses_seed_enrichment_only() -> None:
    promotion = [{"patch_id": "REC_00019", "asset_id": "abc"}]
    seed = [{"patch_id": "REC_00019", "asset_id": "abc", "city": "Recife", "bbox": "1,2,3,4", "crs": "EPSG:32725"}]
    rows = pack.build_template_rows(promotion, seed)
    assert rows[0]["city"] == "Recife"
    assert rows[0]["bbox"] == "1,2,3,4"
    # enrichment must not create a temporal window
    assert rows[0]["temporal_window_start"] == ""


def test_assert_no_autofilled_dates_raises_on_inferred_window() -> None:
    rows = [{"patch_id": "X", "temporal_window_start": "2022-01-01", "temporal_window_end": ""}]
    with pytest.raises(ValueError):
        pack.assert_no_autofilled_dates(rows)


def test_summary_is_blocked_with_zero_side_effects() -> None:
    rows = pack.build_template_rows([{"patch_id": "X", "asset_id": "y"}], [])
    summary = pack.summarize(rows)
    assert summary["promotion_status"] == "BLOCKED_NO_FILLED_TEMPLATE"
    assert summary["filled_windows"] == 0
    assert summary["api_calls"] == 0
    assert summary["downloads"] == 0
    assert summary["rasters"] == 0
    assert summary["crops"] == 0


def test_source_policy_lists_accepted_and_rejected() -> None:
    assert "CEMADEN" in pack.ACCEPTED_SOURCE_TYPES
    assert "ANA" in pack.ACCEPTED_SOURCE_TYPES
    assert "BBOX_ONLY_SEARCH" in pack.REJECTED_SOURCE_TYPES
    assert "DATA_INVENTADA" in pack.REJECTED_SOURCE_TYPES
