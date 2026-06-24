from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import mv2_data_06_temporal_window_promotion as d06


def test_no_filled_template_blocks() -> None:
    status, reason = d06.classify_temporal_row({}, False)
    assert status == "NO_FILLED_TEMPLATE_FOUND"
    assert reason == "BLOCKED_NO_FILLED_TEMPLATE"


def test_missing_source_blocks() -> None:
    status, _ = d06.classify_temporal_row(
        {"temporal_window_start": "2022-01-01", "temporal_window_end": "2022-01-02"},
        True,
    )
    assert status == "BLOCKED_NO_SOURCE"


def test_inverted_window_invalid() -> None:
    status, _ = d06.classify_temporal_row(
        {
            "temporal_window_start": "2022-01-03",
            "temporal_window_end": "2022-01-02",
            "temporal_window_source": "official",
            "source_ref": "doc",
        },
        True,
    )
    assert status == "INVALID_TEMPORAL_WINDOW"


def test_traceable_reviewed_window_promotes() -> None:
    status, reason = d06.classify_temporal_row(
        {
            "temporal_window_start": "2022-01-01",
            "temporal_window_end": "2022-01-02",
            "temporal_window_source": "official",
            "source_ref": "doc",
            "review_status": "APPROVED",
        },
        True,
    )
    assert status == "PROMOTED_METADATA_READY"
    assert reason == ""
