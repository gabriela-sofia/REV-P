"""Guardrail tests for MV2 pre-unification contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import mv2_pre_unification_run as pre  # noqa: E402


def test_required_field_missing_fails() -> None:
    with pytest.raises(ValueError):
        pre.validate_required({"patch_id": "P1"}, ["patch_id", "asset_id"])


def test_unknown_sensor_blocks_spectral_eligibility() -> None:
    result = pre.classify_sensor_family("UNKNOWN")
    assert result["lineage_classification"] == "UNKNOWN_BLOCKED"
    assert result["spectral_eligible"] is False


def test_dino_derived_blocks_spectral_eligibility() -> None:
    result = pre.classify_sensor_family("DINO_DERIVED")
    assert result["lineage_classification"] == "DINO_DERIVED_BLOCKED"
    assert result["spectral_eligible"] is False


def test_empty_temporal_window_blocks_probe() -> None:
    assert pre.temporal_promotion_status({"patch_id": "P1", "asset_id": "A1"}) == "BLOCKED_EMPTY"


def test_scene_without_product_id_does_not_authorize_crop() -> None:
    allowed, status = pre.can_authorize_crop(
        {
            "sensor_family": "SENTINEL_2",
            "temporal_window_start": "2022-01-01",
            "temporal_window_end": "2022-01-02",
            "bbox": "0,0,1,1",
        }
    )
    assert allowed is False
    assert status == "NOT_AUTHORIZED_NO_PRODUCT_ID"


def test_public_raster_manifest_fails() -> None:
    with pytest.raises(ValueError):
        pre.local_raster_manifest_guard(
            {
                "is_public": "true",
                "local_only_path": "outputs_public/bad.tif",
            }
        )


def test_crop_without_scl_qa_does_not_unlock_day10() -> None:
    assert pre.day10_gate({"crop_status": "AUTHORIZED", "scl_qa_status": "NOT_RUN"}) == "BLOCKED"


def test_unknown_is_not_negative() -> None:
    assert pre.unknown_is_negative("UNKNOWN") is False


def test_event_without_geometry_does_not_become_silver() -> None:
    assert (
        pre.observational_silver_status({"event_id": "E1", "adjudication_status": "ADJUDICATED"})
        == "BLOCKED_INSUFFICIENT_EVIDENCE"
    )


def test_textual_anchor_does_not_become_strong_overlay() -> None:
    status = pre.observational_silver_status(
        {
            "geometry_wgs84": '{"type":"Polygon","coordinates":[]}',
            "adjudication_status": "ADJUDICATED",
            "uncertainty_class": "HIGH",
            "evidence_sources": "TEXTUAL_ANCHOR_ONLY",
        }
    )
    assert status == "TEXTUAL_ANCHOR_ONLY"
