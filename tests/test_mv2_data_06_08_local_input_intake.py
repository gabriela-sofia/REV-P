from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import mv2_data_06_08_local_input_intake as intake


def test_no_local_input_is_fail_closed(tmp_path: Path) -> None:
    intake.ensure_local_input_dirs(tmp_path)
    readiness = intake.classify_local_input_readiness(tmp_path)
    assert readiness["data_06_status"] == "NO_LOCAL_INPUT_FOUND"
    assert readiness["data_07_status"] == "NO_LOCAL_INPUT_FOUND"
    assert readiness["data_08_status"] == "BLOCKED_NO_CONFIG"
    assert readiness["live_calls"] == 0
    assert readiness["downloads"] == 0
    assert readiness["rasters"] == 0
    assert readiness["crops"] == 0


def test_data06_valid_requires_traceable_source(tmp_path: Path) -> None:
    path = tmp_path / "data06.csv"
    path.write_text(
        "patch_id,asset_id,temporal_window_start,temporal_window_end,temporal_window_source,source_ref\n"
        "P1,A1,2022-01-01,2022-01-02,DEFESA_CIVIL,REF-1\n",
        encoding="utf-8",
    )
    result = intake.validate_data_06_template(path)
    assert result["status"] == "PROMOTED_METADATA_READY"
    assert result["promoted_rows"] == 1


def test_data06_date_without_source_blocks(tmp_path: Path) -> None:
    path = tmp_path / "data06.csv"
    path.write_text(
        "patch_id,asset_id,temporal_window_start,temporal_window_end,temporal_window_source,source_ref\n"
        "P1,A1,2022-01-01,2022-01-02,,\n",
        encoding="utf-8",
    )
    result = intake.validate_data_06_template(path)
    assert result["status"] == "BLOCKED_INVALID_TEMPLATE"
    assert "row_1:date_without_traceable_source" in result["errors"]


def test_data07_sentinel2_requires_sensor_source_ref(tmp_path: Path) -> None:
    path = tmp_path / "data07.csv"
    path.write_text(
        "patch_id,asset_id,sensor_family,sensor_source_ref\n"
        "P1,A1,SENTINEL_2,\n",
        encoding="utf-8",
    )
    result = intake.validate_data_07_template(path)
    assert result["status"] == "BLOCKED_INVALID_SENSOR_LINEAGE"
    assert "row_1:sentinel_2_without_sensor_source_ref" in result["errors"]


def test_data07_sentinel2_with_ref_promotes(tmp_path: Path) -> None:
    path = tmp_path / "data07.csv"
    path.write_text(
        "patch_id,asset_id,sensor_family,sensor_source_ref\n"
        "P1,A1,SENTINEL_2,S2_PRODUCT_REF\n",
        encoding="utf-8",
    )
    result = intake.validate_data_07_template(path)
    assert result["status"] == "SENTINEL_2_ELIGIBLE_FOUND"
    assert result["sentinel_2_eligible"] == 1


def test_data08_blocks_raster_and_canary_download_flags(tmp_path: Path) -> None:
    path = tmp_path / "api_config.local.json"
    path.write_text(
        json.dumps(
            {
                "allow_network": True,
                "allow_metadata_calls": True,
                "allow_raster_download": True,
                "allow_canary_download": False,
            }
        ),
        encoding="utf-8",
    )
    result = intake.validate_data_08_config_presence(path)
    assert result["status"] == "BLOCKED_BY_FLAGS"


def test_data08_ready_metadata_only_prefight_flags(tmp_path: Path) -> None:
    path = tmp_path / "api_config.local.json"
    path.write_text(
        json.dumps(
            {
                "allow_network": True,
                "allow_metadata_calls": True,
                "allow_raster_download": False,
                "allow_canary_download": False,
            }
        ),
        encoding="utf-8",
    )
    result = intake.validate_data_08_config_presence(path)
    assert result["status"] == "READY_METADATA_ONLY_PREFLIGHT"
    assert result["safe_flags"]["allow_raster_download"] is False


def test_redacts_inputs_local_paths(tmp_path: Path) -> None:
    local_file = tmp_path / "inputs_local" / "data_06_temporal_windows" / "filled.csv"
    local_file.parent.mkdir(parents=True)
    local_file.write_text("x\n", encoding="utf-8")
    assert intake.redact_local_paths(local_file, tmp_path) == "<inputs_local>/data_06_temporal_windows/filled.csv"
