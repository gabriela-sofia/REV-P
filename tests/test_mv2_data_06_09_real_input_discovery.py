from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import mv2_data_06_09_real_input_discovery as disc

VALID_D06 = (
    "patch_id,asset_id,temporal_window_start,temporal_window_end,temporal_window_source,source_ref\n"
    "REC_00019,e07eacbc8a366650,2022-05-24,2022-05-30,COPERNICUS_EMS,EMSR_EXEMPLO_REF\n"
)
VALID_D07 = (
    "patch_id,asset_id,sensor_family,sensor_source_ref\n"
    "REC_00019,e07eacbc8a366650,SENTINEL_2,S2A_MSIL2A_EXEMPLO\n"
)


def _make_project(tmp_path: Path, d06: str | None = None, d07: str | None = None, config: dict | None = None, env: bool = False) -> Path:
    (tmp_path / "inputs_local" / "data_06_temporal_windows").mkdir(parents=True, exist_ok=True)
    (tmp_path / "inputs_local" / "data_07_sensor_lineage").mkdir(parents=True, exist_ok=True)
    (tmp_path / "inputs_local" / "data_08_metadata_config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "datasets" / "schemas").mkdir(parents=True, exist_ok=True)
    if d06:
        (tmp_path / "inputs_local" / "data_06_temporal_windows" / "real.csv").write_text(d06, encoding="utf-8")
    if d07:
        (tmp_path / "inputs_local" / "data_07_sensor_lineage" / "real.csv").write_text(d07, encoding="utf-8")
    if config is not None:
        (tmp_path / "configs" / "api_config.local.json").write_text(json.dumps(config), encoding="utf-8")
    if env:
        (tmp_path / ".env").write_text("SECRET_TOKEN=should_never_be_printed\n", encoding="utf-8")
    return tmp_path


def test_no_real_input_found(tmp_path: Path) -> None:
    result = disc.classify_real_inputs(_make_project(tmp_path))
    assert result["overall_status"] == "NO_REAL_LOCAL_INPUT_FOUND"
    assert result["data_06_status"] == "DATA_06_NO_REAL_INPUT"
    assert result["data_07_status"] == "DATA_07_NO_REAL_INPUT"
    assert result["data_08_status"] == "DATA_08_NO_CONFIG"


def test_valid_data06_is_promotable(tmp_path: Path) -> None:
    result = disc.classify_real_inputs(_make_project(tmp_path, d06=VALID_D06))
    assert result["data_06_status"] == "DATA_06_PROMOTABLE"


def test_valid_data07_is_s2_eligible(tmp_path: Path) -> None:
    result = disc.classify_real_inputs(_make_project(tmp_path, d07=VALID_D07))
    assert result["data_07_status"] == "DATA_07_S2_ELIGIBLE"


def test_config_with_raster_download_is_blocked_by_flags(tmp_path: Path) -> None:
    config = {"allow_network": True, "allow_metadata_calls": True, "allow_raster_download": True, "allow_canary_download": False}
    result = disc.classify_real_inputs(_make_project(tmp_path, config=config))
    assert result["data_08_status"] == "DATA_08_BLOCKED_BY_FLAGS"


def test_config_metadata_only_is_ready(tmp_path: Path) -> None:
    config = {"allow_network": True, "allow_metadata_calls": True, "allow_raster_download": False, "allow_canary_download": False}
    result = disc.classify_real_inputs(_make_project(tmp_path, config=config))
    assert result["data_08_status"] == "DATA_08_READY_METADATA_ONLY"


def test_local_input_never_copied_to_outputs(tmp_path: Path) -> None:
    result = disc.classify_real_inputs(_make_project(tmp_path, d06=VALID_D06))
    manifest_blob = json.dumps(result["public_manifest"])
    # The manifest must not carry filled values from the real template.
    assert "EMSR_EXEMPLO_REF" not in manifest_blob
    assert "2022-05-24" not in manifest_blob
    # It must carry only redacted metadata.
    assert result["public_manifest"]["data_06_templates"][0]["sha256"]


def test_private_paths_and_secrets_are_redacted(tmp_path: Path) -> None:
    result = disc.classify_real_inputs(_make_project(tmp_path, env=True))
    blob = json.dumps(result)
    assert "should_never_be_printed" not in blob
    assert result["env_files_present"] == 1
    # Redacted path uses a placeholder, not an absolute path.
    assert all(not p.startswith("C:") for p in result["public_manifest"]["env_files_present"])


def test_no_side_effects(tmp_path: Path) -> None:
    result = disc.classify_real_inputs(_make_project(tmp_path))
    assert result["live_calls"] == 0
    assert result["downloads"] == 0
    assert result["rasters"] == 0
    assert result["crops"] == 0
