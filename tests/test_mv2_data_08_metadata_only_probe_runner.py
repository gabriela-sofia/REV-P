from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import mv2_data_08_metadata_only_probe_runner as d08


def test_missing_config_blocks(tmp_path: Path) -> None:
    result = d08.metadata_preflight(tmp_path / "missing.json")
    assert result["status"] == "BLOCKED_NO_CONFIG"
    assert result["calls_allowed"] is False


def test_flags_block_calls(tmp_path: Path) -> None:
    cfg = tmp_path / "api_config.local.json"
    cfg.write_text(json.dumps({"allow_network": False, "allow_metadata_calls": False}), encoding="utf-8")
    result = d08.metadata_preflight(cfg)
    assert result["status"] == "BLOCKED_BY_FLAGS"


def test_metadata_ready_requires_no_downloads(tmp_path: Path) -> None:
    cfg = tmp_path / "api_config.local.json"
    cfg.write_text(
        json.dumps(
            {
                "allow_network": True,
                "allow_metadata_calls": True,
                "allow_raster_download": False,
                "allow_canary_download": False,
                "providers": {},
            }
        ),
        encoding="utf-8",
    )
    result = d08.metadata_preflight(cfg)
    assert result["status"] == "READY_METADATA_ONLY"
