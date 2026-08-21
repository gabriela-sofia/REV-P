from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import mv2_scl_local_qa_readiness as scl


def test_no_local_raster_not_run() -> None:
    assert scl.scl_status(False) == "NOT_RUN_NO_LOCAL_RASTER"


def test_local_raster_ready() -> None:
    assert scl.scl_status(True) == "READY"
