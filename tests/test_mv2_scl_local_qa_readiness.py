from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import mv2_scl_local_qa_readiness as scl


def test_no_local_raster_not_run() -> None:
    assert scl.scl_status(False) == "NOT_RUN_NO_LOCAL_RASTER"


def test_local_raster_ready() -> None:
    assert scl.scl_status(True) == "READY"
