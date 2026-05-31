"""Tests for v1oh local Sentinel asset metadata scanner.

All I/O is redirected to tmp_path — datasets/ is never touched.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1oh_local_sentinel_asset_metadata_scanner.py"
DATASETS = ROOT / "datasets"


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_v1oh_scans_sidecar_without_using_mtime_or_pixels(tmp_path: Path) -> None:
    sidecar = tmp_path / "REC_PATCH_A_S2A_MSIL2A_20220525T131241_T25LDD.json"
    sidecar.write_text('{"source":"S2A_MSIL2A_20220525T131241_T25LDD"}', encoding="utf-8")

    out_inv = tmp_path / "inv.csv"
    out_date = tmp_path / "date.csv"
    out_summary = tmp_path / "summary.csv"

    env = {**os.environ,
           "REVP_RECIFE_LOCAL_SENTINEL_SCAN_ROOTS": str(tmp_path),
           "REVP_V1OH_OUT_INV": str(out_inv),
           "REVP_V1OH_OUT_DATE": str(out_date),
           "REVP_V1OH_OUT_SUMMARY": str(out_summary),
           "REVP_V1OH_SCHEMA_INV": str(tmp_path / "s_inv.csv"),
           "REVP_V1OH_SCHEMA_DATE": str(tmp_path / "s_date.csv"),
           "REVP_V1OH_SCHEMA_SUMMARY": str(tmp_path / "s_summary.csv"),
           "REVP_V1OH_DOC": str(tmp_path / "doc.md")}

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--force", "--emit-evidence"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert _read(out_inv)[0]["pixel_read_status"] == "NOT_READ"
    assert _read(out_date)[0]["date_candidate"] == "2022-05-25"
    assert _read(out_summary)[0]["scan_status"] == "LOCAL_SENTINEL_ASSET_FOUND"
    assert "mtime" not in out_inv.read_text(encoding="utf-8").casefold()

    # Verify the test outputs are in tmp_path, not in real datasets/
    assert out_inv.exists(), "Output must be in tmp_path"
    assert not (DATASETS / "recife_local_sentinel_asset_metadata_inventory.csv").samefile(out_inv), \
        "Test must not write to real datasets/"
