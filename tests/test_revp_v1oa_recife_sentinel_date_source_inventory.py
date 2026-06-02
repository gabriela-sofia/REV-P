"""Tests for v1oa Recife Sentinel date source inventory."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1oa_recife_sentinel_date_source_inventory.py"
OUT = ROOT / "datasets/recife_sentinel_date_source_inventory.csv"
READINESS = ROOT / "datasets/recife_sentinel_date_source_readiness_matrix.csv"
SCHEMA = ROOT / "datasets/schemas/recife_sentinel_date_source_inventory_schema.csv"


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_v1oa_inventories_manifest_sources_without_absolute_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_rows(manifest, [{"patch_id": "REC_PATCH_A", "region": "Recife", "asset_path_reference": "data/sentinel/S2A_MSIL2A_20220525T131241_T25LDD_patch.tif"}])
    env = os.environ.copy()
    env["REVP_RECIFE_SENTINEL_DATE_SOURCE_PATHS"] = str(manifest)
    env["REVP_RECIFE_SENTINEL_LOCAL_SCAN_DIRS"] = str(tmp_path / "missing")
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    rows = read_rows(OUT)
    assert rows
    assert rows[0]["patch_id"] == "REC_PATCH_A"
    assert rows[0]["pixel_read_status"] == "NOT_READ"
    assert read_rows(READINESS)[0]["can_train_model"] == "false"
    assert "source_inventory_id" in {row["field"] for row in read_rows(SCHEMA)}
    assert "C:\\" not in OUT.read_text(encoding="utf-8", errors="replace")
