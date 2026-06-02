"""Tests for v1nv Recife Sentinel temporal inventory."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1nv_recife_sentinel_patch_temporal_inventory.py"
OUT = ROOT / "datasets/recife_sentinel_patch_temporal_inventory.csv"
READINESS = ROOT / "datasets/recife_sentinel_patch_temporal_readiness_matrix.csv"
SCHEMA = ROOT / "datasets/schemas/recife_sentinel_patch_temporal_inventory_schema.csv"


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_v1nv_records_missing_sentinel_dates_without_invention(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_rows(
        manifest,
        [
            {"patch_id": "REC_PATCH_WITH_DATE", "region": "Recife", "scene_date": "2022-05-25", "sensor": "Sentinel-2", "file_role": "POST"},
            {"patch_id": "REC_PATCH_NO_DATE", "region": "Recife", "scene_date": "", "sensor": "Sentinel-2", "file_role": "POST"},
        ],
    )
    env = os.environ.copy()
    env["REVP_RECIFE_SENTINEL_MANIFESTS"] = str(manifest)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    rows = read_rows(OUT)
    statuses = {row["patch_id"]: row["scene_date_status"] for row in rows}
    assert statuses["REC_PATCH_WITH_DATE"] == "SENTINEL_DATE_AVAILABLE"
    assert statuses["REC_PATCH_NO_DATE"] == "SENTINEL_DATE_MISSING"
    assert read_rows(READINESS)[0]["can_create_operational_label"] == "false"
    assert "scene_date_status" in {row["field"] for row in read_rows(SCHEMA)}
