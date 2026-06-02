"""Tests for v1oc Recife Sentinel metadata sidecar recovery."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1oc_recife_sentinel_metadata_sidecar_date_recovery.py"
OUT = ROOT / "datasets/recife_sentinel_metadata_date_recovery_registry.csv"
CONFLICT = ROOT / "datasets/recife_sentinel_metadata_date_conflict_registry.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_v1oc_recovers_sidecar_date_without_pixel_read_or_absolute_path(tmp_path: Path) -> None:
    sidecar = tmp_path / "REC_PATCH_A_metadata.json"
    sidecar.write_text('{"sensing_time":"2022-05-25T13:12:41Z"}', encoding="utf-8")
    env = os.environ.copy()
    env["REVP_RECIFE_SENTINEL_LOCAL_SCAN_DIRS"] = str(tmp_path)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    row = read_rows(OUT)[0]
    assert row["scene_date"] == "2022-05-25"
    assert row["pixel_read_status"] == "NOT_READ"
    assert "C:\\" not in OUT.read_text(encoding="utf-8", errors="replace")
    assert CONFLICT.exists()
