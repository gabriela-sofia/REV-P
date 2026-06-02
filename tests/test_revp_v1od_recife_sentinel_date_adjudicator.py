"""Tests for v1od Recife Sentinel scene date adjudication."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1od_recife_sentinel_date_adjudicator.py"
SOURCE = ROOT / "datasets/recife_sentinel_date_source_inventory.csv"
FILENAME = ROOT / "datasets/recife_sentinel_filename_date_parse_registry.csv"
METADATA = ROOT / "datasets/recife_sentinel_metadata_date_recovery_registry.csv"
OUT = ROOT / "datasets/recife_sentinel_scene_date_adjudication_registry.csv"
QUALITY = ROOT / "datasets/recife_sentinel_scene_date_quality_matrix.csv"


def write_rows(path: Path, rows: list[dict[str, str]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_v1od_confirms_metadata_and_rejects_event_window_only_date() -> None:
    write_rows(SOURCE, [{"patch_id": "REC_PATCH_A"}, {"patch_id": "REC_2022_05_24_30"}])
    write_rows(FILENAME, [{"patch_id": "REC_2022_05_24_30", "scene_date": "2022-05-24", "date_source": "event_windows.csv", "date_source_type": "EVENT_WINDOW_NOT_SCENE_DATE", "parse_confidence": "LOW"}])
    write_rows(METADATA, [{"patch_id": "REC_PATCH_A", "scene_date": "2022-05-25", "metadata_source": "REC_PATCH_A_metadata.json", "date_confidence": "HIGH"}])
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    rows = {row["patch_id"]: row for row in read_rows(OUT)}
    assert rows["REC_PATCH_A"]["scene_date_status"] == "SCENE_DATE_CONFIRMED"
    assert rows["REC_2022_05_24_30"]["scene_date_status"] == "SENTINEL_DATE_MISSING"
    assert read_rows(QUALITY)[0]["scene_date_confirmed_count"] == "1"
