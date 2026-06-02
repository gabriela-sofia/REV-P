"""Tests for v1ob Recife Sentinel filename date parser."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1ob_recife_sentinel_filename_date_parser.py"
SOURCE = ROOT / "datasets/recife_sentinel_date_source_inventory.csv"
OUT = ROOT / "datasets/recife_sentinel_filename_date_parse_registry.csv"
CONFLICT = ROOT / "datasets/recife_sentinel_filename_date_conflict_registry.csv"


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_v1ob_parses_sentinel_filename_and_marks_event_window_not_scene_date() -> None:
    write_rows(
        SOURCE,
        [
            {"source_inventory_id": "S1", "patch_id": "REC_PATCH_A", "source_path": "manifest.csv", "source_field": "asset_path_reference", "candidate_text": "S2A_MSIL2A_20220525T131241_T25LDD_patch.tif"},
            {"source_inventory_id": "S2", "patch_id": "REC_2022_05_24_30", "source_path": "datasets/event_sentinel_temporal_window_registry.csv", "source_field": "observed_event_id", "candidate_text": "REC_2022_05_24_30 event_window"},
        ],
    )
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    rows = read_rows(OUT)
    assert any(row["scene_date"] == "2022-05-25" and row["parse_confidence"] == "HIGH" for row in rows)
    assert any(row["date_source_type"] == "EVENT_WINDOW_NOT_SCENE_DATE" for row in rows)
    assert "can_create_operational_label,true" not in OUT.read_text(encoding="utf-8")
    assert CONFLICT.exists()
