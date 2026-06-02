"""Tests for v1oe Recife temporal rematch after date recovery."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1oe_recife_event_patch_temporal_rematch_after_date_recovery.py"
EVENTS = ROOT / "datasets/recife_positive_candidate_date_normalized_registry.csv"
SCENES = ROOT / "datasets/recife_sentinel_scene_date_adjudication_registry.csv"
OUT = ROOT / "datasets/recife_event_patch_temporal_rematch_registry.csv"
UNLOCKED = ROOT / "datasets/recife_event_patch_temporal_unlocked_queue.csv"


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_v1oe_rematches_only_confirmed_scene_dates() -> None:
    write_rows(EVENTS, [{"candidate_id": "C1", "date_parsed": "2022-05-24"}, {"candidate_id": "C2", "date_parsed": "2022-07-01"}])
    write_rows(SCENES, [{"patch_id": "REC_PATCH_A", "scene_date": "2022-05-25", "scene_date_status": "SCENE_DATE_CONFIRMED"}])
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    rows = read_rows(OUT)
    assert rows[0]["temporal_class"] == "TEMPORAL_STRONG"
    assert rows[1]["temporal_class"] == "TEMPORAL_WEAK_OR_BLOCKED"
    assert read_rows(UNLOCKED)[0]["c4_promotion_allowed"] == "false"
