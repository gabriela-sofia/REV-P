"""Tests for v1nw Recife event-patch temporal matcher."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1nw_recife_event_patch_temporal_matcher.py"
OUT = ROOT / "datasets/recife_event_patch_temporal_match_registry.csv"
QUEUE = ROOT / "datasets/recife_event_patch_temporal_priority_queue.csv"


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_v1nw_classifies_temporal_distance_and_blocks_missing_dates(tmp_path: Path) -> None:
    events = tmp_path / "events.csv"
    inventory = tmp_path / "inventory.csv"
    write_rows(events, [{"candidate_id": "C1", "date_parsed": "2022-05-24"}, {"candidate_id": "C2", "date_parsed": ""}])
    write_rows(inventory, [{"patch_id": "REC_PATCH", "scene_date": "2022-05-26", "scene_date_status": "SENTINEL_DATE_AVAILABLE"}])
    env = os.environ.copy()
    env["REVP_RECIFE_DATE_NORMALIZED_REGISTRY"] = str(events)
    env["REVP_RECIFE_SENTINEL_TEMPORAL_INVENTORY"] = str(inventory)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    rows = read_rows(OUT)
    assert rows[0]["temporal_class"] == "TEMPORAL_STRONG"
    assert rows[1]["temporal_class"] == "TEMPORAL_UNKNOWN_BLOCKED"
    assert read_rows(QUEUE)[0]["c4_promotion_allowed"] == "false"
    assert "can_train_model,true" not in OUT.read_text(encoding="utf-8")
