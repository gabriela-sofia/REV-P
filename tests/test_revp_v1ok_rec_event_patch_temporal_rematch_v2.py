"""Tests for v1ok REC temporal rematch v2.

All I/O is redirected to tmp_path — datasets/ is never touched.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1ok_rec_event_patch_temporal_rematch_v2.py"
DATASETS = ROOT / "datasets"


def _write(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fnames = fields or (list(rows[0].keys()) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fnames)
        w.writeheader()
        w.writerows(rows)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_v1ok_separates_confirmed_and_probable_review_only_dates(tmp_path: Path) -> None:
    in_events = tmp_path / "events.csv"
    in_scenes = tmp_path / "scenes.csv"
    out_rematch = tmp_path / "rematch.csv"
    out_queue = tmp_path / "queue.csv"
    out_summary = tmp_path / "summary.csv"

    # Single confirmed scene — avoids tie-breaking ambiguity
    _write(in_events, [{"candidate_id": "RECIFE_POSITIVE_CANDIDATE_V1NR_00001",
                        "date_parsed": "2022-05-24"}])
    _write(in_scenes, [
        {"patch_id": "RECIFE_SENTINEL_PATCH_S2A_001",
         "scene_date": "2022-05-25",
         "scene_date_status": "SCENE_DATE_CONFIRMED"},
    ])

    env = {**os.environ,
           "REVP_V1OK_IN_EVENTS": str(in_events),
           "REVP_V1OK_IN_SCENES": str(in_scenes),
           "REVP_V1OK_OUT_REMATCH": str(out_rematch),
           "REVP_V1OK_OUT_QUEUE": str(out_queue),
           "REVP_V1OK_OUT_SUMMARY": str(out_summary),
           "REVP_V1OK_SCHEMA_REMATCH": str(tmp_path / "s_rematch.csv"),
           "REVP_V1OK_SCHEMA_QUEUE": str(tmp_path / "s_queue.csv"),
           "REVP_V1OK_SCHEMA_SUMMARY": str(tmp_path / "s_summary.csv"),
           "REVP_V1OK_DOC": str(tmp_path / "doc.md")}

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--force", "--emit-evidence"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    rematch_rows = _read(out_rematch)
    assert rematch_rows, "Expected at least one rematch row"
    assert rematch_rows[0]["temporal_class"] == "TEMPORAL_STRONG"
    queue_rows = _read(out_queue)
    assert queue_rows, "Expected at least one queue row"
    assert queue_rows[0]["eligible_for_c3_plus"] == "true"
    assert queue_rows[0]["c4_promotion_allowed"] == "false"

    # Verify the test outputs are in tmp_path, not in real datasets/
    assert out_rematch.exists(), "Output must be in tmp_path"
    assert not (DATASETS / "recife_event_patch_temporal_rematch_v2_registry.csv").samefile(out_rematch), \
        "Test must not write to real datasets/"
