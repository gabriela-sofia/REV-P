from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import mv2_data_06_real_temporal_window_research as r06


def _write(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _synthetic(tmp_path: Path, event_id: str):
    queue = tmp_path / "queue.csv"
    pkg = tmp_path / "pkg.csv"
    temporal = tmp_path / "temporal.csv"
    _write(queue, ["target_rank", "patch_id", "asset_id", "city"], [{"target_rank": "1", "patch_id": "REC_00019", "asset_id": "a1", "city": "Recife"}])
    _write(pkg, ["patch_id", "event_id", "event_window_start", "event_window_end"], [{"patch_id": "REC_00019", "event_id": event_id, "event_window_start": "2022-05-24", "event_window_end": "2022-05-30"}])
    _write(temporal, ["observed_event_id", "event_name"], [{"observed_event_id": event_id, "event_name": "Chuvas extremas Recife 2022"}])
    return queue, pkg, temporal


def test_strong_when_internal_and_external(tmp_path: Path) -> None:
    queue, pkg, temporal = _synthetic(tmp_path, "REC_2022_05_24_30")
    rows = r06.research_targets(queue, pkg, temporal)
    assert rows[0]["evidence_strength"] == "STRONG"
    assert rows[0]["review_status"] == "REAL_TEMPORAL_WINDOW_CANDIDATE"
    assert rows[0]["temporal_window_start"] == "2022-05-24"
    assert rows[0]["source_ref_public"].startswith("https://")


def test_medium_when_internal_only(tmp_path: Path) -> None:
    queue, pkg, temporal = _synthetic(tmp_path, "REC_2099_99_99")  # not in external map
    rows = r06.research_targets(queue, pkg, temporal)
    assert rows[0]["evidence_strength"] == "MEDIUM"
    assert rows[0]["blocked_reason"] == "no_external_official_source"


def test_weak_when_no_internal_linkage(tmp_path: Path) -> None:
    queue = tmp_path / "q.csv"
    pkg = tmp_path / "p.csv"
    temporal = tmp_path / "t.csv"
    _write(queue, ["target_rank", "patch_id", "asset_id", "city"], [{"target_rank": "1", "patch_id": "ZZZ_00001", "asset_id": "a1", "city": "Recife"}])
    _write(pkg, ["patch_id", "event_id", "event_window_start", "event_window_end"], [])
    _write(temporal, ["observed_event_id", "event_name"], [])
    rows = r06.research_targets(queue, pkg, temporal)
    assert rows[0]["evidence_strength"] == "WEAK"
    assert rows[0]["blocked_reason"] == "no_internal_event_linkage"


def test_no_date_without_source(tmp_path: Path) -> None:
    queue, pkg, temporal = _synthetic(tmp_path, "REC_2022_05_24_30")
    rows = r06.research_targets(queue, pkg, temporal)
    for row in rows:
        if row["temporal_window_start"] and row["evidence_strength"] == "STRONG":
            assert row["source_ref_public"], "janela forte sem fonte publica"


def test_local_candidate_written_when_strong(tmp_path: Path) -> None:
    queue, pkg, temporal = _synthetic(tmp_path, "REC_2022_05_24_30")
    rows = r06.research_targets(queue, pkg, temporal)
    dest = tmp_path / "inputs_local" / "data_06.csv"
    assert r06.write_local_candidate_if_strong(rows, dest) is True
    written = list(csv.DictReader(dest.open(encoding="utf-8")))
    assert set(r06.LOCAL_FIELDS).issubset(written[0].keys())
    assert written[0]["temporal_window_start"] == "2022-05-24"
    assert "STW_REC_2022_05_24_30" in written[0]["source_ref"]


def test_real_run_all_ten_strong() -> None:
    rows = r06.research_targets()
    assert len(rows) == 10
    assert all(row["evidence_strength"] == "STRONG" for row in rows)


def test_no_side_effects() -> None:
    summary = r06.summarize(r06.research_targets(), local_created=False)
    assert summary["live_calls"] == 0
    assert summary["downloads"] == 0
    assert summary["rasters"] == 0
    assert summary["crops"] == 0
