"""v2az - manual validation and normalized feed tests."""

from __future__ import annotations

import csv

import scripts.v2az_turning_point_replay_orchestrator as engine
from v2az_test_helpers import make_dataset


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def run(tmp_path, **kwargs):
    ds, out, cfg, docs = make_dataset(tmp_path, **kwargs)
    code, summary = engine.run("dry_run", str(ds), str(out), str(cfg), str(docs))
    assert code == 0
    return ds, summary


def test_missing_inputs_block_and_feeds_are_empty(tmp_path):
    ds, summary = run(tmp_path)
    snapshot = read(ds / "v2az_manual_intake_validation_snapshot.csv")
    assert len(snapshot) == 2
    assert all(row["blocking_reason"] == "BLOCKED_PENDING_MANUAL_GEOMETRY" for row in snapshot)
    assert summary["feed_v2aw_rows"] == summary["feed_v2av_rows"] == summary["feed_v2au_rows"] == 0


def test_valid_patch_bbox_enables_v2av_and_tp1(tmp_path):
    ds, summary = run(tmp_path, valid_patch=True)
    assert len(read(ds / "v2az_feed_v2av_patch_geometry_sources.csv")) == 1
    assert summary["turning_point_level"] == "TP1_ONE_PATCH_BOUNDARY_VALIDATED"


def test_valid_event_polygon_enables_event_feeds_and_tp2(tmp_path):
    ds, summary = run(tmp_path, valid_event=True)
    assert len(read(ds / "v2az_feed_v2aw_event_sources.csv")) == 1
    assert len(read(ds / "v2az_feed_v2au_geometry_sources.csv")) == 1
    assert summary["turning_point_level"] == "TP2_ONE_EVENT_POLYGON_VALIDATED"


def test_event_point_is_not_promoted_to_overlay(tmp_path):
    ds, summary = run(tmp_path, event_point=True)
    event = next(row for row in read(ds / "v2az_manual_intake_validation_snapshot.csv")
                 if row["target_type"] == "event")
    assert event["blocking_reason"] == "POINT_ANCHOR_NOT_OVERLAY"
    assert summary["valid_event_polygons"] == 0
