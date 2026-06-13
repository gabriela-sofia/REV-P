"""v2az - TP0 through TP4 progress tests."""

from __future__ import annotations

import csv

import scripts.v2az_turning_point_replay_orchestrator as engine
from v2az_test_helpers import make_dataset


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_canonical_progress_has_only_tp0_passed():
    progress = read("datasets/v2az_turning_point_progress_registry.csv")
    assert progress[0]["turning_point_level"] == "TP0_DOCUMENTED_ABSENCE"
    assert progress[0]["gate_passed"] == "true"
    assert all(row["gate_passed"] == "false" for row in progress[1:])


def test_valid_paired_patch_and_event_pass_tp3_but_not_tp4(tmp_path):
    ds, out, cfg, docs = make_dataset(tmp_path, valid_patch=True, valid_event=True)
    code, summary = engine.run("dry_run", str(ds), str(out), str(cfg), str(docs))
    assert code == 0
    assert summary["valid_patch_event_pairs"] == 1
    assert summary["turning_point_level"] == "TP3_ONE_PATCH_EVENT_PAIR_READY_FOR_OVERLAY"
    progress = {row["turning_point_level"]: row for row in read(ds / "v2az_turning_point_progress_registry.csv")}
    assert progress["TP3_ONE_PATCH_EVENT_PAIR_READY_FOR_OVERLAY"]["gate_passed"] == "true"
    assert progress["TP4_ONE_OVERLAY_CONFIRMED_REQUIRES_HUMAN_REVIEW"]["gate_passed"] == "false"


def test_unpaired_valid_geometries_do_not_pass_tp3(tmp_path):
    ds, out, cfg, docs = make_dataset(tmp_path, valid_patch=True, valid_event=True,
                                      package_patch="REC_OTHER")
    code, summary = engine.run("dry_run", str(ds), str(out), str(cfg), str(docs))
    assert code == 0
    assert summary["valid_patch_boundaries"] == 1 and summary["valid_event_polygons"] == 1
    assert summary["valid_patch_event_pairs"] == 0
    assert summary["turning_point_level"] == "TP2_ONE_EVENT_POLYGON_VALIDATED"
