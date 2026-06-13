"""v2aw - Recife P1 geometry readiness tests."""

from __future__ import annotations

import csv


def _run(engine, ds, tmp_path):
    code, summary = engine.run(dataset_dir=str(ds), output_dir=str(tmp_path / "out"),
                               config_dir=str(tmp_path / "cfg"))
    assert code == 0
    return summary


def _read(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_recife_readiness_blocks_when_geometries_missing(v2aw_engine, v2aw_dataset, tmp_path):
    ds = v2aw_dataset(n_recife=55)
    summary = _run(v2aw_engine, ds, tmp_path)
    rows = _read(ds / "v2aw_recife_p1_geometry_readiness.csv")
    assert len(rows) == 55
    assert all(row["ready_for_v2av"] == "false" for row in rows)
    assert all(row["ready_for_v2au_overlay"] == "false" for row in rows)
    assert all(row["remaining_blocker"] == "NO_PATCH_BOUNDARY_SOURCE_PROVIDED" for row in rows)
    assert summary["ready_for_v2av_count"] == 0
    assert summary["ready_for_v2au_count"] == 0


def test_valid_patch_boundary_only_readies_v2av(v2aw_engine, v2aw_dataset, tmp_path,
                                                v2aw_make_patch_source):
    ds = v2aw_dataset(n_recife=1, provided_patch=[v2aw_make_patch_source()])
    _run(v2aw_engine, ds, tmp_path)
    row = _read(ds / "v2aw_recife_p1_geometry_readiness.csv")[0]
    assert row["patch_boundary_valid"] == "true"
    assert row["ready_for_v2av"] == "true"
    assert row["ready_for_v2au_overlay"] == "false"
    assert row["remaining_blocker"] == "NO_VALID_OBSERVED_EVENT_POLYGON"


def test_valid_patch_and_event_polygon_ready_for_overlay(v2aw_engine, v2aw_dataset, tmp_path,
                                                         v2aw_make_patch_source,
                                                         v2aw_make_event_source):
    ds = v2aw_dataset(n_recife=1, provided_patch=[v2aw_make_patch_source()],
                      provided_event=[v2aw_make_event_source()])
    summary = _run(v2aw_engine, ds, tmp_path)
    row = _read(ds / "v2aw_recife_p1_geometry_readiness.csv")[0]
    assert row["ready_for_v2av"] == "true"
    assert row["ready_for_v2au_overlay"] == "true"
    assert row["remaining_blocker"] == ""
    assert summary["ready_for_v2av_count"] == 1
    assert summary["ready_for_v2au_count"] == 1


def test_cprm_event_point_stays_non_overlay(v2aw_engine, v2aw_dataset, tmp_path, v2aw_cprm_event):
    ds = v2aw_dataset(n_recife=1, ground_events=[v2aw_cprm_event()])
    summary = _run(v2aw_engine, ds, tmp_path)
    validation = _read(ds / "v2aw_geometry_source_validation_registry.csv")
    assert len(validation) == 1
    assert validation[0]["blocking_reason"] == "POINT_ANCHOR_NOT_OVERLAY"
    assert summary["event_point_anchors_seeded"] == 1
    assert summary["event_sources_valid"] == 0
