"""v2at - reviewer queue seed tests."""

from __future__ import annotations

import csv


def _run(engine, dataset_dir, tmp_path):
    out = tmp_path / "out"
    cfg = tmp_path / "cfg"
    code, summary = engine.run(dataset_dir=str(dataset_dir), output_dir=str(out), config_dir=str(cfg))
    assert code == 0
    return summary, out


def _read(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


REQUIRED_COLUMNS = {
    "review_item_id", "package_id", "event_id", "patch_id", "region", "city",
    "hazard_type", "priority_rank", "priority_reason", "suggested_review_action",
    "evidence_score", "uncertainty_score", "blocking_reason",
    "nearest_dino_neighbors_available", "notes",
}


def test_queue_required_columns(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset()
    _run(v2at_engine, ds, tmp_path)
    rows = _read(ds / "v2at_reviewer_queue_seed.csv")
    assert rows
    assert REQUIRED_COLUMNS.issubset(set(rows[0].keys()))


def test_queue_has_one_item_per_package(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset()
    _run(v2at_engine, ds, tmp_path)
    queue = _read(ds / "v2at_reviewer_queue_seed.csv")
    packages = _read(ds / "v2at_event_patch_package_registry.csv")
    assert len(queue) == len(packages)
    assert {q["package_id"] for q in queue} == {p["package_id"] for p in packages}


def test_queue_sorted_by_priority(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset()
    _run(v2at_engine, ds, tmp_path)
    queue = _read(ds / "v2at_reviewer_queue_seed.csv")
    ranks = [int(q["priority_rank"]) for q in queue]
    assert ranks == sorted(ranks)


def test_priority_one_is_temporal_without_geometry(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset()
    _run(v2at_engine, ds, tmp_path)
    queue = _read(ds / "v2at_reviewer_queue_seed.csv")
    top = [q for q in queue if int(q["priority_rank"]) == 1]
    assert top, "expected at least one priority-1 item (good temporal, missing geometry)"
    for q in top:
        assert "geometry" in q["priority_reason"].lower() or "overlay" in q["priority_reason"].lower()


def test_dino_availability_recorded(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset()
    _run(v2at_engine, ds, tmp_path)
    queue = _read(ds / "v2at_reviewer_queue_seed.csv")
    assert all(q["nearest_dino_neighbors_available"] in {"true", "false"} for q in queue)
    # DINO must stay review-only in the queue notes.
    assert all("review-only" in q["notes"].lower() or "review only" in q["notes"].lower()
               for q in queue)


def test_review_item_ids_unique_and_stable(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset()
    _run(v2at_engine, ds, tmp_path)
    queue = _read(ds / "v2at_reviewer_queue_seed.csv")
    ids = [q["review_item_id"] for q in queue]
    assert len(ids) == len(set(ids))
    assert all(i.startswith("RQ_") for i in ids)
