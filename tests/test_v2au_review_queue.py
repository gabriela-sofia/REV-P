"""v2au - overlay review/digitization queue tests."""

from __future__ import annotations

import csv


def _run(engine, ds, tmp_path):
    code, _ = engine.run(dataset_dir=str(ds), output_dir=str(tmp_path / "out"),
                         config_dir=str(tmp_path / "cfg"))
    assert code == 0


def _read(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


REQUIRED_COLUMNS = {
    "review_item_id", "package_id", "event_id", "patch_id", "region", "city",
    "hazard_type", "priority_rank", "priority_reason", "missing_geometry_type",
    "suggested_action", "evidence_score", "overlay_status", "remaining_blocking_reason",
}


def test_queue_required_columns(v2au_engine, v2au_dataset, tmp_path):
    ds = v2au_dataset()
    _run(v2au_engine, ds, tmp_path)
    rows = _read(ds / "v2au_overlay_review_queue.csv")
    assert REQUIRED_COLUMNS.issubset(set(rows[0].keys()))


def test_queue_one_item_per_package(v2au_engine, v2au_dataset, tmp_path, v2au_make_package):
    pkgs = [v2au_make_package("PKG_a", "E1", "P1"), v2au_make_package("PKG_b", "E2", "P2")]
    ds = v2au_dataset(packages=pkgs)
    _run(v2au_engine, ds, tmp_path)
    queue = _read(ds / "v2au_overlay_review_queue.csv")
    overlays = _read(ds / "v2au_patch_event_overlay_registry.csv")
    assert len(queue) == len(overlays)


def test_queue_sorted_by_priority(v2au_engine, v2au_dataset, tmp_path):
    ds = v2au_dataset()
    _run(v2au_engine, ds, tmp_path)
    queue = _read(ds / "v2au_overlay_review_queue.csv")
    ranks = [int(q["priority_rank"]) for q in queue]
    assert ranks == sorted(ranks)


def test_recife_candidate_reference_is_priority_one(v2au_engine, v2au_dataset, tmp_path, v2au_make_package):
    pkgs = [
        v2au_make_package("PKG_rec", "REC_2022_05_24_30", "REC_00205", region="Recife",
                          allowed_use="candidate_reference"),
        v2au_make_package("PKG_other", "PET_2022_02_15", "PET_00016", region="Petropolis",
                          allowed_use="secondary_evaluation_candidate", has_spatial_support="false",
                          urban_context="false", evidence_score="0.4"),
    ]
    ds = v2au_dataset(packages=pkgs)
    _run(v2au_engine, ds, tmp_path)
    queue = _read(ds / "v2au_overlay_review_queue.csv")
    top = queue[0]
    assert top["region"] == "Recife"
    assert top["priority_rank"] == "1"
    assert "missing_geometry_type" in top and top["missing_geometry_type"] != "none"


def test_missing_geometry_type_recorded(v2au_engine, v2au_dataset, tmp_path):
    ds = v2au_dataset()
    _run(v2au_engine, ds, tmp_path)
    queue = _read(ds / "v2au_overlay_review_queue.csv")
    for q in queue:
        assert q["missing_geometry_type"] != ""
        assert q["suggested_action"] != ""


def test_review_item_ids_unique_and_stable(v2au_engine, v2au_dataset, tmp_path):
    ds = v2au_dataset()
    _run(v2au_engine, ds, tmp_path)
    queue = _read(ds / "v2au_overlay_review_queue.csv")
    ids = [q["review_item_id"] for q in queue]
    assert len(ids) == len(set(ids))
    assert all(i.startswith("OVRQ_") for i in ids)
