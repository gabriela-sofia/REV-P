"""v2av - patch boundary recovery queue tests."""

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
    "review_item_id", "patch_id", "region", "city", "priority_rank", "priority_reason",
    "missing_fields", "suggested_recovery_action", "candidate_source_files",
    "is_needed_by_packages_count", "is_recife_priority", "blocking_reason",
}


def test_queue_required_columns(v2av_engine, v2av_dataset, tmp_path):
    ds = v2av_dataset()
    _run(v2av_engine, ds, tmp_path)
    rows = _read(ds / "v2av_patch_boundary_recovery_queue.csv")
    assert REQUIRED_COLUMNS.issubset(set(rows[0].keys()))


def test_queue_one_item_per_patch(v2av_engine, v2av_dataset, tmp_path, v2av_make_patch):
    patches = [v2av_make_patch("REC_00205"), v2av_make_patch("PET_00016", "PET", "Petropolis")]
    ds = v2av_dataset(patches=patches)
    _run(v2av_engine, ds, tmp_path)
    queue = _read(ds / "v2av_patch_boundary_recovery_queue.csv")
    assert {q["patch_id"] for q in queue} == {"REC_00205", "PET_00016"}


def test_queue_sorted_by_priority(v2av_engine, v2av_dataset, tmp_path, v2av_make_patch):
    patches = [v2av_make_patch("CUR_00038", "CUR", "Curitiba"),
               v2av_make_patch("REC_00205", "REC", "Recife"),
               v2av_make_patch("PET_00016", "PET", "Petropolis")]
    ds = v2av_dataset(patches=patches)
    _run(v2av_engine, ds, tmp_path)
    queue = _read(ds / "v2av_patch_boundary_recovery_queue.csv")
    ranks = [int(q["priority_rank"]) for q in queue]
    assert ranks == sorted(ranks)


def test_recife_candidate_reference_is_priority_one(v2av_engine, v2av_dataset, tmp_path, v2av_make_patch):
    patches = [v2av_make_patch("CUR_00038", "CUR", "Curitiba"),
               v2av_make_patch("REC_00205", "REC", "Recife")]
    packages = [
        {"package_id": "PKG_rec", "event_id": "REC_2022_05_24_30", "patch_id": "REC_00205",
         "region": "Recife", "city": "Recife", "hazard_type": "urban_flood",
         "allowed_use": "candidate_reference"},
        {"package_id": "PKG_cur", "event_id": "CUR_X", "patch_id": "CUR_00038",
         "region": "Curitiba", "city": "Curitiba", "hazard_type": "urban_flood",
         "allowed_use": "rejected_context_only"},
    ]
    ds = v2av_dataset(patches=patches, packages=packages)
    _run(v2av_engine, ds, tmp_path)
    queue = _read(ds / "v2av_patch_boundary_recovery_queue.csv")
    top = queue[0]
    assert top["patch_id"] == "REC_00205"
    assert top["priority_rank"] == "1"
    assert top["is_recife_priority"] == "true"


def test_missing_fields_recorded(v2av_engine, v2av_dataset, tmp_path):
    ds = v2av_dataset()
    _run(v2av_engine, ds, tmp_path)
    queue = _read(ds / "v2av_patch_boundary_recovery_queue.csv")
    for q in queue:
        assert q["missing_fields"] != ""
        assert q["suggested_recovery_action"] != ""


def test_review_item_ids_unique_and_stable(v2av_engine, v2av_dataset, tmp_path):
    ds = v2av_dataset()
    _run(v2av_engine, ds, tmp_path)
    queue = _read(ds / "v2av_patch_boundary_recovery_queue.csv")
    ids = [q["review_item_id"] for q in queue]
    assert len(ids) == len(set(ids))
    assert all(i.startswith("PBRQ_") for i in ids)
