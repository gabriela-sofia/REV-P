"""v2at - event-patch package engine tests."""

from __future__ import annotations

import csv


def _run(engine, dataset_dir, tmp_path, sub="out"):
    out = tmp_path / sub
    cfg = tmp_path / "cfg"
    code, summary = engine.run(dataset_dir=str(dataset_dir), output_dir=str(out), config_dir=str(cfg))
    assert code == 0, f"engine returned {code}"
    return summary, out


def _read(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


REQUIRED_COLUMNS = {
    "package_id", "event_id", "patch_id", "region", "city", "hazard_type",
    "sentinel_asset_id", "sentinel_sensor_family", "sentinel_observation_date",
    "event_window_start", "event_window_end", "time_delta_days", "has_temporal_anchor",
    "has_spatial_support", "has_official_source", "has_vhr_support",
    "has_only_contextual_sources", "has_geometry", "has_patch_overlay",
    "intersection_ratio", "valid_data_fraction", "urban_context", "permanent_water_risk",
    "occlusion_risk", "evidence_count", "strong_evidence_count", "weak_evidence_count",
    "conflict_count", "evidence_score", "uncertainty_score", "promotion_candidate_level",
    "promotion_decision", "blocking_reason", "allowed_use", "notes",
}

ALL_OUTPUTS = [
    "v2at_external_evidence_source_catalog.csv",
    "v2at_evidence_observation_registry.csv",
    "v2at_event_patch_package_registry.csv",
    "v2at_promotion_gate_decision_audit.csv",
    "v2at_reviewer_queue_seed.csv",
    "v2at_operational_label_blocklist.csv",
]


def test_all_v2at_csvs_generated(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset()
    _run(v2at_engine, ds, tmp_path)
    for name in ALL_OUTPUTS:
        assert (ds / name).exists(), f"missing output {name}"


def test_package_required_columns(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset()
    _run(v2at_engine, ds, tmp_path)
    rows = _read(ds / "v2at_event_patch_package_registry.csv")
    assert rows
    assert REQUIRED_COLUMNS.issubset(set(rows[0].keys()))


def test_missing_geometry_blocks_c4(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset()
    _run(v2at_engine, ds, tmp_path)
    rows = _read(ds / "v2at_event_patch_package_registry.csv")
    # No patch-event overlay is ever computed, so no package may reach C4_CANDIDATE.
    for r in rows:
        assert r["has_patch_overlay"] == "false"
        assert r["promotion_candidate_level"] != "C4"
    assert not any(r["promotion_candidate_level"] == "C4_CANDIDATE" for r in rows)


def test_missing_temporal_window_blocks_strong_promotion(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset()
    _run(v2at_engine, ds, tmp_path)
    rows = _read(ds / "v2at_event_patch_package_registry.csv")
    cur = [r for r in rows if r["region"] == "Curitiba"]
    assert cur, "expected a Curitiba package"
    for r in cur:
        assert r["event_window_start"] == "UNKNOWN"
        assert r["allowed_use"] in {"rejected_context_only", "review_only"}
        assert r["promotion_candidate_level"] in {"C0", "C1"}


def test_official_temporal_without_overlay_caps_at_c3(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset()
    _run(v2at_engine, ds, tmp_path)
    rows = _read(ds / "v2at_event_patch_package_registry.csv")
    rec = [r for r in rows if r["region"] == "Recife"]
    assert rec
    for r in rec:
        assert r["promotion_candidate_level"] == "C3"
        assert r["allowed_use"] == "candidate_reference"
        assert r["blocking_reason"] == "NO_PATCH_EVENT_OVERLAY_GEOMETRY"


def test_contextual_only_region_not_promoted_to_reference(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset(context_only_region=True, include_curitiba_missing=True)
    _run(v2at_engine, ds, tmp_path)
    rows = _read(ds / "v2at_event_patch_package_registry.csv")
    cur = [r for r in rows if r["region"] == "Curitiba"]
    assert cur
    for r in cur:
        assert r["allowed_use"] != "candidate_reference"


def test_conflict_requires_review(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset(include_conflict=True)
    _run(v2at_engine, ds, tmp_path)
    rows = _read(ds / "v2at_event_patch_package_registry.csv")
    pet = [r for r in rows if r["region"] == "Petropolis"]
    assert pet
    for r in pet:
        assert int(r["conflict_count"]) >= 1
        assert r["promotion_decision"] == "CONFLICT_REVIEW_REQUIRED"
        assert r["allowed_use"] == "review_only"


def test_determinism_two_runs_byte_identical(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset()
    _run(v2at_engine, ds, tmp_path, sub="run1")
    first = {name: (ds / name).read_bytes() for name in ALL_OUTPUTS}
    _run(v2at_engine, ds, tmp_path, sub="run2")
    second = {name: (ds / name).read_bytes() for name in ALL_OUTPUTS}
    for name in ALL_OUTPUTS:
        assert first[name] == second[name], f"{name} not deterministic"


def test_stable_package_ids(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset()
    _run(v2at_engine, ds, tmp_path)
    rows = _read(ds / "v2at_event_patch_package_registry.csv")
    ids = [r["package_id"] for r in rows]
    assert all(i.startswith("PKG_") and len(i) == 16 for i in ids)
    assert len(ids) == len(set(ids)), "package_id must be unique"
    # Recompute one id with the engine's stable hash and compare.
    rec = next(r for r in rows if r["region"] == "Recife")
    expected = v2at_engine.stable_id("PKG_", rec["region"], rec["event_id"], rec["patch_id"])
    assert rec["package_id"] == expected


def test_rows_sorted_by_stable_keys(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset()
    _run(v2at_engine, ds, tmp_path)
    rows = _read(ds / "v2at_event_patch_package_registry.csv")
    keys = [(r["region"], r["event_id"], r["patch_id"]) for r in rows]
    assert keys == sorted(keys)
