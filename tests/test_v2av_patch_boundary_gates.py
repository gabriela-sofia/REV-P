"""v2av - patch boundary build audit gate tests."""

from __future__ import annotations

import csv


def _run(engine, ds, tmp_path):
    code, _ = engine.run(dataset_dir=str(ds), output_dir=str(tmp_path / "out"),
                         config_dir=str(tmp_path / "cfg"))
    assert code == 0


def _read(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


GATES = [
    "PATCH_BOUNDARY_GATE_01_PATCH_ID_EXISTS", "PATCH_BOUNDARY_GATE_02_SOURCE_METADATA_FOUND",
    "PATCH_BOUNDARY_GATE_03_CRS_EXISTS", "PATCH_BOUNDARY_GATE_04_CRS_ACCEPTED",
    "PATCH_BOUNDARY_GATE_05_BOUNDARY_METHOD_ALLOWED", "PATCH_BOUNDARY_GATE_06_GEOMETRY_VALID",
    "PATCH_BOUNDARY_GATE_07_AREA_COMPUTABLE", "PATCH_BOUNDARY_GATE_08_NOT_BUILT_FROM_UNVERIFIED_DEFAULT",
    "PATCH_BOUNDARY_GATE_09_NO_EVENT_LABEL_CREATED", "PATCH_BOUNDARY_GATE_10_READY_FOR_V2AU_OVERLAY",
]

REQUIRED_COLUMNS = {
    "decision_id", "patch_id", "gate_name", "gate_passed", "gate_status",
    "required_condition", "observed_value", "severity", "blocking_reason", "recommended_action",
}


def test_gates_required_columns(v2av_engine, v2av_dataset, tmp_path):
    ds = v2av_dataset()
    _run(v2av_engine, ds, tmp_path)
    rows = _read(ds / "v2av_patch_boundary_build_audit.csv")
    assert REQUIRED_COLUMNS.issubset(set(rows[0].keys()))


def test_ten_gates_per_patch(v2av_engine, v2av_dataset, tmp_path, v2av_make_patch):
    patches = [v2av_make_patch("REC_00205"), v2av_make_patch("PET_00016", "PET", "Petropolis")]
    ds = v2av_dataset(patches=patches)
    _run(v2av_engine, ds, tmp_path)
    audit = _read(ds / "v2av_patch_boundary_build_audit.csv")
    assert len(audit) == 10 * 2
    by_patch = {}
    for g in audit:
        by_patch.setdefault(g["patch_id"], set()).add(g["gate_name"])
    for names in by_patch.values():
        assert names == set(GATES)


def test_gate09_no_label_always_passes(v2av_engine, v2av_dataset, tmp_path, v2av_make_geom_source):
    geoms = [v2av_make_geom_source("REC_00205", "bbox", "0,0,10,10", crs="EPSG:3857")]
    ds = v2av_dataset(geometry_sources=geoms)
    _run(v2av_engine, ds, tmp_path)
    audit = _read(ds / "v2av_patch_boundary_build_audit.csv")
    g09 = [g for g in audit if g["gate_name"] == "PATCH_BOUNDARY_GATE_09_NO_EVENT_LABEL_CREATED"]
    assert g09 and all(g["gate_passed"] == "true" for g in g09)


def test_source_gate_fails_without_metadata(v2av_engine, v2av_dataset, tmp_path):
    ds = v2av_dataset()
    _run(v2av_engine, ds, tmp_path)
    audit = _read(ds / "v2av_patch_boundary_build_audit.csv")
    g02 = [g for g in audit if g["gate_name"] == "PATCH_BOUNDARY_GATE_02_SOURCE_METADATA_FOUND"]
    assert g02 and all(g["gate_passed"] == "false" for g in g02)


def test_all_gates_pass_on_built_boundary(v2av_engine, v2av_dataset, tmp_path, v2av_make_geom_source):
    geoms = [v2av_make_geom_source("REC_00205", "bbox", "0,0,10,10", crs="EPSG:3857")]
    ds = v2av_dataset(geometry_sources=geoms)
    _run(v2av_engine, ds, tmp_path)
    audit = _read(ds / "v2av_patch_boundary_build_audit.csv")
    for g in audit:
        assert g["gate_passed"] == "true", f"{g['gate_name']} should pass on a built boundary"


def test_decision_ids_unique_and_stable(v2av_engine, v2av_dataset, tmp_path):
    ds = v2av_dataset()
    _run(v2av_engine, ds, tmp_path)
    audit = _read(ds / "v2av_patch_boundary_build_audit.csv")
    ids = [g["decision_id"] for g in audit]
    assert len(ids) == len(set(ids))
    s = audit[0]
    assert s["decision_id"] == v2av_engine.stable_id("PBD_", s["patch_id"], s["gate_name"])
