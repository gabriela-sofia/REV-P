"""v2au - overlay gate decision audit tests."""

from __future__ import annotations

import csv


def _run(engine, ds, tmp_path):
    code, _ = engine.run(dataset_dir=str(ds), output_dir=str(tmp_path / "out"),
                         config_dir=str(tmp_path / "cfg"))
    assert code == 0
    return code


def _read(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


GATES = [
    "OVERLAY_GATE_01_PATCH_GEOMETRY_EXISTS", "OVERLAY_GATE_02_EVENT_GEOMETRY_EXISTS",
    "OVERLAY_GATE_03_PATCH_CRS_KNOWN", "OVERLAY_GATE_04_EVENT_CRS_KNOWN",
    "OVERLAY_GATE_05_GEOMETRIES_VALID", "OVERLAY_GATE_06_AREA_COMPUTABLE",
    "OVERLAY_GATE_07_INTERSECTION_COMPUTABLE", "OVERLAY_GATE_08_INTERSECTION_RATIO_ACCEPTABLE",
    "OVERLAY_GATE_09_CONTEXT_GEOMETRY_NOT_PROMOTED", "OVERLAY_GATE_10_POINT_ANCHOR_NOT_OVERLAY",
    "OVERLAY_GATE_11_NO_OPERATIONAL_LABEL_CREATED", "OVERLAY_GATE_12_HUMAN_REVIEW_REQUIRED_FOR_C4",
]

REQUIRED_COLUMNS = {
    "decision_id", "package_id", "event_id", "patch_id", "gate_name", "gate_passed",
    "gate_status", "required_condition", "observed_value", "severity",
    "blocking_reason", "recommended_action",
}


def _overlap(make_geom, event_id, patch_id):
    return [
        make_geom("patch_boundary", "bbox", "0,0,10,10", linked_patch_id=patch_id),
        make_geom("event_observed_geometry", "bbox", "5,5,15,15", linked_event_id=event_id),
    ]


def test_gates_required_columns(v2au_engine, v2au_dataset, tmp_path):
    ds = v2au_dataset()
    _run(v2au_engine, ds, tmp_path)
    rows = _read(ds / "v2au_overlay_gate_decision_audit.csv")
    assert REQUIRED_COLUMNS.issubset(set(rows[0].keys()))


def test_twelve_gates_per_package(v2au_engine, v2au_dataset, tmp_path):
    ds = v2au_dataset()
    _run(v2au_engine, ds, tmp_path)
    gates = _read(ds / "v2au_overlay_gate_decision_audit.csv")
    overlays = _read(ds / "v2au_patch_event_overlay_registry.csv")
    assert len(gates) == 12 * len(overlays)
    by_pkg = {}
    for g in gates:
        by_pkg.setdefault(g["package_id"], set()).add(g["gate_name"])
    for names in by_pkg.values():
        assert names == set(GATES)


def test_gate11_no_label_always_passes(v2au_engine, v2au_dataset, tmp_path, v2au_make_package, v2au_make_geom):
    pkgs = [v2au_make_package("PKG_g11", "E1", "P1")]
    ds = v2au_dataset(packages=pkgs, geometry_sources=_overlap(v2au_make_geom, "E1", "P1"))
    _run(v2au_engine, ds, tmp_path)
    gates = _read(ds / "v2au_overlay_gate_decision_audit.csv")
    g11 = [g for g in gates if g["gate_name"] == "OVERLAY_GATE_11_NO_OPERATIONAL_LABEL_CREATED"]
    assert g11 and all(g["gate_passed"] == "true" for g in g11)


def test_guardrail_gates_always_pass(v2au_engine, v2au_dataset, tmp_path):
    ds = v2au_dataset()
    _run(v2au_engine, ds, tmp_path)
    gates = _read(ds / "v2au_overlay_gate_decision_audit.csv")
    for name in ("OVERLAY_GATE_09_CONTEXT_GEOMETRY_NOT_PROMOTED",
                 "OVERLAY_GATE_10_POINT_ANCHOR_NOT_OVERLAY",
                 "OVERLAY_GATE_11_NO_OPERATIONAL_LABEL_CREATED",
                 "OVERLAY_GATE_12_HUMAN_REVIEW_REQUIRED_FOR_C4"):
        rows = [g for g in gates if g["gate_name"] == name]
        assert rows and all(g["gate_passed"] == "true" for g in rows)


def test_patch_geometry_gate_fails_when_missing(v2au_engine, v2au_dataset, tmp_path):
    ds = v2au_dataset()  # no geometry sources
    _run(v2au_engine, ds, tmp_path)
    gates = _read(ds / "v2au_overlay_gate_decision_audit.csv")
    g01 = [g for g in gates if g["gate_name"] == "OVERLAY_GATE_01_PATCH_GEOMETRY_EXISTS"]
    assert g01 and all(g["gate_passed"] == "false" for g in g01)


def test_all_overlay_gates_pass_on_confirmed(v2au_engine, v2au_dataset, tmp_path, v2au_make_package, v2au_make_geom):
    pkgs = [v2au_make_package("PKG_ok", "E1", "P1")]
    ds = v2au_dataset(packages=pkgs, geometry_sources=_overlap(v2au_make_geom, "E1", "P1"))
    _run(v2au_engine, ds, tmp_path)
    gates = _read(ds / "v2au_overlay_gate_decision_audit.csv")
    for g in gates:
        assert g["gate_passed"] == "true", f"{g['gate_name']} should pass on a confirmed overlay"


def test_decision_ids_unique_and_stable(v2au_engine, v2au_dataset, tmp_path):
    ds = v2au_dataset()
    _run(v2au_engine, ds, tmp_path)
    gates = _read(ds / "v2au_overlay_gate_decision_audit.csv")
    ids = [g["decision_id"] for g in gates]
    assert len(ids) == len(set(ids))
    sample = gates[0]
    assert sample["decision_id"] == v2au_engine.stable_id("OVD_", sample["package_id"], sample["gate_name"])
