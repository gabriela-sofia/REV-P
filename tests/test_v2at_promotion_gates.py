"""v2at - promotion gate decision audit tests."""

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


GATES = [
    "GATE_01_EVENT_ID_EXISTS", "GATE_02_HAZARD_TYPE_TYPED",
    "GATE_03_TEMPORAL_WINDOW_EXISTS", "GATE_04_SENTINEL_OBSERVATION_EXISTS",
    "GATE_05_TIME_DELTA_ACCEPTABLE", "GATE_06_OFFICIAL_OR_VALIDATED_SOURCE_EXISTS",
    "GATE_07_GEOMETRY_AVAILABLE", "GATE_08_PATCH_OVERLAY_AVAILABLE",
    "GATE_09_INTERSECTION_RATIO_ACCEPTABLE", "GATE_10_CONTEXT_ONLY_NOT_PROMOTED",
    "GATE_11_QUICKVIEW_NOT_PROMOTED_ALONE", "GATE_12_BENCHMARK_NOT_LOCAL_TRUTH",
    "GATE_13_CONFLICTS_RESOLVED", "GATE_14_UNCERTAINTY_RECORDED",
    "GATE_15_NO_TRAINING_LABEL_CREATED",
]

REQUIRED_COLUMNS = {
    "decision_id", "package_id", "event_id", "patch_id", "gate_name", "gate_passed",
    "gate_status", "required_condition", "observed_value", "severity",
    "blocking_reason", "recommended_action",
}


def test_gates_required_columns(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset()
    _run(v2at_engine, ds, tmp_path)
    rows = _read(ds / "v2at_promotion_gate_decision_audit.csv")
    assert rows
    assert REQUIRED_COLUMNS.issubset(set(rows[0].keys()))


def test_fifteen_gates_per_package(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset()
    _run(v2at_engine, ds, tmp_path)
    gates = _read(ds / "v2at_promotion_gate_decision_audit.csv")
    packages = _read(ds / "v2at_event_patch_package_registry.csv")
    assert len(gates) == 15 * len(packages)
    by_pkg = {}
    for g in gates:
        by_pkg.setdefault(g["package_id"], set()).add(g["gate_name"])
    for pkg, names in by_pkg.items():
        assert names == set(GATES), f"package {pkg} missing gates: {set(GATES) - names}"


def test_gate15_passes_for_every_package(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset()
    _run(v2at_engine, ds, tmp_path)
    gates = _read(ds / "v2at_promotion_gate_decision_audit.csv")
    g15 = [g for g in gates if g["gate_name"] == "GATE_15_NO_TRAINING_LABEL_CREATED"]
    assert g15
    for g in g15:
        assert g["gate_passed"] == "true"
        assert g["observed_value"] == "no_operational_label_created"


def test_guardrail_gates_always_pass(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset()
    _run(v2at_engine, ds, tmp_path)
    gates = _read(ds / "v2at_promotion_gate_decision_audit.csv")
    for name in ("GATE_10_CONTEXT_ONLY_NOT_PROMOTED", "GATE_11_QUICKVIEW_NOT_PROMOTED_ALONE",
                 "GATE_12_BENCHMARK_NOT_LOCAL_TRUTH"):
        rows = [g for g in gates if g["gate_name"] == name]
        assert rows
        assert all(g["gate_passed"] == "true" for g in rows)


def test_overlay_gate_fails_everywhere(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset()
    _run(v2at_engine, ds, tmp_path)
    gates = _read(ds / "v2at_promotion_gate_decision_audit.csv")
    overlay = [g for g in gates if g["gate_name"] == "GATE_08_PATCH_OVERLAY_AVAILABLE"]
    assert overlay
    assert all(g["gate_passed"] == "false" for g in overlay)


def test_conflict_gate_flags_conflict(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset(include_conflict=True)
    _run(v2at_engine, ds, tmp_path)
    packages = {p["package_id"]: p for p in _read(ds / "v2at_event_patch_package_registry.csv")}
    gates = _read(ds / "v2at_promotion_gate_decision_audit.csv")
    pet_ids = {pid for pid, p in packages.items() if p["region"] == "Petropolis"}
    conflict_gates = [g for g in gates
                      if g["gate_name"] == "GATE_13_CONFLICTS_RESOLVED" and g["package_id"] in pet_ids]
    assert conflict_gates
    assert all(g["gate_passed"] == "false" for g in conflict_gates)


def test_decision_ids_unique_and_stable(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset()
    _run(v2at_engine, ds, tmp_path)
    gates = _read(ds / "v2at_promotion_gate_decision_audit.csv")
    ids = [g["decision_id"] for g in gates]
    assert len(ids) == len(set(ids))
    sample = gates[0]
    expected = v2at_engine.stable_id("DEC_", sample["package_id"], sample["gate_name"])
    assert sample["decision_id"] == expected
