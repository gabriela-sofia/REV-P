"""v2at - external evidence source catalog tests."""

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


REQUIRED_SOURCES = {
    "ANA_HIDROWEB", "ANA_TELEMETRY", "INMET_HISTORICAL", "CEMADEN_MONITORING",
    "CEMADEN_BULLETIN", "SGB_RISK_CARTOGRAPHY", "SGB_SUSCEPTIBILITY",
    "S2ID_DISASTER_RECORD", "ATLAS_DIGITAL_DESASTRES", "COPERNICUS_EMS_MAPPING",
    "COPERNICUS_GFM", "INTERNATIONAL_CHARTER_PRODUCT", "INTERNATIONAL_CHARTER_QUICKVIEW",
    "VANTOR_OPEN_DATA", "PLANET_DISASTER_DATA", "URBANSARFLOODS_BENCHMARK",
    "SEN1FLOODS11_BENCHMARK", "EMDAT_CONTEXT", "MEDIA_CONTEXT", "SOCIAL_CONTEXT",
}

REQUIRED_COLUMNS = {
    "source_id", "source_name", "source_class", "institution_type", "country_scope",
    "spatial_role", "temporal_role", "geometry_role", "license_status", "access_mode",
    "expected_data_type", "evidence_weight", "can_open_candidate", "can_promote_alone", "notes",
}


def test_catalog_has_required_columns(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset()
    _run(v2at_engine, ds, tmp_path)
    rows = _read(ds / "v2at_external_evidence_source_catalog.csv")
    assert rows
    assert REQUIRED_COLUMNS.issubset(set(rows[0].keys()))


def test_catalog_contains_canonical_sources(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset()
    _run(v2at_engine, ds, tmp_path)
    rows = _read(ds / "v2at_external_evidence_source_catalog.csv")
    ids = {r["source_id"] for r in rows}
    missing = REQUIRED_SOURCES - ids
    assert not missing, f"missing canonical sources: {missing}"


def test_quickview_media_social_emdat_cannot_promote_alone(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset()
    _run(v2at_engine, ds, tmp_path)
    by_id = {r["source_id"]: r for r in _read(ds / "v2at_external_evidence_source_catalog.csv")}
    for sid in ("INTERNATIONAL_CHARTER_QUICKVIEW", "MEDIA_CONTEXT", "SOCIAL_CONTEXT", "EMDAT_CONTEXT"):
        assert by_id[sid]["can_promote_alone"] == "false"
    # A quickview cannot even open a candidate alone.
    assert by_id["INTERNATIONAL_CHARTER_QUICKVIEW"]["can_open_candidate"] == "false"


def test_benchmarks_never_local_truth(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset()
    _run(v2at_engine, ds, tmp_path)
    by_id = {r["source_id"]: r for r in _read(ds / "v2at_external_evidence_source_catalog.csv")}
    for sid in ("URBANSARFLOODS_BENCHMARK", "SEN1FLOODS11_BENCHMARK"):
        assert by_id[sid]["source_class"] == "methodological_benchmark"
        assert by_id[sid]["can_promote_alone"] == "false"
        assert by_id[sid]["can_open_candidate"] == "false"


def test_official_brazilian_sources_can_open_but_not_promote_alone(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset()
    _run(v2at_engine, ds, tmp_path)
    by_id = {r["source_id"]: r for r in _read(ds / "v2at_external_evidence_source_catalog.csv")}
    for sid in ("ANA_HIDROWEB", "INMET_HISTORICAL", "CEMADEN_MONITORING", "SGB_RISK_CARTOGRAPHY"):
        assert by_id[sid]["can_open_candidate"] == "true"
        assert by_id[sid]["can_promote_alone"] == "false"
        assert float(by_id[sid]["evidence_weight"]) >= 0.8


def test_catalog_matches_json_schema(v2at_engine, v2at_dataset, tmp_path):
    import json
    import os
    ds = v2at_dataset()
    _run(v2at_engine, ds, tmp_path)
    rows = _read(ds / "v2at_external_evidence_source_catalog.csv")
    schema = json.load(open(os.path.join(
        v2at_engine.PROJECT_ROOT, "datasets", "schemas",
        "v2at_external_evidence_source_catalog.schema.json"), encoding="utf-8"))
    assert set(schema["required"]).issubset(set(rows[0].keys()))
    enums = {c: p["enum"] for c, p in schema["properties"].items() if "enum" in p}
    for row in rows:
        for col, allowed in enums.items():
            assert row[col] in allowed, f"{col}={row[col]} not in enum"
