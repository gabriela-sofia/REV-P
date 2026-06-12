import datetime as dt
import json
import os
import zipfile

import pytest

import scripts.protocolo_c.revp_v2bi_common as common

FIXTURES = os.path.join("tests", "fixtures", "v2bi")


@pytest.fixture(scope="module", autouse=True)
def generated(tmp_path_factory):
    # Isolate the manual-intake scan to an empty temporary cache so the
    # fail-closed empty-cache baseline stays deterministic even when real
    # evidence has been placed in the local (git-ignored) intake cache.
    saved = (common.CACHE_DIR, common.CHARTER_CACHE, common.TEMPORAL_CACHE)
    empty = tmp_path_factory.mktemp("v2bi_empty_cache")
    common.CACHE_DIR = str(empty)
    common.CHARTER_CACHE = str(empty / "manual_charter_758")
    common.TEMPORAL_CACHE = str(empty / "manual_temporal")
    os.makedirs(common.CHARTER_CACHE, exist_ok=True)
    os.makedirs(common.TEMPORAL_CACHE, exist_ok=True)
    common.run_orchestrator()
    yield
    common.CACHE_DIR, common.CHARTER_CACHE, common.TEMPORAL_CACHE = saved


@pytest.mark.parametrize("field,expected", list(common.INVARIANTS.items()))
def test_invariants(field, expected):
    assert common.with_invariants({})[field] == expected


@pytest.mark.parametrize("name", list(common.INPUTS))
def test_inputs(name):
    if name == "gap_selection":
        assert common.load_csv(common.dataset_path(common.INPUTS[name])) == []
    else:
        assert common.load_csv(common.dataset_path(common.INPUTS[name]))


@pytest.mark.parametrize("output", common.OUTPUTS + ["v2bi_orchestrator_manifest.csv"])
def test_outputs(output):
    assert os.path.exists(common.dataset_path(output))
    assert common.load_csv(common.dataset_path(output))


@pytest.mark.parametrize("filename,expected", [
    ("a.zip", "ZIP"), ("a.shp", "SHP"), ("a.geojson", "GEOJSON"), ("a.json", "GEOJSON"),
    ("a.kml", "KML"), ("a.gpkg", "GPKG"), ("a.pdf", "PDF"), ("a.png", "PNG"),
    ("a.jpg", "JPG"), ("a.jpeg", "JPG"), ("a.csv", "CSV"), ("a.txt", "CSV"),
    ("a.xlsx", "XLSX"), ("a.html", "HTML"), ("a.bin", "UNKNOWN"),
])
def test_detect_type(filename, expected):
    assert common.detect_type(filename) == expected


@pytest.mark.parametrize("name,expected", [
    ("apac_recife.csv", "APAC"), ("CEMADEN.csv", "CEMADEN"), ("ana_hidroweb.zip", "ANA_HIDROWEB"),
    ("inmet_proxy.csv", "INMET_PROXY"), ("other.csv", "UNKNOWN"),
])
def test_source_candidate(name, expected):
    assert common.source_candidate(name) == expected


def test_real_cache_empty():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[0]))
    assert len(rows) == 1 and rows[0]["note"] == "NO_MANUAL_INTAKE_FILES_FOUND"


def test_charter_absent():
    row = common.load_csv(common.dataset_path(common.OUTPUTS[1]))[0]
    assert row["audit_status"] == "NO_MANUAL_CHARTER_FILE_FOUND"


def test_vector_not_available():
    row = common.load_csv(common.dataset_path(common.OUTPUTS[2]))[0]
    assert row["metadata_status"] == "VECTOR_NOT_AVAILABLE"


@pytest.mark.parametrize("fixture,crs,count,geometry,status", [
    ("recife_valid.geojson", "EPSG:4326", 1, "Polygon", "VECTOR_METADATA_EXTRACTED"),
    ("recife_no_crs.geojson", "", 1, "Point", "VECTOR_METADATA_EXTRACTED"),
    ("outside.geojson", "EPSG:4326", 1, "Point", "VECTOR_METADATA_EXTRACTED"),
])
def test_geojson_metadata(fixture, crs, count, geometry, status):
    result = common.geojson_metadata(os.path.join(FIXTURES, fixture))
    assert result["crs"] == crs and result["count"] == count
    assert result["geometry"] == geometry and result["status"] == status


def test_geojson_read_failed(tmp_path):
    path = tmp_path / "bad.geojson"; path.write_text("{", encoding="utf-8")
    assert common.geojson_metadata(str(path))["status"] == "READ_FAILED"


def meta(crs="true", geometry="Polygon", count="1", bbox=("-34.91", "-8.10", "-34.90", "-8.09"), vector="true"):
    return {"vector_file_detected": vector, "crs_detected": crs, "geometry_type": geometry, "feature_count": count,
            "bbox_minx": bbox[0], "bbox_miny": bbox[1], "bbox_maxx": bbox[2], "bbox_maxy": bbox[3]}


@pytest.mark.parametrize("data,expected", [
    (meta(), "VALID_FOR_HUMAN_REVIEW"), (meta(crs="false"), "CRS_MISSING"),
    (meta(geometry="", count="0"), "GEOMETRY_MISSING"), (meta(bbox=("-43.3", "-23", "-43.1", "-22.8")), "OUTSIDE_EXPECTED_AREA"),
    (meta(vector="false"), "NOT_AVAILABLE"),
])
def test_validation_status(data, expected):
    assert common.validation_status(data) == expected


@pytest.mark.parametrize("data,brazil,recife", [
    (meta(), True, True), (meta(bbox=("-43.3", "-23", "-43.1", "-22.8")), True, False),
    (meta(bbox=("10", "10", "11", "11")), False, False),
    (meta(bbox=("", "", "", "")), False, False),
])
def test_bbox_checks(data, brazil, recife):
    assert common.bbox_checks(data) == (brazil, recife)


def test_zip_shapefile_detection(tmp_path):
    path = tmp_path / "charter.zip"
    with zipfile.ZipFile(path, "w") as archive: archive.writestr("recife.shp", "x")
    summary, vector = common.archive_summary(str(path))
    assert "recife.shp" in summary and vector


def test_bad_zip(tmp_path):
    path = tmp_path / "bad.zip"; path.write_text("bad", encoding="utf-8")
    assert common.archive_summary(str(path)) == ("ZIP_READ_FAILED", False)


@pytest.mark.parametrize("kind,vector,expected", [
    ("ZIP", True, "VECTOR_CANDIDATE_FOUND"), ("GEOJSON", True, "VECTOR_CANDIDATE_FOUND"),
    ("PDF", False, "MAP_ONLY_FOUND"), ("PNG", False, "PREVIEW_ONLY_FOUND"),
    ("JPG", False, "PREVIEW_ONLY_FOUND"), ("HTML", False, "UNSUPPORTED_FILE"),
])
def test_charter_audit_status(kind, vector, expected):
    assert common.charter_audit_status(kind, vector) == expected


@pytest.mark.parametrize("raw,expected", [
    ("10,5", 10.5), ("10.5", 10.5), ("1.234,5", 1234.5), ("", None), ("abc", None),
])
def test_parse_number(raw, expected):
    assert common.parse_number(raw) == expected


@pytest.mark.parametrize("raw,hour,expected", [
    ("2022-05-24 01:00", "", dt.datetime(2022, 5, 24, 1)), ("24/05/2022", "01:00", dt.datetime(2022, 5, 24, 1)),
    ("2022-05-24", "", dt.datetime(2022, 5, 24)), ("bad", "", None),
])
def test_parse_datetime(raw, hour, expected):
    assert common.parse_datetime(raw, hour) == expected


@pytest.mark.parametrize("fixture,source,count,status,precip_col", [
    ("apac_semicolon.csv", "APAC", 2, "PARSED", "chuva"),
    ("cemaden_comma.csv", "CEMADEN", 2, "PARSED", "precipitacao"),
    ("unknown_schema.csv", "UNKNOWN", 0, "UNSUPPORTED_SCHEMA", ""),
])
def test_parse_temporal_csv(fixture, source, count, status, precip_col):
    rows, info = common.parse_temporal_csv(os.path.join(FIXTURES, fixture), source)
    assert len(rows) == count and info["status"] == status and info["precip"] == precip_col


def test_apac_decimal_comma():
    rows, _ = common.parse_temporal_csv(os.path.join(FIXTURES, "apac_semicolon.csv"), "APAC")
    assert [r["precipitation"] for r in rows] == [10.5, 2.5]


def test_cemaden_station():
    rows, _ = common.parse_temporal_csv(os.path.join(FIXTURES, "cemaden_comma.csv"), "CEMADEN")
    assert all(r["station"] == "CEM_REC_01" and r["municipality"] == "Recife" for r in rows)


def test_simple_temporal_zip(tmp_path):
    path = tmp_path / "apac_series.zip"
    payload = open(os.path.join(FIXTURES, "apac_semicolon.csv"), encoding="utf-8").read()
    with zipfile.ZipFile(path, "w") as archive: archive.writestr("apac.csv", payload)
    rows, info = common.parse_temporal_file(str(path), "ZIP", "APAC")
    assert len(rows) == 2 and info["status"] == "PARSED"


def test_temporal_zip_without_csv(tmp_path):
    path = tmp_path / "empty.zip"
    with zipfile.ZipFile(path, "w") as archive: archive.writestr("readme.md", "none")
    rows, info = common.parse_temporal_file(str(path), "ZIP", "APAC")
    assert rows == [] and info["status"] == "UNSUPPORTED_SCHEMA"


def test_unsupported_xlsx_fail_closed(tmp_path):
    path = tmp_path / "series.xlsx"; path.write_text("not an xlsx", encoding="utf-8")
    rows, info = common.parse_temporal_file(str(path), "XLSX", "APAC")
    assert rows == [] and info["status"] == "READ_FAILED"


def test_temporal_metrics_ready():
    rows = [{"timestamp": dt.datetime(2022, 5, 24, h), "precipitation": 1.0, "station": "S", "municipality": "Recife", "source": "APAC"} for h in range(24)]
    result = common.temporal_metrics(rows, dt.date(2022, 5, 24), dt.date(2022, 5, 24))
    assert result["status"] == "TEMPORAL_EVIDENCE_READY_FOR_REVIEW"
    assert result["total"] == "24.000" and result["max1h"] == "1.000" and result["max24h"] == "24.000"


def test_temporal_metrics_partial():
    rows = [{"timestamp": dt.datetime(2022, 5, 24), "precipitation": 3.0, "station": "S", "municipality": "Recife", "source": "APAC"}]
    assert common.temporal_metrics(rows, dt.date(2022, 5, 24), dt.date(2022, 5, 24))["status"] == "TEMPORAL_EVIDENCE_PARTIAL"


def test_temporal_metrics_empty():
    result = common.temporal_metrics([], dt.date(2022, 5, 24), dt.date(2022, 5, 24))
    assert result["status"] == "NO_SERIES_AVAILABLE" and result["missing"] == "1.000"


def test_real_mode_a_readiness():
    row = common.load_csv(common.dataset_path(common.OUTPUTS[4]))[0]
    assert row["updated_candidate_status"] == "NO_FILE_AVAILABLE"
    assert row["required_next_action"] == "MANUALLY_DOWNLOAD_CHARTER_758_PRODUCT_AND_APAC_CEMADEN_SERIES"


def test_real_mode_a_temporal():
    parse = common.load_csv(common.dataset_path(common.OUTPUTS[6]))[0]
    assert parse["parse_status"] == "NO_TEMPORAL_SERIES_FOUND"
    assert all(r["temporal_status"] == "NO_SERIES_AVAILABLE" for r in common.load_csv(common.dataset_path(common.OUTPUTS[7])))


def gate(candidate, gate_id):
    return next(r for r in common.load_csv(common.dataset_path(common.OUTPUTS[8])) if r["candidate_id"] == candidate and r["gate_id"] == gate_id)


@pytest.mark.parametrize("gate_id,status", [
    ("C0_PROVENANCE", "PASS"), ("C1_TEMPORALITY", "PENDING"), ("C2_VALID_SERIES_OR_STATION", "BLOCKED"),
    ("C3_SPATIAL_ANCHOR", "PASS"), ("C4_CANDIDATE_GEOMETRY", "PENDING_VECTOR_CRS"), ("C5_HUMAN_REVIEW", "PENDING"),
    ("C6_CANDIDATE_REFERENCE", "BLOCKED"), ("C7_FINAL_GROUND_TRUTH", "BLOCKED"),
])
def test_mode_a_may_gates(gate_id, status):
    assert gate("REC_2022_05_24_30", gate_id)["updated_status"] == status


def test_other_events_not_promoted():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[8]))
    other = [r for r in rows if r["candidate_id"] != "REC_2022_05_24_30"]
    assert all(r["updated_status"] == "BLOCKED" for r in other if r["gate_id"] in {"C3_SPATIAL_ANCHOR", "C4_CANDIDATE_GEOMETRY", "C7_FINAL_GROUND_TRUTH"})


@pytest.mark.parametrize("blocker", ["CHARTER_VECTOR_CRS_ACCESS", "APAC_CEMADEN_SERIES_ACCESS", "FEATURE_TYPE_CONFIRMATION", "LICENSE_TERMS_CONFIRMATION", "HUMAN_REVIEW_REQUIRED"])
def test_manual_blockers(blocker):
    assert any(r["blocker_type"] == blocker for r in common.load_csv(common.dataset_path(common.OUTPUTS[9])))


@pytest.mark.parametrize("field", ["can_create_ground_truth", "can_create_patch_truth", "can_create_label", "can_create_negative", "can_train_model", "raw_data_versioned"])
def test_zero_forbidden(field):
    for output in common.OUTPUTS[:10]:
        assert all(r[field] == "false" for r in common.load_csv(common.dataset_path(output)))


@pytest.mark.parametrize("folder", ["evidence_cache", "evidence_cache/manual_charter_758", "evidence_cache/manual_temporal"])
def test_cache_markers(folder):
    assert open(common.doc_path(folder, ".gitignore"), encoding="utf-8").read() == "*\n!.gitignore\n"


@pytest.mark.parametrize("text", ["BLOCKED_MANUAL_ACCESS_REQUIRED", "C3 permanece PASS", "C7 permanece BLOCKED", "Ground truth final, labels, negativos e treino continuam zero"])
def test_readme(text):
    assert text in open(common.doc_path("README.md"), encoding="utf-8").read()


def test_guardrails():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[10]))
    assert len(rows) == 11 and all(r["status"] == "PASS" for r in rows)


def test_orchestrator():
    rows = common.load_csv(common.dataset_path("v2bi_orchestrator_manifest.csv"))
    assert len(rows) == 12 and all(r["status"] == "OK" for r in rows)
