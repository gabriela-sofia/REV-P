import csv
import os

import pytest

import scripts.protocolo_c.revp_v2at_common as common


FIXTURE = os.path.join("tests", "fixtures", "v2at", "assertion_cases.csv")


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def install_env(tmp_path, monkeypatch):
    datasets = tmp_path / "datasets"
    protocol = datasets / "protocolo_c"
    docs = tmp_path / "docs" / "protocolo_c" / "v2at_evidence_fact_hardening"
    cache = docs / "evidence_cache"
    for path in (datasets, protocol, docs, cache):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common, "DATASET_ROOT", str(datasets))
    monkeypatch.setattr(common, "DATASET_DIR", str(protocol))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "CACHE_DIR", str(cache))
    monkeypatch.delenv(common.NETWORK_ENV, raising=False)
    priorities = [{"candidate_id": "REC_2023_02_05_06", "region": "Recife"}]
    geometry = [{"candidate_id": "REC_2023_02_05_06", "geometry_present": "false"}]
    sources = [{"source_registry_id": "SRC_1", "candidate_id": "REC_2023_02_05_06",
                "source_role": "primary", "source_name": "Prefeitura", "source_url_or_document": "https://example.test"}]
    licenses = [{"checklist_id": "LC_REC_2023_02_05_06_primary", "candidate_id": "REC_2023_02_05_06",
                 "license_status": "UNKNOWN_NEEDS_REVIEW", "crs_status": "NOT_DOCUMENTED_NEEDS_ASSIGNMENT"}]
    observed = [{"observed_event_id": "REC_2023_02_05_06", "observed_event_confirmed": "true",
                 "temporal_alignment_status": "CLOSED", "event_type": "flood",
                 "secondary_source_name": "CEMADEN"}]
    write_csv(protocol / "v2as_deep_probe_priority.csv", list(priorities[0]), priorities)
    write_csv(protocol / "v2as_geojson_candidate_index.csv", list(geometry[0]), geometry)
    write_csv(protocol / "v2ar_official_geometry_source_registry.csv", list(sources[0]), sources)
    write_csv(protocol / "v2ar_license_crs_checklist.csv", list(licenses[0]), licenses)
    write_csv(datasets / "observed_event_reference_candidate_registry.csv", list(observed[0]), observed)
    return datasets, protocol, docs, cache


CASES = read_csv(FIXTURE)


@pytest.mark.parametrize("case", CASES, ids=[case["case_id"] for case in CASES])
def test_classification_fixture_cases(case):
    classification, blocker = common.classify_fact_assertion(case)
    assert classification == case["expected"]
    if classification.startswith("BLOCKED_"):
        assert blocker == classification


@pytest.mark.parametrize("value,expected", [
    ("PUBLIC_OPEN", "LICENSE_EXPLICIT"),
    ("CC-BY-4.0", "LICENSE_EXPLICIT"),
    ("EXPLICIT_REUSE", "LICENSE_EXPLICIT"),
    ("UNKNOWN_NEEDS_REVIEW", "LICENSE_UNKNOWN_BLOCKING"),
    ("", "LICENSE_UNKNOWN_BLOCKING"),
    ("PENDING", "LICENSE_UNKNOWN_BLOCKING"),
])
def test_license_normalization(value, expected):
    assert common.normalize_license(value) == expected


@pytest.mark.parametrize("value,role,expected", [
    ("EPSG:4326", "PRODUCT", "CRS_EXPLICIT"),
    ("SIRGAS 2000", "PRODUCT", "CRS_EXPLICIT"),
    ("UNKNOWN", "PRODUCT", "CRS_UNKNOWN_BLOCKING"),
    ("NOT_DOCUMENTED", "PRODUCT", "CRS_UNKNOWN_BLOCKING"),
    ("", "OBSERVED_MEASUREMENT", "CRS_IRRELEVANT_FOR_NON_GEOMETRIC_ASSERTION"),
    ("", "DOCUMENTARY_EVENT", "CRS_IRRELEVANT_FOR_NON_GEOMETRIC_ASSERTION"),
])
def test_crs_normalization(value, role, expected):
    assert common.normalize_crs(value, role) == expected


@pytest.mark.parametrize("role,expected", [
    ("SUSCEPTIBILITY_CONTEXT", True), ("RISK_MAP", True), ("STATIC_CARTOGRAPHY", True),
    ("DINO_SIGNAL", True), ("GIS_PROXY", True), ("CONTEXT_ONLY", True),
    ("PRODUCT", False), ("QUICKVIEW", False), ("OBSERVED_MEASUREMENT", False),
])
def test_context_roles(role, expected):
    assert common.role_is_context(role) is expected


@pytest.mark.parametrize("flags,score", [
    ({}, 0),
    ({"source_identified": "true"}, 10),
    ({"source_identified": "true", "license_explicit": "true"}, 20),
    ({"source_identified": "true", "license_explicit": "true", "crs_resolved": "true"}, 30),
    ({"source_identified": "true", "license_explicit": "true", "crs_resolved": "true",
      "observed_event": "true", "temporal_compatible": "true", "hazard_typed": "true",
      "geometry_or_measurement_compatible": "true", "human_review_complete": "true",
      "independent_corroboration": "true"}, 90),
])
def test_evidence_score(flags, score):
    assert common.evidence_score(flags) == score


def test_network_disabled_default(tmp_path, monkeypatch):
    install_env(tmp_path, monkeypatch)
    result = common.fetch_to_cache("https://example.test", "X")
    assert result["download_status"] == "NETWORK_DISABLED_DETERMINISTIC_RUN"
    assert result["raw_data_versioned"] == "false"


def test_cache_policy_exact(tmp_path, monkeypatch):
    install_env(tmp_path, monkeypatch)
    status = common.validate_cache_policy()
    assert status["cache_policy_valid"] == "true"
    assert (tmp_path / "docs/protocolo_c/v2at_evidence_fact_hardening/evidence_cache/.gitignore").read_text() == "*\n!.gitignore\n"


def test_source_authority_matrix_invariants(tmp_path, monkeypatch):
    install_env(tmp_path, monkeypatch)
    rows = common.run_build_source_authority_matrix()
    assert len(rows) == 6
    assert all(row["quickview_can_promote"] == "false" for row in rows)
    assert all(row["susceptibility_can_promote"] == "false" for row in rows)
    assert all(row["absence_can_create_negative"] == "false" for row in rows)


def test_fact_registry_blocks_unknown_license(tmp_path, monkeypatch):
    install_env(tmp_path, monkeypatch)
    rows = common.run_build_fact_assertion_registry()
    assert rows[0]["fact_classification"] == "BLOCKED_LICENSE_UNKNOWN"
    assert rows[0]["can_create_ground_truth"] == "false"
    assert rows[0]["can_create_training_label"] == "false"


def test_orchestrator_runs_all_steps(tmp_path, monkeypatch):
    install_env(tmp_path, monkeypatch)
    rows = common.run_orchestrator()
    assert len(rows) == 12
    assert all(row["status"] == "OK" for row in rows)


def test_gap_report_is_explicit(tmp_path, monkeypatch):
    install_env(tmp_path, monkeypatch)
    common.run_orchestrator()
    rows = read_csv(tmp_path / "datasets/protocolo_c/v2at_non_fact_gap_report.csv")
    assert rows
    assert all(row["do_not_infer"] == "true" for row in rows)


def test_guardrail_detects_ground_truth_promotion(tmp_path, monkeypatch):
    _, protocol, _, _ = install_env(tmp_path, monkeypatch)
    write_csv(protocol / "v2at_bad.csv", ["can_create_ground_truth"], [{"can_create_ground_truth": "true"}])
    common.run_build_source_authority_matrix()
    with pytest.raises(ValueError):
        common.run_guardrail_regression()
