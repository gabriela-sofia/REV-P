import csv
import os

import pytest

import scripts.protocolo_c.revp_v2au_common as common


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def install(tmp_path, monkeypatch):
    protocol = tmp_path / "datasets" / "protocolo_c"
    docs = tmp_path / "docs" / "protocolo_c" / "v2au_source_resolution_plan"
    cache = docs / "evidence_cache"
    for path in (protocol, docs, cache):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common, "DATASET_DIR", str(protocol))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "CACHE_DIR", str(cache))
    monkeypatch.delenv(common.NETWORK_ENV, raising=False)
    facts = [{"assertion_id": "FACT_1", "candidate_id": "REC_1", "source_id": "SRC_1",
              "source_name": "Official Source", "evidence_role": "DOCUMENTARY_EVENT",
              "critical_blocker": "BLOCKED_LICENSE_UNKNOWN", "source_identified": "true",
              "license_explicit": "false", "crs_resolved": "true", "observed_event": "true",
              "temporal_compatible": "true", "hazard_typed": "true",
              "geometry_or_measurement_compatible": "false", "human_review_complete": "false",
              "independent_corroboration": "true"}]
    gaps = [{"assertion_id": "FACT_1", "blocking_gap": "BLOCKED_LICENSE_UNKNOWN"}]
    blockers = [{"blocker_id": "B1", "scope": "FACT_1", "blocking_reason": "BLOCKED_LICENSE_UNKNOWN"}]
    downloads = [{"source_id": "INMET", "target_url": "https://example.test/inmet"}]
    write_csv(protocol / common.INPUTS["facts"], facts)
    write_csv(protocol / common.INPUTS["gaps"], gaps)
    write_csv(protocol / common.INPUTS["blockers"], blockers)
    write_csv(protocol / common.INPUTS["downloads"], downloads)
    return protocol, docs, cache


CASES = []
with open(os.path.join("tests", "fixtures", "v2au", "classification_cases.csv"), encoding="utf-8", newline="") as handle:
    CASES = list(csv.DictReader(handle))


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['kind']}-{c['expected']}")
def test_classification_cases(case):
    result = common.classify_license(case["value"]) if case["kind"] == "license" else common.classify_crs(case["value"], case["role"])
    assert result == case["expected"]


@pytest.mark.parametrize("blocker,expected", [
    ("BLOCKED_LICENSE_UNKNOWN", "MANUALLY_RESOLVE_LICENSE_AND_TERMS"),
    ("BLOCKED_CRS_UNKNOWN", "MANUALLY_RESOLVE_CRS"),
    ("BLOCKED_TEMPORAL_MISMATCH", "DOWNLOAD_OFFICIAL_HYDROMETEOROLOGICAL_SERIES_FOR_REVIEW_PACKETS"),
    ("BLOCKED_GEOMETRY_OR_MEASUREMENT_MISSING", "ACQUIRE_EXPLICIT_GEOMETRY_OR_OBSERVED_MEASUREMENT"),
    ("BLOCKED_HAZARD_AMBIGUOUS", "RESOLVE_HAZARD_TYPE_FROM_OFFICIAL_SOURCE"),
    ("OTHER", "BUILD_MANUAL_REVIEW_PACKET"),
])
def test_action_mapping(blocker, expected):
    assert common.action_for_blocker(blocker) == expected


@pytest.mark.parametrize("count,expected", [(0, "HIGH"), (2, "HIGH"), (3, "HIGH"), (4, "MEDIUM"), (5, "MEDIUM"), (6, "LOW")])
def test_priority_band(count, expected):
    fields = ["source_identified", "license_explicit", "crs_resolved", "observed_event",
              "temporal_compatible", "hazard_typed", "geometry_or_measurement_compatible",
              "human_review_complete", "independent_corroboration"]
    fact = {field: "false" if index < count else "true" for index, field in enumerate(fields)}
    assert common.priority_band(fact) == expected


@pytest.mark.parametrize("field,expected", list(common.INVARIANTS.items()))
def test_invariant_values(field, expected):
    assert common.with_invariants({})[field] == expected


def test_required_inputs_fail_closed(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch)
    os.remove(common.dataset_path(common.INPUTS["facts"]))
    with pytest.raises(FileNotFoundError):
        common.load_v2at_inputs()


def test_offline_endpoint_attempt(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch)
    result = common.attempt_endpoint("https://example.test", "X")
    assert result["attempt_status"] == "NETWORK_DISABLED_DETERMINISTIC_RUN"
    assert result["raw_data_versioned"] == "false"


def test_cache_policy_exact(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch)
    assert common.validate_cache_policy()
    assert open(common.cache_path(".gitignore"), encoding="utf-8").read() == "*\n!.gitignore\n"


def test_resolution_task_per_fact(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch)
    rows = common.run_build_resolution_task_registry()
    assert len(rows) == 1
    assert rows[0]["resolution_action"] == "MANUALLY_RESOLVE_LICENSE_AND_TERMS"


def test_endpoint_attempts_offline(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch)
    rows = common.run_build_source_endpoint_attempts()
    assert rows[0]["attempt_status"] == "NETWORK_DISABLED_DETERMINISTIC_RUN"


def test_orchestrator_all_ok(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch)
    rows = common.run_orchestrator()
    assert len(rows) == 12
    assert all(row["status"] == "OK" for row in rows)


def test_next_action_rank_1(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch)
    common.run_orchestrator()
    rows = common.load_csv(common.dataset_path("v2au_next_actions_registry.csv"))
    assert rows[0]["next_action"] == "MANUALLY_RESOLVE_LICENSE_AND_CRS_FOR_HIGH_PRIORITY_SOURCES"


def test_guardrail_rejects_promotion(tmp_path, monkeypatch):
    protocol, _, _ = install(tmp_path, monkeypatch)
    write_csv(protocol / "v2au_bad.csv", [{**common.INVARIANTS, "can_create_label": "true"}])
    with pytest.raises(ValueError):
        common.run_guardrail_regression()
