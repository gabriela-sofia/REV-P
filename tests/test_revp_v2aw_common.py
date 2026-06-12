import csv
import os

import pytest

import scripts.protocolo_c.revp_v2aw_common as common


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def install(tmp_path, monkeypatch):
    protocol = tmp_path / "datasets" / "protocolo_c"
    docs = tmp_path / "docs/protocolo_c/v2aw_public_data_observational_acquisition"
    cache = docs / "evidence_cache"
    for path in (protocol, docs, cache):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common, "DATASET_DIR", str(protocol))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "CACHE_DIR", str(cache))
    monkeypatch.delenv(common.NETWORK_ENV, raising=False)
    rows = {
        "snapshots": [{"source_id": "INMET", "official_url": "https://example.test/inmet"}],
        "manual_packets": [{"assertion_id": "A1", "source_id": "INMET"}],
        "tasks": [{"assertion_id": "A1", "source_id": "SRC1"}],
        "facts": [{"assertion_id": "A1", "candidate_id": "REC_2023_01_01", "event_id": "REC_2023_01_01",
                   "patch_id": "", "source_id": "SRC1", "fact_classification": "BLOCKED_LICENSE_UNKNOWN",
                   "critical_blocker": "BLOCKED_LICENSE_UNKNOWN", "license_explicit": "false",
                   "temporal_compatible": "true", "hazard_typed": "true",
                   "geometry_or_measurement_compatible": "false"}],
    }
    for key, name in common.INPUTS.items():
        write_csv(protocol / name, rows[key])
    write_csv(protocol / "v2as_geojson_candidate_index.csv",
              [{"candidate_id": "REC_2023_01_01", "geometry_present": "false"}])
    return protocol, docs, cache


with open(os.path.join("tests", "fixtures", "v2aw", "classification_cases.csv"), encoding="utf-8", newline="") as handle:
    CASES = list(csv.DictReader(handle))


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['kind']}-{c['expected']}")
def test_classifications(case):
    a, b, c = case["a"] == "true", case["b"] == "true", case["c"] == "true"
    if case["kind"] == "license":
        result = common.reclassify_license(a, b, c)
    elif case["kind"] == "temporal":
        result = common.temporal_status(a, b)
    else:
        result = common.package_status({"temporal_compatible": str(a), "hazard_typed": str(b),
                                        "geometry_or_measurement_compatible": str(c)})[0]
    assert result == case["expected"]


@pytest.mark.parametrize("source,role,expected", [
    ("INMET", "", "TEMPORAL_OBSERVATION_TARGET"),
    ("CEMADEN", "", "TEMPORAL_OBSERVATION_TARGET"),
    ("ANA_HIDROWEB", "", "TEMPORAL_OBSERVATION_TARGET"),
    ("SGB_CPRM", "SUSCEPTIBILITY_OR_RISK_CONTEXT", "CONTEXT_ONLY_SUSCEPTIBILITY"),
    ("INTERNATIONAL_CHARTER", "QUICKVIEW_OR_PRODUCT", "REVIEW_ONLY_QUICKVIEW"),
    ("X", "DINO_SIGNAL", "REVIEW_ONLY_DINO_SIGNAL"),
    ("COPERNICUS_EMS", "OFFICIAL_SPATIAL_PRODUCT", "PUBLIC_DATA_CITATION_REQUIRED"),
])
def test_observational_target_classes(source, role, expected):
    assert common.observational_target_class(source, role) == expected


@pytest.mark.parametrize("field,expected", list(common.INVARIANTS.items()))
def test_invariants(field, expected):
    assert common.with_invariants({})[field] == expected


def test_license_unknown_no_longer_blocks_acquisition(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch)
    row = common.run_reclassify_license_blockers()[0]
    assert row["license_blocker"] == "false"
    assert row["acquisition_allowed"] == "true"
    assert row["can_create_ground_truth"] == "false"


def test_public_provenance_requires_citation(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch)
    row = common.run_build_public_source_provenance()[0]
    assert row["public_data_assumed"] == "true"
    assert row["legal_use_assumed"] == "true"
    assert row["source_citation_required"] == "true"


def test_offline_hydromet_plan(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch)
    rows = common.run_build_hydrometeorological_acquisition_plan()
    assert len(rows) == 3
    assert all(row["acquisition_status"] == "NETWORK_DISABLED_DETERMINISTIC_RUN" for row in rows)


def test_geometry_equivalent_input_used(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch)
    stack = common.load_inputs()
    assert stack["geometry_source"].endswith("v2as_geojson_candidate_index.csv")


def test_event_package_not_ready_without_geometry(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch)
    row = common.run_build_event_patch_observation_packages()[0]
    assert row["review_status"] == "NOT_READY_FOR_REVIEW"
    assert row["blocking_reason"] == "GEOMETRY_STILL_MISSING"


def test_readiness_next_action(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch)
    common.run_build_event_patch_observation_packages()
    rows = common.run_generate_observational_readiness_report()
    assert rows[-1]["value"] == "ACQUIRE_PUBLIC_HYDROMETEOROLOGICAL_TIME_SERIES_FOR_EVENT_PATCH_REVIEW"


def test_cache_policy(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch)
    marker = common.ensure_cache_policy()
    assert open(marker, encoding="utf-8").read() == "*\n!.gitignore\n"


def test_orchestrator_all_ok(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch)
    rows = common.run_orchestrator()
    assert len(rows) == 8
    assert all(row["status"] == "OK" for row in rows)


def test_guardrail_rejects_label(tmp_path, monkeypatch):
    protocol, _, _ = install(tmp_path, monkeypatch)
    write_csv(protocol / "v2aw_bad.csv", [{**common.INVARIANTS, "can_create_label": "true"}])
    with pytest.raises(ValueError):
        common.run_guardrail_regression()
