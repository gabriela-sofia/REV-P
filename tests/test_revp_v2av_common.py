import csv
import os

import pytest

import scripts.protocolo_c.revp_v2av_common as common


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def install(tmp_path, monkeypatch):
    protocol = tmp_path / "datasets" / "protocolo_c"
    docs = tmp_path / "docs" / "protocolo_c" / "v2av_official_source_terms_snapshot"
    cache = docs / "evidence_cache"
    for path in (protocol, docs, cache):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common, "DATASET_DIR", str(protocol))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "CACHE_DIR", str(cache))
    monkeypatch.delenv(common.NETWORK_ENV, raising=False)
    rows = {
        "tasks": [{"task_id": "T1", "assertion_id": "A1", "source_id": "SRC1", "source_name": "Official",
                   "priority_band": "HIGH", "blocker_count": "2"}],
        "terms": [{"source_id": "SRC1", "license_status": "UNKNOWN"}],
        "licenses": [{"assertion_id": "A1", "source_id": "SRC1", "license_resolution_status": "UNKNOWN"}],
        "crs": [{"assertion_id": "A1", "source_id": "SRC1", "crs_resolution_status": "NOT_SPATIAL"}],
        "temporal": [{"assertion_id": "A1", "candidate_id": "REC_1", "requested_sources": "INMET|CEMADEN"}],
        "instructions": [{"source_id": "INMET", "step_1": "open"}],
        "endpoints": [{"source_id": "INMET", "target_url": "https://example.test"}],
        "queue": [{"queue_rank": "1", "assertion_id": "A1", "source_id": "INMET", "source_name": "INMET",
                   "priority_band": "HIGH", "blocker_count": "2"}],
    }
    for key, name in common.INPUTS.items():
        write_csv(protocol / name, rows[key])
    return protocol, docs, cache


with open(os.path.join("tests", "fixtures", "v2av", "classification_cases.csv"), encoding="utf-8", newline="") as handle:
    CASES = list(csv.DictReader(handle))


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['kind']}-{c['expected']}")
def test_classifications(case):
    a, b, c = case["a"] == "true", case["b"] == "true", case["c"] == "true"
    if case["kind"] == "page":
        result = common.classify_page(a, b, c)
    elif case["kind"] == "license":
        result = common.classify_license_candidate(a, b, c)
    else:
        result = common.classify_crs_candidate(a, b)
    assert result == case["expected"]


@pytest.mark.parametrize("source,expected", [
    ("INMET", "OFFICIAL_OBSERVED_TIME_SERIES"), ("CEMADEN", "OFFICIAL_OBSERVED_TIME_SERIES"),
    ("ANA_HIDROWEB", "OFFICIAL_OBSERVED_TIME_SERIES"), ("SGB_CPRM", "SUSCEPTIBILITY_OR_RISK_CONTEXT"),
    ("INTERNATIONAL_CHARTER", "QUICKVIEW_OR_PRODUCT"), ("COPERNICUS_EMS", "OFFICIAL_SPATIAL_PRODUCT"),
])
def test_source_roles(source, expected):
    assert common.source_role(source) == expected


@pytest.mark.parametrize("field,expected", list(common.INVARIANTS.items()))
def test_invariants(field, expected):
    assert common.with_invariants({})[field] == expected


def test_missing_input_fails(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch)
    os.remove(common.dataset_path(common.INPUTS["tasks"]))
    with pytest.raises(FileNotFoundError):
        common.load_v2au_inputs()


def test_offline_snapshot(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch)
    row = common.snapshot_page("https://example.test", "X")
    assert row["snapshot_status"] == "NETWORK_DISABLED_DETERMINISTIC_RUN"
    assert row["raw_data_versioned"] == "false"


def test_cache_only_marker(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch)
    marker = common.ensure_cache_policy()
    assert open(marker, encoding="utf-8").read() == "*\n!.gitignore\n"


def test_terms_stay_unknown_offline(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch)
    rows = common.run_build_terms_snapshot_registry()
    assert rows[0]["terms_classification"] == "TERMS_STILL_UNKNOWN"
    assert rows[0]["license_candidate_status"] == "LICENSE_STILL_UNKNOWN"


def test_download_not_license(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch)
    rows = common.run_build_access_page_registry()
    assert rows[0]["download_public_is_public_license"] == "false"


def test_manual_packet_generated(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch)
    rows = common.run_build_manual_acquisition_packets()
    assert rows[0]["manual_steps_available"] == "true"


def test_temporal_target_no_geometry(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch)
    rows = common.run_build_temporal_observation_targets()
    assert rows[0]["can_improve_readiness"] == "true"
    assert rows[0]["can_create_observed_geometry_alone"] == "false"


def test_orchestrator_all_ok(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch)
    rows = common.run_orchestrator()
    assert len(rows) == 8
    assert all(row["status"] == "OK" for row in rows)


def test_guardrail_rejects_label(tmp_path, monkeypatch):
    protocol, _, _ = install(tmp_path, monkeypatch)
    write_csv(protocol / "v2av_bad.csv", [{**common.INVARIANTS, "can_create_label": "true"}])
    with pytest.raises(ValueError):
        common.run_guardrail_regression()
