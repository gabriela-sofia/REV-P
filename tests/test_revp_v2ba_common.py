import json
import os

import pytest

import scripts.protocolo_c.revp_v2ba_common as common


@pytest.fixture(scope="module", autouse=True)
def generated():
    common.run_orchestrator()


@pytest.mark.parametrize("field,expected", list(common.INVARIANTS.items()))
def test_invariants(field, expected):
    assert common.with_invariants({})[field] == expected


@pytest.mark.parametrize("source_type,source_name,kwargs,expected", [
    ("OFFICIAL_VECTOR", "official vector", {}, "CANDIDATE_GEOMETRY_FROM_REVIEWABLE_EVIDENCE"),
    ("TECHNICAL_REPORT", "official report", {}, "TEXTUAL_LOCATION_ONLY"),
    ("OFFICIAL_MUNICIPAL", "municipal page", {}, "TEXTUAL_LOCATION_ONLY"),
    ("ACADEMIC_PAPER", "paper", {}, "REVIEW_ONLY_VISUAL_SUPPORT"),
    ("COPERNICUS_PRODUCT", "Copernicus product", {}, "REVIEW_ONLY_VISUAL_SUPPORT"),
    ("LOCAL_NEWS", "quickview", {"quickview": True}, "QUICKVIEW_ONLY"),
    ("OFFICIAL_GEOLOGICAL", "SGB CPRM", {}, "SUSCEPTIBILITY_CONTEXT_ONLY"),
    ("OTHER", "context", {}, "CONTEXT_ONLY"),
    ("OTHER", "map", {"found_map": True}, "OFFICIAL_MAP_PRODUCT_REQUIRES_VALIDATION"),
    ("OTHER", "geometry", {"has_geometry": True}, "OFFICIAL_OBSERVED_GEOMETRY"),
])
def test_source_classification(source_type, source_name, kwargs, expected):
    assert common.source_classification(source_type, source_name, **kwargs) == expected


@pytest.mark.parametrize("evidence,allowed", [
    ("OFFICIAL_OBSERVED_GEOMETRY", True),
    ("OFFICIAL_MAP_PRODUCT_REQUIRES_VALIDATION", True),
    ("CANDIDATE_GEOMETRY_FROM_REVIEWABLE_EVIDENCE", True),
    ("TEXTUAL_LOCATION_ONLY", False),
    ("REVIEW_ONLY_VISUAL_SUPPORT", False),
    ("QUICKVIEW_ONLY", False),
    ("SUSCEPTIBILITY_CONTEXT_ONLY", False),
    ("CONTEXT_ONLY", False),
    ("NO_GEOMETRY_FOUND", False),
])
def test_candidate_allowed(evidence, allowed):
    assert common.candidate_allowed(evidence) is allowed


@pytest.mark.parametrize("evidence,method", [
    ("OFFICIAL_OBSERVED_GEOMETRY", "OFFICIAL_VECTOR"),
    ("OFFICIAL_MAP_PRODUCT_REQUIRES_VALIDATION", "OFFICIAL_RASTER_TRACE"),
    ("CANDIDATE_GEOMETRY_FROM_REVIEWABLE_EVIDENCE", "REVIEWABLE_VISUAL_TRACE"),
    ("TEXTUAL_LOCATION_ONLY", "MANUAL_DIGITIZATION_REQUIRED"),
    ("REVIEW_ONLY_VISUAL_SUPPORT", "MANUAL_DIGITIZATION_REQUIRED"),
    ("QUICKVIEW_ONLY", "NONE"),
    ("SUSCEPTIBILITY_CONTEXT_ONLY", "NONE"),
])
def test_digitization_method(evidence, method):
    assert common.digitization_method(evidence) == method


def test_null_geometry_validation():
    result = common.validate_geometry_payload(None)
    assert result["geometry_present"] == "false"
    assert result["validation_status"] == "NULL_GEOMETRY_VALID_FOR_BLOCKED_CANDIDATE"


def test_valid_brazil_point():
    result = common.validate_geometry_payload({"type": "Point", "coordinates": [-43.2, -22.4]}, "EPSG:4326")
    assert result["geometry_valid"] == "true"
    assert result["coordinates_within_brazil"] == "true"


def test_crs_missing_blocks_validation():
    result = common.validate_geometry_payload({"type": "Point", "coordinates": [-43.2, -22.4]})
    assert result["geometry_valid"] == "false"
    assert "CRS_MISSING" in result["fail_reason"]


def test_outside_brazil_fails():
    result = common.validate_geometry_payload({"type": "Point", "coordinates": [10, 45]}, "EPSG:4326")
    assert result["geometry_valid"] == "false"
    assert "COORDINATES_OUTSIDE_BRAZIL_OR_MISSING" in result["fail_reason"]


def test_invalid_geometry_type_fails():
    result = common.validate_geometry_payload({"type": "Circle", "coordinates": [-43, -22]}, "EPSG:4326")
    assert "INVALID_GEOMETRY_TYPE" in result["fail_reason"]


def test_polygon_coordinates_flatten():
    points = common.flatten_coordinates([[[-43, -22], [-43.1, -22.1], [-43, -22]]])
    assert len(points) == 3


@pytest.mark.parametrize("status,hazard,expected", [
    ("NOT_CREATED", "MIXED", "VERY_HIGH"),
    ("BLOCKED", "FLOOD", "VERY_HIGH"),
    ("CANDIDATE_REQUIRES_HUMAN_VALIDATION", "MIXED", "HIGH"),
    ("CANDIDATE_REQUIRES_HUMAN_VALIDATION", "FLOOD", "MODERATE"),
])
def test_uncertainty_level(status, hazard, expected):
    assert common.uncertainty_level(status, hazard) == expected


@pytest.mark.parametrize("output", common.OUTPUTS + ["v2ba_orchestrator_manifest.csv"])
def test_required_outputs_exist(output):
    assert os.path.exists(common.dataset_path(output))
    assert common.load_csv(common.dataset_path(output))


def test_review_ready_selection_six():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[0]))
    assert len(rows) == 6
    assert all(row["selection_status"] == "REVIEW_READY_SELECTED" for row in rows)


def test_recife_not_selected():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[0]))
    assert not any(row["region"] == "Recife" for row in rows)


def test_search_plan_has_two_tasks_per_packet():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[1]))
    assert len(rows) == 12
    assert {sum(row["review_packet_id"] == packet for row in rows) for packet in {row["review_packet_id"] for row in rows}} == {2}


@pytest.mark.parametrize("field", [
    "search_task_id", "review_packet_id", "event_patch_package_id", "region", "city", "patch_id",
    "event_date", "temporal_support_status", "source_target", "source_type", "search_url_or_reference",
    "expected_geometry_type", "search_priority", "search_reason", "can_promote_directly",
])
def test_search_plan_fields(field):
    assert field in common.load_csv(common.dataset_path(common.OUTPUTS[1]))[0]


def test_offline_probes():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[2]))
    assert len(rows) == 12
    assert all(row["probe_mode"] == "OFFLINE_DETERMINISTIC" for row in rows)
    assert all(row["status"] == "NETWORK_DISABLED_DETERMINISTIC_RUN" for row in rows)


def test_no_raw_payload_cached():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[2]))
    assert all(row["raw_payload_cached"] == "false" and row["raw_data_versioned"] == "false" for row in rows)


def test_source_probe_summaries_generated():
    paths = [name for name in os.listdir(common.doc_path("source_probe_summaries")) if name.endswith(".md")]
    assert len(paths) == 6
    assert all("Nenhum payload bruto foi versionado" in open(common.doc_path("source_probe_summaries", name), encoding="utf-8").read() for name in paths)


def test_actual_evidence_has_no_allowed_geometry():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[3]))
    assert len(rows) == 12
    assert all(row["geometry_candidate_allowed"] == "false" for row in rows)


def test_textual_location_not_geometry():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[3]))
    textual = [row for row in rows if row["evidence_class"] == "TEXTUAL_LOCATION_ONLY"]
    assert textual
    assert all(row["automatic_digitization_allowed"] == "false" for row in textual)


def test_candidate_registry_six_not_created():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[4]))
    assert len(rows) == 6
    assert all(row["candidate_status"] == "NOT_CREATED" for row in rows)
    assert all(row["digitization_method"] == "NONE" for row in rows)


def test_broad_text_not_digitized():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[4]))
    assert all("TEXTUAL_OR_VISUAL_CONTEXT_CANNOT_BE_AUTO_DIGITIZED" in row["blocker_reason"] for row in rows)


def test_patch_boundary_not_geometry():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[4]))
    assert all(row["patch_boundary_is_not_event_geometry"] == "true" for row in rows)


def test_candidate_geojson_manifest_six_null():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[5]))
    assert len(rows) == 6
    assert all(row["geometry_present"] == "false" and row["geometry_null_allowed"] == "true" for row in rows)


@pytest.mark.parametrize("candidate", [
    "pet-2022-02-15", "ctb-2023-10-28-30", "pet-2022-03-20-21",
    "ctb-2022-01-15-16", "ctb-2024-02-18-20", "pet-2024-03-21-28",
])
def test_null_candidate_geojson_valid(candidate):
    payload = json.load(open(common.doc_path("candidate_geojsons", f"{candidate}.geojson"), encoding="utf-8"))
    feature = payload["features"][0]
    assert feature["geometry"] is None
    assert feature["properties"]["candidate_status"] == "NOT_CREATED"
    assert feature["properties"]["can_create_ground_truth"] is False
    assert feature["properties"]["can_create_label"] is False


def test_candidate_geometry_validation_six():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[6]))
    assert len(rows) == 6
    assert all(row["validation_status"] == "NULL_GEOMETRY_VALID_FOR_BLOCKED_CANDIDATE" for row in rows)
    assert all(row["can_promote"] == "false" for row in rows)


def test_uncertainty_all_very_high():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[7]))
    assert all(row["overall_uncertainty"] == "VERY_HIGH" for row in rows)


def test_adjudication_queue_six():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[8]))
    assert len(rows) == 6
    assert all(row["current_truth_status"] == "NOT_GROUND_TRUTH" for row in rows)


@pytest.mark.parametrize("option", [
    "ACCEPT_AS_CANDIDATE_FOR_NEXT_REVIEW", "REJECT_GEOMETRY_CANDIDATE", "REQUEST_MORE_EVIDENCE",
    "MARK_HAZARD_AMBIGUOUS", "KEEP_GEOMETRY_MISSING",
])
def test_adjudication_options(option):
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[8]))
    assert all(option in row["decision_options"] for row in rows)


def test_audit_report_counts():
    rows = {row["metric"]: row["value"] for row in common.load_csv(common.dataset_path(common.OUTPUTS[9]))}
    assert rows["review_ready_packets"] == "6"
    assert rows["recife_gap_packets"] == "3"
    assert rows["candidate_geometries_created"] == "0"
    assert rows["null_geojsons"] == "6"
    assert rows["next_action_rank_1"] == "MANUAL_DIGITIZATION_FROM_REVIEWABLE_EVIDENCE_OR_SECONDARY_SOURCE_SEARCH"


@pytest.mark.parametrize("metric", ["ground_truth_created", "labels_created", "negatives_created", "training_runs"])
def test_audit_report_zero_promotions(metric):
    rows = {row["metric"]: row["value"] for row in common.load_csv(common.dataset_path(common.OUTPUTS[9]))}
    assert rows[metric] == "0"


@pytest.mark.parametrize("field", ["can_create_ground_truth", "can_create_patch_truth", "can_create_label", "can_create_negative", "can_train_model"])
def test_all_outputs_block_promotions(field):
    for output in common.OUTPUTS[:10]:
        assert all(row.get(field) == "false" for row in common.load_csv(common.dataset_path(output)))


def test_cache_marker():
    assert open(common.doc_path("evidence_cache", ".gitignore"), encoding="utf-8").read() == "*\n!.gitignore\n"


def test_guardrails_pass():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[10]))
    assert len(rows) == 11
    assert all(row["status"] == "PASS" and row["violation_count"] == "0" for row in rows)


def test_orchestrator_ok():
    rows = common.load_csv(common.dataset_path("v2ba_orchestrator_manifest.csv"))
    assert len(rows) == 11
    assert all(row["status"] == "OK" for row in rows)
