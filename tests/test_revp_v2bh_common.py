import os

import pytest

import scripts.protocolo_c.revp_v2bh_common as common


@pytest.fixture(scope="module", autouse=True)
def generated():
    common.run_orchestrator()


@pytest.mark.parametrize("field,expected", list(common.INVARIANTS.items()))
def test_invariants(field, expected):
    assert common.with_invariants({})[field] == expected


@pytest.mark.parametrize("input_name", ["charter", "priority", "evidence", "gates", "tasks", "recife_queue", "temporal"])
def test_required_existing_inputs(input_name):
    assert common.load_inputs()[input_name]


def test_missing_v2bg_gap_selection_is_explicit():
    assert common.load_inputs()["gap_selection"] == []


@pytest.mark.parametrize("output", common.OUTPUTS + ["v2bh_orchestrator_manifest.csv"])
def test_outputs_exist(output):
    assert os.path.exists(common.dataset_path(output))
    assert common.load_csv(common.dataset_path(output))


def inventory():
    return common.load_csv(common.dataset_path(common.OUTPUTS[0]))


def test_activation_758_loaded():
    assert common.run_load_charter_registry()["charter"][0]["charter_activation_id"] == "758"


def test_inventory_preserves_count_51():
    assert all(row["activation_product_count_reported"] == "51" for row in inventory())


def test_recife_product_loaded():
    row = next(r for r in inventory() if r["product_id"] == "CH758_RECIFE_20220602_001")
    assert row["product_title"] == "Landslides after effects in Recife/PE - Brazil"
    assert row["product_date"] == "2022-06-02"
    assert row["product_inventory_status"] == "CONFIRMED_FROM_REGISTRY"


def test_olinda_product_not_invented():
    row = next(r for r in inventory() if "OLINDA" in r["product_id"])
    assert row["product_title"] == ""
    assert row["product_date"] == "2022-06-03"
    assert row["product_inventory_status"] == "MANUAL_REVIEW_REQUIRED"


def test_remaining_products_not_invented():
    row = next(r for r in inventory() if "REMAINING" in r["product_id"])
    assert row["product_title"] == "" and row["product_date"] == ""
    assert "not inventoried offline" in row["note"]


@pytest.mark.parametrize("area,classification,apply,blocker", [
    ("Recife/PE", "RECIFE", "true", ""),
    ("Olinda/PE", "OLINDA", "false", "OLINDA_PRODUCT_NOT_TRANSFERABLE_TO_RECIFE"),
    ("Recife Olinda/PE", "RECIFE_OLINDA", "true", ""),
    ("Pernambuco", "PERNAMBUCO_REGIONAL", "false", "RECIFE_AREA_NOT_EXPLICIT"),
    ("Unknown", "UNKNOWN", "false", "PRODUCT_AREA_UNKNOWN"),
])
def test_municipality_classification(area, classification, apply, blocker):
    result = common.municipality_classification({"product_area_text": area})
    assert result[:3] == (classification, apply, blocker)


def test_classification_output_separates_olinda():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[1]))
    recife = next(r for r in rows if r["municipality_classification"] == "RECIFE")
    olinda = next(r for r in rows if r["municipality_classification"] == "OLINDA")
    assert recife["can_apply_to_recife"] == "true"
    assert olinda["can_apply_to_recife"] == "false"


@pytest.mark.parametrize("vector,crs,raster,map_file,preview,status,blocker", [
    (True, True, False, False, False, "VECTOR_ACCESS_CONFIRMED", ""),
    (True, False, False, False, False, "ACCESS_NOT_CONFIRMED", "CRS_NOT_CONFIRMED"),
    (False, False, True, False, False, "RASTER_OR_MAP_ONLY", "VECTOR_AND_CRS_NOT_CONFIRMED"),
    (False, False, False, True, False, "RASTER_OR_MAP_ONLY", "VECTOR_AND_CRS_NOT_CONFIRMED"),
    (False, False, False, False, True, "PREVIEW_ONLY", "PREVIEW_IS_NOT_VECTOR_GEOMETRY"),
    (False, False, False, False, False, "ACCESS_NOT_CONFIRMED", "DOWNLOAD_VECTOR_CRS_LICENSE_NOT_CONFIRMED"),
])
def test_access_status(vector, crs, raster, map_file, preview, status, blocker):
    assert common.access_status(vector, crs, raster, map_file, preview) == (status, blocker)


@pytest.mark.parametrize("field", ["download_link_found", "vector_file_found", "raster_file_found", "pdf_or_image_found", "preview_only", "crs_confirmed", "license_or_terms_confirmed", "redistribution_allowed_confirmed", "raw_payload_cached"])
def test_offline_access_not_confirmed(field):
    assert all(r[field] == "false" for r in common.load_csv(common.dataset_path(common.OUTPUTS[2])))


def test_offline_network_status():
    assert all(r["network_mode"] == "NETWORK_DISABLED_DETERMINISTIC_RUN" for r in common.load_csv(common.dataset_path(common.OUTPUTS[2])))


def test_landslide_product_not_flood_extent():
    row = next(r for r in common.load_csv(common.dataset_path(common.OUTPUTS[3])) if r["product_id"] == "CH758_RECIFE_20220602_001")
    assert row["hazard_type_candidate"] == "LANDSLIDE"
    assert row["geometry_feature_type_candidate"] == "UNKNOWN"
    assert row["can_support_flood_truth"] == "false"
    assert row["can_support_landslide_truth"] == "true"


@pytest.mark.parametrize("apply,access,vector,crs,expected", [
    (False, "ACCESS_NOT_CONFIRMED", False, False, "CONTEXT_ONLY"),
    (True, "PREVIEW_ONLY", False, False, "PREVIEW_ONLY_NOT_READY"),
    (True, "ACCESS_NOT_CONFIRMED", False, False, "CANDIDATE_GEOMETRY_SOURCE_PENDING_VECTOR_CRS"),
    (True, "RASTER_OR_MAP_ONLY", False, False, "CANDIDATE_GEOMETRY_SOURCE_PENDING_VECTOR_CRS"),
    (True, "VECTOR_ACCESS_CONFIRMED", True, True, "CANDIDATE_GEOMETRY_SOURCE_READY_FOR_HUMAN_REVIEW"),
    (True, "UNKNOWN", False, False, "BLOCKED"),
])
def test_candidate_status(apply, access, vector, crs, expected):
    assert common.candidate_status(apply, access, vector, crs) == expected


def test_recife_candidate_pending_vector_crs():
    row = next(r for r in common.load_csv(common.dataset_path(common.OUTPUTS[4])) if r["product_id"] == "CH758_RECIFE_20220602_001")
    assert row["candidate_status"] == "CANDIDATE_GEOMETRY_SOURCE_PENDING_VECTOR_CRS"
    assert row["required_human_action"] == "REQUEST_OR_MANUALLY_ACCESS_CHARTER_PRODUCT_VECTOR_CRS"


def test_olinda_candidate_context_only():
    row = next(r for r in common.load_csv(common.dataset_path(common.OUTPUTS[4])) if "OLINDA" in r["product_id"])
    assert row["candidate_status"] == "CONTEXT_ONLY"


def gate(candidate, gate_id):
    return next(r for r in common.load_csv(common.dataset_path(common.OUTPUTS[5])) if r["candidate_id"] == candidate and r["previous_gate_id"] == gate_id)


@pytest.mark.parametrize("gate_id,status", [
    ("C0_PROVENANCE", "PASS"), ("C1_TEMPORALITY", "PENDING"),
    ("C2_VALID_SERIES_OR_STATION", "BLOCKED"), ("C3_SPATIAL_ANCHOR", "PASS"),
    ("C4_CANDIDATE_GEOMETRY", "PENDING_VECTOR_CRS"), ("C5_HUMAN_REVIEW", "PENDING"),
    ("C6_CANDIDATE_REFERENCE", "BLOCKED"), ("C7_FINAL_GROUND_TRUTH", "BLOCKED"),
])
def test_may_2022_gate_updates(gate_id, status):
    assert gate("REC_2022_05_24_30", gate_id)["updated_gate_status"] == status


def test_other_recife_events_not_advanced():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[5]))
    other = [r for r in rows if r["candidate_id"] != "REC_2022_05_24_30"]
    assert all(r["evidence_used"] == "" for r in other)
    assert all(r["updated_gate_status"] == "BLOCKED" for r in other if r["previous_gate_id"] in {"C3_SPATIAL_ANCHOR", "C4_CANDIDATE_GEOMETRY"})


def test_next_actions():
    rows = [r for r in common.load_csv(common.dataset_path(common.OUTPUTS[5])) if r["candidate_id"] == "REC_2022_05_24_30"]
    assert all(r["next_action_rank_1"] == "REQUEST_OR_MANUALLY_ACCESS_CHARTER_PRODUCT_VECTOR_CRS" for r in rows)
    assert all(r["parallel_action"] == "ACQUIRE_CEMADEN_APAC_RECIFE_MAY_2022_TEMPORAL_SERIES" for r in rows)


def test_may_2022_event_patch_package_trace():
    rows = [r for r in common.load_csv(common.dataset_path(common.OUTPUTS[5])) if r["candidate_id"] == "REC_2022_05_24_30"]
    assert rows and all(r["event_patch_package_id"] == "FACT_v2at_0005" for r in rows)


@pytest.mark.parametrize("action", ["OPEN_PRODUCT_PAGE", "VERIFY_DOWNLOAD_LINK", "REQUEST_VECTOR_FILE", "VERIFY_CRS", "VERIFY_LICENSE_TERMS", "VERIFY_FEATURE_TYPE", "SAVE_DERIVED_METADATA"])
def test_manual_tasks(action):
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[6]))
    assert any(r["required_action"] == action and r["priority"] == "P0" for r in rows)


def test_review_packet_generated():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[7]))
    assert len(rows) == 1 and rows[0]["municipality_classification"] == "RECIFE"
    assert os.path.exists(rows[0]["packet_path"])


@pytest.mark.parametrize("text", ["Activation 758", "Vetor, CRS e licenca", "Nao e ground truth final", "Produto Charter nao e ground truth final"])
def test_review_packet_content(text):
    path = common.load_csv(common.dataset_path(common.OUTPUTS[7]))[0]["packet_path"]
    assert text in open(path, encoding="utf-8").read()


@pytest.mark.parametrize("text", ["Charter 758 e P0", "C4 permanece `PENDING_VECTOR_CRS`", "C7 continua bloqueado"])
def test_readme_content(text):
    assert text in open(common.doc_path("README.md"), encoding="utf-8").read()


@pytest.mark.parametrize("field", ["can_create_ground_truth", "can_create_patch_truth", "can_create_label", "can_create_negative", "can_train_model", "raw_data_versioned"])
def test_zero_forbidden_outputs(field):
    for output in common.OUTPUTS[:8]:
        assert all(r[field] == "false" for r in common.load_csv(common.dataset_path(output)))


def test_cache_marker_only():
    cache = common.doc_path("evidence_cache")
    assert os.listdir(cache) == [".gitignore"]
    assert open(os.path.join(cache, ".gitignore"), encoding="utf-8").read() == "*\n!.gitignore\n"


def test_guardrails_pass():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[8]))
    assert len(rows) == 10 and all(r["status"] == "PASS" and r["violation_count"] == "0" for r in rows)


def test_orchestrator_ok():
    rows = common.load_csv(common.dataset_path("v2bh_orchestrator_manifest.csv"))
    assert len(rows) == 10 and all(r["status"] == "OK" for r in rows)
