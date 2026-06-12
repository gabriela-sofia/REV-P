import os

import pytest

import scripts.protocolo_c.revp_v2bg_common as common


@pytest.fixture(scope="module", autouse=True)
def generated():
    common.run_orchestrator()


@pytest.mark.parametrize("field,expected", list(common.INVARIANTS.items()))
def test_invariants(field, expected):
    assert common.with_invariants({})[field] == expected


@pytest.mark.parametrize("output", common.OUTPUTS + ["v2bg_orchestrator_manifest.csv"])
def test_outputs_exist(output):
    assert os.path.exists(common.dataset_path(output))
    assert common.load_csv(common.dataset_path(output))


def test_charter_758_p0_source():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[1]))
    row = next(r for r in rows if r["source_name"] == "International Charter Activation 758")
    assert row["priority"] == "P0" and row["can_support_spatial_anchor"] == "true"
    assert row["can_resolve_temporal_gap"] == "false" and row["context_only"] == "false"


def test_charter_product_metadata():
    row = common.load_csv(common.dataset_path(common.OUTPUTS[0]))[0]
    assert row["charter_activation_id"] == "758"
    assert row["product_date"] == "2022-06-02"
    assert row["product_count_reported"] == "51"
    assert row["product_area"] == "Recife/PE"
    assert row["event_type"] == "LANDSLIDE_FLOOD"
    assert row["product_status"] == "PRODUCT_PAGE_CONFIRMED"
    assert row["geometry_status"] == "PUBLISHED_MAP_GEOMETRY_VISIBLE_OR_PRODUCT_LISTED"
    assert row["vector_status"] == "VECTOR_NOT_CONFIRMED"
    assert row["crs_status"] == row["redistribution_status"] == "UNKNOWN"


@pytest.mark.parametrize("field", ["vector_download_confirmed", "crs_confirmed", "redistribution_confirmed", "can_create_ground_truth", "can_create_label"])
def test_charter_unresolved_or_forbidden(field):
    assert common.load_csv(common.dataset_path(common.OUTPUTS[0]))[0][field] == "false"


def test_charter_not_automatic_flood_extent():
    row = common.load_csv(common.dataset_path(common.OUTPUTS[0]))[0]
    assert row["landslide_scar_is_not_flood_extent"] == "true"
    assert "not automatic flood extent" in row["limitation"]


def test_evidence_axes_separated():
    rows = [r for r in common.load_csv(common.dataset_path(common.OUTPUTS[2])) if r["evidence_item"] == "Charter 758 Recife product"]
    assert {r["evidence_axis"] for r in rows} == {"OCCURRENCE", "SPATIALITY", "HAZARD_SEMANTICS"}
    assert all(r["supports_temporal_gap_resolution"] == "false" for r in rows)


def test_olinda_not_transferred():
    row = next(r for r in common.load_csv(common.dataset_path(common.OUTPUTS[2])) if "Olinda" in r["evidence_item"])
    assert row["supports_occurrence"] == "false" and row["supports_spatial_anchor"] == "false"
    assert row["context_only"] == "true"


def test_copernicus_not_found_not_positive():
    row = next(r for r in common.load_csv(common.dataset_path(common.OUTPUTS[2])) if "Copernicus" in r["evidence_item"])
    assert row["evidence_class"] == "NOT_FOUND_SEARCH_RESULT"
    assert row["supports_occurrence"] == row["supports_spatial_anchor"] == row["supports_geometry_candidate"] == "false"


def test_inmet_proxies_do_not_replace_a301():
    rows = [r for r in common.load_csv(common.dataset_path(common.OUTPUTS[1])) if r["source_status"] == "REGIONAL_PROXY"]
    assert {r["source_name"] for r in rows} == {"INMET A357 Palmares", "INMET A328 Surubim", "INMET A320 Joao Pessoa"}
    assert all(r["can_resolve_temporal_gap"] == "false" and r["regional_proxy_does_not_replace_a301"] == "true" for r in rows)


def gate(candidate, name):
    return next(r for r in common.load_csv(common.dataset_path(common.OUTPUTS[3])) if r["candidate_id"] == candidate and r["gate"] == name)


def test_may_2022_gate_progression_is_fail_closed():
    candidate = "REC_2022_05_24_30"
    assert gate(candidate, "C0_PROVENANCE")["status"] == "PASS"
    assert gate(candidate, "C1_TEMPORALITY")["status"] == "PENDING"
    assert gate(candidate, "C3_SPATIAL_ANCHOR")["status"] == "PENDING_REVIEW"
    assert gate(candidate, "C4_CANDIDATE_GEOMETRY")["status"] == "PENDING"
    assert gate(candidate, "C7_FINAL_GROUND_TRUTH")["status"] == "BLOCKED"


def test_charter_not_applied_to_other_recife_events():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[3]))
    other = [r for r in rows if r["candidate_id"] != "REC_2022_05_24_30"]
    assert other and all(r["charter_758_applicable"] == "false" for r in other)
    assert all(r["status"] == "BLOCKED" for r in other if r["gate"] in {"C3_SPATIAL_ANCHOR", "C4_CANDIDATE_GEOMETRY"})


def test_next_action_ranking():
    rows = [r for r in common.load_csv(common.dataset_path(common.OUTPUTS[3])) if r["candidate_id"] == "REC_2022_05_24_30"]
    assert all(r["next_action_rank_1"] == "INVENTORY_CHARTER_758_RECIFE_PRODUCTS_AND_ACCESS_VECTOR_CRS" for r in rows)
    assert all(r["next_action_rank_2"] == "ACQUIRE_CEMADEN_APAC_RECIFE_MAY_2022_TEMPORAL_SERIES" for r in rows)


def test_manual_tasks_complete():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[4]))
    assert len(rows) == 8 and all(r["priority"] == "P0" for r in rows)
    assert {"VECTOR_NOT_CONFIRMED", "CRS_UNKNOWN", "REDISTRIBUTION_UNKNOWN", "FEATURE_TYPE_NOT_CONFIRMED"} <= {r["blocker"] for r in rows}


@pytest.mark.parametrize("field", ["can_create_ground_truth", "can_create_patch_truth", "can_create_label", "can_create_negative", "can_train_model", "raw_data_versioned"])
def test_zero_forbidden_outputs(field):
    for output in common.OUTPUTS[:5]:
        assert all(r[field] == "false" for r in common.load_csv(common.dataset_path(output)))


def test_readme_required_section():
    text = open(common.doc_path("README.md"), encoding="utf-8").read()
    assert "International Charter Activation 758 as Recife P0 Evidence" in text
    assert "nao resolve sozinha o ground truth" in text


def test_cache_marker():
    assert open(common.doc_path("evidence_cache", ".gitignore"), encoding="utf-8").read() == "*\n!.gitignore\n"


def test_guardrails_pass():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[5]))
    assert len(rows) == 10 and all(r["status"] == "PASS" and r["violation_count"] == "0" for r in rows)


def test_orchestrator_ok():
    rows = common.load_csv(common.dataset_path("v2bg_orchestrator_manifest.csv"))
    assert len(rows) == 7 and all(r["status"] == "OK" for r in rows)
