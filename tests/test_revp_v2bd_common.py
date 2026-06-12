import os

import pytest

import scripts.protocolo_c.revp_v2bd_common as common


@pytest.fixture(scope="module", autouse=True)
def generated():
    common.run_orchestrator()


@pytest.mark.parametrize("field,expected", list(common.INVARIANTS.items()))
def test_invariants(field, expected):
    assert common.with_invariants({})[field] == expected


@pytest.mark.parametrize("start,end,acquired,delta,status", [
    ("2023-10-28","2023-10-30","2023-10-28","0","WITHIN_EVENT_WINDOW"),
    ("2023-10-28","2023-10-30","2023-10-29","0","WITHIN_EVENT_WINDOW"),
    ("2023-10-28","2023-10-30","2023-10-30","0","WITHIN_EVENT_WINDOW"),
    ("2023-10-28","2023-10-30","2023-10-31","1","OUTSIDE_EVENT_WINDOW"),
    ("2023-10-28","2023-10-30","2023-10-25","3","OUTSIDE_EVENT_WINDOW"),
    ("2023-10-28","2023-10-30","","","UNKNOWN_DATE_MISSING"),
    ("","","2023-10-29","","UNKNOWN_DATE_MISSING"),
])
def test_temporal_match(start, end, acquired, delta, status):
    assert common.temporal_match(start, end, acquired) == (delta, status)


@pytest.mark.parametrize("seed_patch,asset_patch,same_region,expected", [
    ("CUR_00038","CUR_00038",True,"EXPLICIT_PATCH_MATCH"),
    ("CUR_00038","CUR_00249",True,"REGION_ONLY_NOT_PATCH_MATCH"),
    ("","CUR_00038",True,"REGION_ONLY_NOT_PATCH_MATCH"),
    ("","CUR_00038",False,"NO_EXPLICIT_PATCH_MATCH"),
    ("CUR_00038","",False,"NO_EXPLICIT_PATCH_MATCH"),
])
def test_patch_match(seed_patch, asset_patch, same_region, expected):
    assert common.patch_match(seed_patch, asset_patch, same_region) == expected


@pytest.mark.parametrize("city,station,role,status,score,expected", [
    ("Curitiba","A807","LOCAL","CANDIDATE_GROUND_TRUTH_SEED","MODERATE_SEED_FOR_REVIEW",True),
    ("Curitiba","A807","LOCAL","CANDIDATE_GROUND_TRUTH_SEED","STRONG_SEED_FOR_REVIEW",True),
    ("Petropolis","A610","REGIONAL_PROXY","CANDIDATE_GROUND_TRUTH_SEED","MODERATE_SEED_FOR_REVIEW",False),
    ("Recife","A301","LOCAL_WITH_TEMPORAL_GAP","CANDIDATE_GROUND_TRUTH_SEED","MODERATE_SEED_FOR_REVIEW",False),
    ("Curitiba","A610","REGIONAL_PROXY","CANDIDATE_GROUND_TRUTH_SEED","MODERATE_SEED_FOR_REVIEW",False),
    ("Curitiba","A807","LOCAL","NOT_A_SEED","MODERATE_SEED_FOR_REVIEW",False),
    ("Curitiba","A807","LOCAL","CANDIDATE_GROUND_TRUTH_SEED","WEAK_SEED",False),
])
def test_eligible_seed(city, station, role, status, score, expected):
    assert common.eligible_seed({"city":city,"seed_status":status},{"station_id":station,"station_role":role},{"score_class":score}) is expected


@pytest.mark.parametrize("found,patch,region,date,expected", [
    (False,False,False,"","NO_ASSET_FOUND"),
    (True,True,False,"2023-10-29","EXPLICIT_SEED_ASSET_LINK"),
    (True,True,False,"","CANDIDATE_SEED_ASSET_LINK"),
    (True,False,True,"","NEEDS_MANUAL_REVIEW"),
    (True,False,False,"","CANDIDATE_SEED_ASSET_LINK"),
])
def test_seed_asset_crosswalk_status(found, patch, region, date, expected):
    assert common.seed_asset_crosswalk_status(found, patch, region, date) == expected


@pytest.mark.parametrize("available,linked,expected", [
    (True,True,"AVAILABLE_FOR_REVIEW"), (True,False,"NOT_LINKED"), (False,False,"MISSING"),
])
def test_dino_review_status(available, linked, expected):
    assert common.dino_review_status(available, linked) == expected


@pytest.mark.parametrize("patch,sentinel,date,dino,geometry,human,expected", [
    (False,False,False,False,False,False,"CANDIDATE_REFERENCE_BLOCKED_BY_GEOMETRY"),
    (True,True,True,True,False,True,"CANDIDATE_REFERENCE_BLOCKED_BY_GEOMETRY"),
    (False,True,True,True,True,True,"CANDIDATE_REFERENCE_NEEDS_SENTINEL_CROSSWALK"),
    (True,False,True,True,True,True,"CANDIDATE_REFERENCE_NEEDS_SENTINEL_CROSSWALK"),
    (True,True,False,True,True,True,"CANDIDATE_REFERENCE_NEEDS_SENTINEL_CROSSWALK"),
    (True,True,True,True,True,False,"CANDIDATE_REFERENCE_NEEDS_HUMAN_REVIEW"),
    (True,True,True,False,True,True,"CANDIDATE_REFERENCE_READY_WITHOUT_DINO"),
    (True,True,True,True,True,True,"CANDIDATE_REFERENCE_READY_FOR_ADJUDICATION"),
])
def test_readiness_class(patch, sentinel, date, dino, geometry, human, expected):
    assert common.readiness_class(patch, sentinel, date, dino, geometry, human) == expected


@pytest.mark.parametrize("readiness,temporal,geometry,human,status,allowed", [
    ("CANDIDATE_REFERENCE_READY_FOR_ADJUDICATION","STRONG",True,True,"CANDIDATE_REFERENCE_FOR_ADJUDICATION",True),
    ("CANDIDATE_REFERENCE_READY_WITHOUT_DINO","STRONG",True,True,"CANDIDATE_REFERENCE_FOR_ADJUDICATION",True),
    ("CANDIDATE_REFERENCE_READY_FOR_ADJUDICATION","WEAK",True,True,"REMAIN_CANDIDATE_GROUND_TRUTH_SEED",False),
    ("CANDIDATE_REFERENCE_READY_FOR_ADJUDICATION","STRONG",False,True,"REMAIN_CANDIDATE_GROUND_TRUTH_SEED",False),
    ("CANDIDATE_REFERENCE_READY_FOR_ADJUDICATION","STRONG",True,False,"REMAIN_CANDIDATE_GROUND_TRUTH_SEED",False),
    ("CANDIDATE_REFERENCE_BLOCKED_BY_GEOMETRY","STRONG",False,False,"REMAIN_CANDIDATE_GROUND_TRUTH_SEED",False),
])
def test_promotion_decision(readiness, temporal, geometry, human, status, allowed):
    assert common.promotion_decision(readiness, temporal, geometry, human) == (status, allowed)


@pytest.mark.parametrize("output", common.OUTPUTS + ["v2bd_orchestrator_manifest.csv"])
def test_outputs_exist(output):
    assert os.path.exists(common.dataset_path(output))
    assert common.load_csv(common.dataset_path(output))


def test_three_selected_seeds():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[0]))
    assert len(rows) == 3
    assert all(row["city"] == "Curitiba" for row in rows)
    assert all(row["selection_status"] == "SELECTED_FOR_SEED_SENTINEL_CROSSWALK" for row in rows)


@pytest.mark.parametrize("candidate_id", ["CTB_2023_10_28_30","CTB_2022_01_15_16","CTB_2024_02_18_20"])
def test_expected_candidate_selected(candidate_id):
    assert sum(row["candidate_id"] == candidate_id for row in common.load_csv(common.dataset_path(common.OUTPUTS[0]))) == 1


def test_real_assets_discovered_review_only():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[1]))
    assert len(rows) > 0
    assert all(row["discovery_status"] == "DISCOVERED_REVIEW_ONLY_DATE_MISSING" for row in rows)
    assert all(row["acquisition_date"] == "" and row["dino_allowed_use"] == "REVIEW_ONLY_REPRESENTATION" for row in rows)


def test_crosswalk_is_cartesian_audit_without_explicit_link():
    assets = common.load_csv(common.dataset_path(common.OUTPUTS[1]))
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[2]))
    assert len(rows) == 3 * len(assets)
    assert all(row["explicit_seed_asset_link"] == "false" for row in rows)
    assert all(row["patch_match_status"] == "CITY_REGION_MATCH" for row in rows)


def test_crosswalk_date_missing():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[2]))
    assert all(row["temporal_match_status"] == "UNKNOWN" for row in rows)
    assert all(row["crosswalk_status"] == "NEEDS_MANUAL_REVIEW" for row in rows)


def test_seed_patch_crosswalk_blocked():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[3]))
    assert len(rows) == 3
    assert all(row["crosswalk_status"] == "NO_EXPLICIT_PATCH_LINK" for row in rows)
    assert all(row["geometry_status"] == "GEOMETRY_MISSING" for row in rows)


def test_seed_dino_crosswalk_not_linked():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[4]))
    assert len(rows) == 3
    assert all(row["dino_link_status"] == "NOT_LINKED" and row["embedding_available"] == "false" for row in rows)
    assert all(row["dino_decision_allowed"] == "false" for row in rows)


def test_visual_assets_regional_only():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[5]))
    assert len(rows) == 3 * len(common.load_csv(common.dataset_path(common.OUTPUTS[1])))
    assert all(row["ready_for_regional_human_visual_review"] == "true" for row in rows)
    assert all(row["ready_for_seed_specific_adjudication"] == "false" for row in rows)


def test_readiness_blocked_by_geometry():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[6]))
    assert len(rows) == 3
    assert all(row["candidate_reference_readiness"] == "CANDIDATE_REFERENCE_BLOCKED_BY_GEOMETRY" for row in rows)
    assert all(row["temporal_support_strong"] == "true" and row["geometry_available"] == "false" for row in rows)


def test_all_blockers_explicit():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[6]))
    required = {"NO_EXPLICIT_PATCH_LINK","NO_EXPLICIT_SENTINEL_LINK","SENTINEL_DATE_MISSING","GEOMETRY_MISSING","HUMAN_REVIEW_PENDING"}
    assert all(required == set(row["blocking_factors"].split("|")) for row in rows)


def test_zero_candidate_reference_promotions_real_outputs():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[7]))
    assert len(rows) == 3
    assert all(row["promotion_allowed"] == "false" and row["gate_status"] == "PROMOTION_BLOCKED" for row in rows)
    assert all(row["proposed_status"] == "REMAIN_CANDIDATE_GROUND_TRUTH_SEED" for row in rows)


def test_next_action_rank_1():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[7]))
    assert all(row["next_action_rank_1"] == "RESOLVE_CURITIBA_SENTINEL_ASSET_CROSSWALK" for row in rows)


def test_three_packets():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[8]))
    assert len(rows) == 3
    assert all(row["promotion_allowed"] == "false" for row in rows)


@pytest.mark.parametrize("folder", ["candidate_reference_packets","crosswalk_tables","visual_review_assets","evidence_cache"])
def test_doc_folder_exists(folder):
    assert os.path.isdir(common.doc_path(folder))


def test_packet_docs_guardrail_language():
    files = [name for name in os.listdir(common.doc_path("candidate_reference_packets")) if name.endswith(".md")]
    assert len(files) == 3
    for name in files:
        text = open(common.doc_path("candidate_reference_packets", name), encoding="utf-8").read()
        assert "Nao e ground truth final" in text
        assert "PROMOTION_BLOCKED" in text


@pytest.mark.parametrize("field", ["can_create_ground_truth","can_create_patch_truth","can_create_label","can_create_negative","can_train_model"])
def test_zero_truth_labels_negatives_training(field):
    for output in common.OUTPUTS[:9]:
        assert all(row.get(field) == "false" for row in common.load_csv(common.dataset_path(output)))


def test_cache_marker_only_policy():
    assert open(common.doc_path("evidence_cache", ".gitignore"), encoding="utf-8").read() == "*\n!.gitignore\n"


def test_guardrails_pass():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[9]))
    assert len(rows) == 10
    assert all(row["status"] == "PASS" and row["violation_count"] == "0" for row in rows)


def test_orchestrator_all_ok():
    rows = common.load_csv(common.dataset_path("v2bd_orchestrator_manifest.csv"))
    assert len(rows) == 10
    assert all(row["status"] == "OK" for row in rows)
