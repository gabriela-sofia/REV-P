import os

import pytest

import scripts.protocolo_c.revp_v2be_common as common


@pytest.fixture(scope="module", autouse=True)
def generated():
    common.run_orchestrator()


@pytest.mark.parametrize("field,expected", list(common.INVARIANTS.items()))
def test_invariants(field, expected):
    assert common.with_invariants({})[field] == expected


@pytest.mark.parametrize("event,start,end,acquired,delta,within,expanded,score", [
    ("2023-10-28","2023-10-21","2023-11-02","2023-10-28","0",True,True,40),
    ("2023-10-28","2023-10-21","2023-11-02","2023-10-21","7",True,True,40),
    ("2023-10-28","2023-10-21","2023-11-02","2023-11-10","13",False,True,25),
    ("2023-10-28","2023-10-21","2023-11-02","2023-12-20","53",False,False,0),
    ("2023-10-28","2023-10-21","2023-11-02","","",False,False,0),
])
def test_date_features(event, start, end, acquired, delta, within, expanded, score):
    assert common.date_features(event, start, end, acquired) == (delta, within, expanded, score)


@pytest.mark.parametrize("kwargs,final,classification", [
    ({"event_date":"2023-10-28","window_start":"2023-10-21","window_end":"2023-11-02","acquisition_date":"2023-10-28","patch_match":True,"visual":True,"spectral":True,"dino":True,"public":True},100,"BEST_REVIEW_ASSET"),
    ({"event_date":"2023-10-28","window_start":"2023-10-21","window_end":"2023-11-02","acquisition_date":"2023-10-29","city_region":True,"visual":True},55,"GOOD_REVIEW_ASSET"),
    ({"city_region":True,"visual":True},5,"WEAK_REVIEW_ASSET"),
    ({"city_region":True,"visual":True,"spectral":True,"dino":True,"public":True},20,"WEAK_REVIEW_ASSET"),
    ({"city_region":False,"visual":False},-35,"NOT_RECOMMENDED"),
])
def test_asset_score(kwargs, final, classification):
    result = common.asset_score(**kwargs)
    assert result[8] == final and result[9] == classification


@pytest.mark.parametrize("score,patch,date,visual,link,confidence", [
    ("BEST_REVIEW_ASSET",True,True,True,"EXPLICIT_PATCH_DATE_LINK","HIGH"),
    ("GOOD_REVIEW_ASSET",False,True,True,"CANDIDATE_TEMPORAL_REGIONAL_LINK","MODERATE"),
    ("WEAK_REVIEW_ASSET",False,False,True,"CANDIDATE_VISUAL_REVIEW_LINK","LOW"),
    ("NOT_RECOMMENDED",False,False,True,"NO_LINK","VERY_LOW"),
    ("WEAK_REVIEW_ASSET",False,False,False,"WEAK_CONTEXT_LINK","VERY_LOW"),
])
def test_link_class(score, patch, date, visual, link, confidence):
    assert common.link_class(score, patch, date, visual) == (link, confidence)


@pytest.mark.parametrize("path,nonempty,spectral,expected", [
    ("asset.tif",True,True,"READY_FOR_HUMAN_VISUAL_REVIEW"), ("asset.tif",True,False,"NEEDS_ASSET_RENDERING"),
    ("asset.tif",False,False,"ASSET_REFERENCE_ONLY"), ("",False,False,"MISSING"),
])
def test_visual_status(path, nonempty, spectral, expected):
    assert common.visual_status(path, nonempty, spectral) == expected


@pytest.mark.parametrize("confidence,visual,primary,patch,expected", [
    ("HIGH","READY_FOR_HUMAN_VISUAL_REVIEW",True,True,"READY_FOR_CANDIDATE_REFERENCE_ADJUDICATION"),
    ("MODERATE","ASSET_REFERENCE_ONLY",True,False,"READY_FOR_CANDIDATE_REFERENCE_ADJUDICATION"),
    ("LOW","ASSET_REFERENCE_ONLY",True,False,"READY_FOR_HUMAN_VISUAL_REVIEW_ONLY"),
    ("LOW","NEEDS_ASSET_RENDERING",True,False,"NEEDS_ASSET_RENDERING"),
    ("LOW","MISSING",True,False,"NEEDS_EXPLICIT_PATCH_ASSET_LINK"),
    ("LOW","MISSING",False,False,"REMAINS_SEED_ONLY"),
])
def test_readiness_status(confidence, visual, primary, patch, expected):
    assert common.readiness_status(confidence, visual, primary, patch) == expected


@pytest.mark.parametrize("readiness,confidence,human,status,allowed,typ", [
    ("READY_FOR_CANDIDATE_REFERENCE_ADJUDICATION","HIGH",True,"CANDIDATE_REFERENCE_FOR_ADJUDICATION",True,"METHODOLOGICAL_ONLY"),
    ("READY_FOR_CANDIDATE_REFERENCE_ADJUDICATION","MODERATE",True,"CANDIDATE_REFERENCE_FOR_ADJUDICATION",True,"METHODOLOGICAL_ONLY"),
    ("READY_FOR_CANDIDATE_REFERENCE_ADJUDICATION","LOW",True,"REMAIN_CANDIDATE_GROUND_TRUTH_SEED",False,"NONE"),
    ("READY_FOR_CANDIDATE_REFERENCE_ADJUDICATION","HIGH",False,"REMAIN_CANDIDATE_GROUND_TRUTH_SEED",False,"NONE"),
    ("READY_FOR_HUMAN_VISUAL_REVIEW_ONLY","LOW",True,"READY_FOR_HUMAN_VISUAL_REVIEW_ONLY",False,"NONE"),
])
def test_revised_gate(readiness, confidence, human, status, allowed, typ):
    assert common.revised_gate(readiness, confidence, human) == (status, allowed, typ)


def test_best_selection_tie_breaker_and_alternates():
    rows = [{"sentinel_asset_id":f"A{i}","final_asset_match_score":"5","score_class":"WEAK_REVIEW_ASSET"} for i in range(5)]
    primary, alternates, status = common.best_selection(rows)
    assert primary["sentinel_asset_id"] == "A0" and len(alternates) == 3 and status == "ONLY_WEAK_ASSETS_AVAILABLE"


def test_best_selection_no_assets():
    assert common.best_selection([]) == (None, [], "NO_REVIEW_ASSET_FOUND")


@pytest.mark.parametrize("output", common.OUTPUTS + ["v2be_orchestrator_manifest.csv"])
def test_outputs_exist(output):
    assert os.path.exists(common.dataset_path(output))
    assert common.load_csv(common.dataset_path(output))


def test_three_blocked_seeds_selected():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[0]))
    assert len(rows) == 3
    assert all(r["previous_gate_status"] == "PROMOTION_BLOCKED" and r["station_id"] == "A807" for r in rows)


@pytest.mark.parametrize("candidate", ["SEED_v2bc_0001","SEED_v2bc_0002","SEED_v2bc_0003"])
def test_expected_seed_selected(candidate):
    assert sum(r["seed_id"] == candidate for r in common.load_csv(common.dataset_path(common.OUTPUTS[0]))) == 1


def test_129_scores():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[1]))
    assert len(rows) == 129
    assert all(r["sentinel_acquisition_date"] == "" and r["date_delta_days"] == "" for r in rows)


def test_actual_scores_weak_review_only():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[1]))
    assert all(r["score_class"] == "WEAK_REVIEW_ASSET" for r in rows)
    assert all(int(r["penalty_score"]) < 0 for r in rows)


def test_actual_region_only_and_visual_reference():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[1]))
    assert all(r["city_region_match"] == "true" and r["patch_id_match"] == "false" for r in rows)
    assert all(r["visual_asset_available"] == "true" and r["spectral_context_available"] == "false" for r in rows)


def test_129_links_low_visual_review():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[2]))
    assert len(rows) == 129
    assert all(r["link_type"] == "CANDIDATE_VISUAL_REVIEW_LINK" and r["link_confidence"] == "LOW" for r in rows)
    assert all(r["requires_human_review"] == "true" and r["link_is_ground_truth"] == "false" for r in rows)


def test_best_asset_per_seed_deterministic():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[3]))
    assert len(rows) == 3
    assert all(r["primary_sentinel_asset_id"] == "V1PU_VA_00001" for r in rows)
    assert all(r["selection_status"] == "ONLY_WEAK_ASSETS_AVAILABLE" for r in rows)
    assert all(len(r["alternate_asset_ids"].split("|")) == 3 for r in rows)


def test_visual_binding_reference_only():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[4]))
    assert len(rows) == 3
    assert all(r["visual_review_status"] == "ASSET_REFERENCE_ONLY" for r in rows)
    assert all(r["visual_is_ground_truth"] == "false" for r in rows)


def test_dino_not_linked():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[5]))
    assert len(rows) == 3
    assert all(r["dino_link_status"] == "NOT_LINKED" and r["embedding_available"] == "false" for r in rows)
    assert all(r["dino_is_ground_truth"] == "false" and r["dino_can_create_label"] == "false" for r in rows)


def test_readiness_visual_review_only():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[6]))
    assert len(rows) == 3
    assert all(r["updated_readiness_status"] == "READY_FOR_HUMAN_VISUAL_REVIEW_ONLY" for r in rows)
    assert all("SENTINEL_DATE_MISSING" in r["remaining_blockers"] for r in rows)


def test_revised_gate_zero_promotion():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[7]))
    assert len(rows) == 3
    assert all(r["proposed_status"] == "READY_FOR_HUMAN_VISUAL_REVIEW_ONLY" for r in rows)
    assert all(r["promotion_allowed"] == "false" and r["promotion_type"] == "NONE" for r in rows)


def test_next_action():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[7]))
    assert all(r["next_action_rank_1"] == "MANUALLY_RESOLVE_PATCH_ASSET_LINK_FOR_CURITIBA" for r in rows)


def test_three_packets():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[8]))
    assert len(rows) == 3
    assert all(r["primary_sentinel_asset_id"] == "V1PU_VA_00001" for r in rows)


@pytest.mark.parametrize("folder", ["visual_review_packets","crosswalk_tables","asset_score_summaries","evidence_cache"])
def test_doc_folders(folder):
    assert os.path.isdir(common.doc_path(folder))


def test_packet_markdowns():
    files = [f for f in os.listdir(common.doc_path("visual_review_packets")) if f.endswith(".md")]
    assert len(files) == 3
    for name in files:
        text = open(common.doc_path("visual_review_packets", name), encoding="utf-8").read()
        assert "GEOMETRY_MISSING" in text and "Crosswalk candidato nao e truth" in text


@pytest.mark.parametrize("field", ["can_create_ground_truth","can_create_patch_truth","can_create_label","can_create_negative","can_train_model"])
def test_zero_forbidden_outputs(field):
    for output in common.OUTPUTS[:9]:
        assert all(r[field] == "false" for r in common.load_csv(common.dataset_path(output)))


def test_cache_marker():
    assert open(common.doc_path("evidence_cache", ".gitignore"), encoding="utf-8").read() == "*\n!.gitignore\n"


def test_guardrails_pass():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[9]))
    assert len(rows) == 10 and all(r["status"] == "PASS" for r in rows)


def test_orchestrator_ok():
    rows = common.load_csv(common.dataset_path("v2be_orchestrator_manifest.csv"))
    assert len(rows) == 10 and all(r["status"] == "OK" for r in rows)
