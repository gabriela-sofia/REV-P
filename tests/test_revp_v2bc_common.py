import os

import pytest

import scripts.protocolo_c.revp_v2bc_common as common


@pytest.fixture(scope="module", autouse=True)
def generated():
    common.run_orchestrator()


@pytest.mark.parametrize("field,expected", list(common.INVARIANTS.items()))
def test_invariants(field, expected):
    assert common.with_invariants({})[field] == expected


@pytest.mark.parametrize("packet,metric,ready,expected", [
    ({"city":"Curitiba","region":"Curitiba","station_role":"LOCAL"},{"station_id":"A807","missing_rate":"0.0","precip_signal_status":"PRECIPITATION_PRESENT"},{"temporal_readiness_status":"TEMPORAL_EVIDENCE_READY_FOR_REVIEW"},True),
    ({"city":"Petropolis","region":"Petropolis","station_role":"REGIONAL_PROXY"},{"station_id":"A610","missing_rate":"0.0","precip_signal_status":"PRECIPITATION_PRESENT"},{"temporal_readiness_status":"TEMPORAL_EVIDENCE_READY_FOR_REVIEW"},False),
    ({"city":"Curitiba","region":"Curitiba","station_role":"LOCAL"},{"station_id":"A807","missing_rate":"1.0","precip_signal_status":"UNKNOWN"},{"temporal_readiness_status":"TEMPORAL_EVIDENCE_NOT_READY"},False),
    ({"city":"Curitiba","region":"Curitiba","station_role":"REGIONAL_PROXY"},{"station_id":"A807","missing_rate":"0.0","precip_signal_status":"PRECIPITATION_PRESENT"},{"temporal_readiness_status":"TEMPORAL_EVIDENCE_READY_FOR_REVIEW"},False),
])
def test_seed_selection_rule(packet, metric, ready, expected):
    assert common.is_curitiba_local_seed(packet, metric, ready) is expected


@pytest.mark.parametrize("value,expected", [
    ("evento_misto","MULTIHAZARD"),("flood_or_heavy_rain","URBAN_FLOODING"),("enxurrada","FLASH_FLOOD"),
    ("landslide","LANDSLIDE"),("inundacao","URBAN_FLOODING"),("","UNKNOWN"),
])
def test_phenomenon(value, expected):
    assert common.phenomenon(value) == expected


@pytest.mark.parametrize("sentinel,geometry,proxy,quality,final,classification", [
    (False,False,False,True,60,"MODERATE_SEED_FOR_REVIEW"),
    (True,False,False,True,70,"MODERATE_SEED_FOR_REVIEW"),
    (True,True,False,True,85,"STRONG_SEED_FOR_REVIEW"),
    (False,False,True,True,20,"WEAK_SEED"),
    (False,False,False,False,45,"WEAK_SEED"),
])
def test_score_seed(sentinel, geometry, proxy, quality, final, classification):
    result = common.score_seed(sentinel, geometry, proxy, quality)
    assert result[7] == final and result[8] == classification


@pytest.mark.parametrize("output", common.OUTPUTS + ["v2bc_orchestrator_manifest.csv"])
def test_outputs_exist(output):
    assert os.path.exists(common.dataset_path(output))
    assert common.load_csv(common.dataset_path(output))


def test_diagnosis_ranks_a807_first():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[0]))
    assert len(rows) == 3
    assert rows[0]["station_id"] == "A807"
    assert rows[0]["validation_strength_rank"] == "1"


def test_diagnosis_proxy_and_gap():
    rows = {row["station_id"]: row for row in common.load_csv(common.dataset_path(common.OUTPUTS[0]))}
    assert int(rows["A807"]["records_parsed"]) > int(rows["A301"]["records_parsed"])
    assert rows["A610"]["station_role"] == "REGIONAL_PROXY"
    assert "proxy" in rows["A610"]["limitation"].lower()
    assert "No usable" in rows["A301"]["limitation"]


def test_three_curitiba_candidates():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[1]))
    assert len(rows) == 3
    assert all(row["city"] == "Curitiba" and row["station_id"] == "A807" and row["station_role"] == "LOCAL" for row in rows)
    assert all(row["selected_as_seed_candidate"] == "true" for row in rows)


@pytest.mark.parametrize("field", [
    "seed_candidate_id","event_patch_package_id","patch_id","city","region","event_date","window_start","window_end",
    "station_id","station_name","station_role","missing_rate","precip_total_window","precip_max_1h","precip_max_24h",
    "precip_signal_status","temporal_evidence_strength","selected_as_seed_candidate","selection_reason",
])
def test_candidate_fields(field):
    assert field in common.load_csv(common.dataset_path(common.OUTPUTS[1]))[0]


def test_three_candidate_seeds_not_final_truth():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[2]))
    assert len(rows) == 3
    assert all(row["seed_status"] == "CANDIDATE_GROUND_TRUTH_SEED" for row in rows)
    assert all(row["can_create_ground_truth"] == "false" and row["seed_is_not_final_ground_truth"] == "true" for row in rows)


def test_seed_geometry_missing_and_high_uncertainty():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[2]))
    assert all(row["geometry_status"] == "GEOMETRY_MISSING" and row["uncertainty_level"] == "HIGH" for row in rows)
    assert all(row["spatial_support_level"] == "LOCAL_STATION_SUPPORT" for row in rows)


def test_evidence_bundle_four_per_seed():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[3]))
    assert len(rows) == 12
    assert {sum(row["seed_id"] == seed for row in rows) for seed in {row["seed_id"] for row in rows}} == {4}


@pytest.mark.parametrize("evidence_type", ["INMET_TEMPORAL_SERIES","PATCH_CONTEXT","TEXTUAL_CONTEXT","GEOMETRY_GAP"])
def test_evidence_types(evidence_type):
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[3]))
    assert sum(row["evidence_type"] == evidence_type for row in rows) == 3


def test_inmet_is_strong_temporal_not_spatial_truth():
    rows = [row for row in common.load_csv(common.dataset_path(common.OUTPUTS[3])) if row["evidence_type"] == "INMET_TEMPORAL_SERIES"]
    assert all(row["evidence_strength"] == "STRONG" and row["supports_temporal_component"] == "true" for row in rows)
    assert all(row["supports_spatial_component"] == "false" for row in rows)


def test_sentinel_crosscheck_missing():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[4]))
    assert len(rows) == 3
    assert all(row["sentinel_support_status"] == "MISSING" for row in rows)
    assert all(row["sentinel_is_ground_truth"] == "false" and row["dino_is_ground_truth"] == "false" for row in rows)


def test_sentinel_windows_available_but_no_crosswalk():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[4]))
    assert all(row["available_visual_context"] == "true" for row in rows)
    assert all(row["available_spectral_context"] == "false" for row in rows)


def test_dino_availability_does_not_promote():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[4]))
    assert all(row["dino_embedding_available"] == "false" for row in rows)
    assert all(row["dino_signal_is_not_ground_truth"] == "true" for row in rows)


def test_scores_moderate_due_geometry_gap():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[5]))
    assert len(rows) == 3
    assert all(row["score_class"] == "MODERATE_SEED_FOR_REVIEW" for row in rows)
    assert all(int(row["geometry_penalty"]) < 0 and int(row["uncertainty_penalty"]) < 0 for row in rows)


def test_non_selected_queue_six():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[6]))
    assert len(rows) == 6
    assert sum(row["reason_not_selected"] == "REGIONAL_PROXY_LIMITATION" for row in rows) == 3
    assert sum(row["reason_not_selected"] == "TEMPORAL_GAP" for row in rows) == 3


def test_non_selected_actions():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[6]))
    assert all(row["required_action"] == "RESOLVE_WITH_LOCAL_STATION" for row in rows if row["region"] == "Petropolis")
    assert all(row["required_action"] == "RESOLVE_WITH_CEMADEN" for row in rows if row["region"] == "Recife")


def test_seed_packets_three():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[7]))
    assert len(rows) == 3
    assert all(row["next_action_rank_1"] == "REVIEW_CURITIBA_SEEDS_WITH_SENTINEL_CONTEXT_AND_GEOMETRY_GAP" for row in rows)


def test_seed_markdowns_three_and_guardrail():
    paths = [name for name in os.listdir(common.doc_path("seed_review_packets")) if name.endswith(".md")]
    assert len(paths) == 3
    for name in paths:
        text = open(common.doc_path("seed_review_packets", name), encoding="utf-8").read()
        assert "CANDIDATE_GROUND_TRUTH_SEED, nao ground truth final" in text
        assert "GEOMETRY_MISSING" in text


def test_evidence_bundle_docs_three():
    assert len([name for name in os.listdir(common.doc_path("evidence_bundles")) if name.endswith(".md")]) == 3


@pytest.mark.parametrize("field", ["can_create_ground_truth","can_create_patch_truth","can_create_label","can_create_negative","can_train_model"])
def test_zero_promotions(field):
    for output in common.OUTPUTS[:8]:
        assert all(row.get(field) == "false" for row in common.load_csv(common.dataset_path(output)))


def test_cache_marker():
    assert open(common.doc_path("evidence_cache", ".gitignore"), encoding="utf-8").read() == "*\n!.gitignore\n"


def test_guardrails_pass():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[8]))
    assert len(rows) == 9
    assert all(row["status"] == "PASS" and row["violation_count"] == "0" for row in rows)


def test_orchestrator_ok():
    rows = common.load_csv(common.dataset_path("v2bc_orchestrator_manifest.csv"))
    assert len(rows) == 9
    assert all(row["status"] == "OK" for row in rows)
