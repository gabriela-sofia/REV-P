import os

import pytest

import scripts.protocolo_c.revp_v2bf_common as common


@pytest.fixture(scope="module", autouse=True)
def generated():
    common.run_orchestrator()


@pytest.mark.parametrize("field,expected", list(common.INVARIANTS.items()))
def test_invariants(field, expected):
    assert common.with_invariants({})[field] == expected


@pytest.mark.parametrize("fields,expected", [
    (["sentinel_asset_id","patch_id","acquisition_date","region","visual_reference","dino_embedding"], (True,True,True,True,True,True)),
    (["visual_asset_id"], (True,False,False,False,True,False)),
    (["canonical_patch_id","preview_file"], (False,True,False,False,True,False)),
    (["scene_date","city"], (False,False,True,True,False,False)),
    (["embedding_id","pca_1"], (False,False,False,False,False,True)),
])
def test_detect_fields(fields, expected):
    result = common.detect_fields(fields)
    assert tuple(result.values()) == expected


@pytest.mark.parametrize("values,date,source,method,confidence", [
    (("2023-10-28",),"2023-10-28","FIELD_OR_FILENAME_DATE","PATTERN_PARSE","MODERATE"),
    (("S2_20231028_TILE",),"2023-10-28","FIELD_OR_FILENAME_DATE","PATTERN_PARSE","MODERATE"),
    (("folder/2023_10_28/asset.tif",),"2023-10-28","FIELD_OR_FILENAME_DATE","PATTERN_PARSE","MODERATE"),
    (("patch_curitiba_00038.tif",),"","NO_PRODUCT_OR_SCENE_DATE","NO_ACCEPTABLE_DATE_EVIDENCE","UNKNOWN"),
    (("",),"","NO_PRODUCT_OR_SCENE_DATE","NO_ACCEPTABLE_DATE_EVIDENCE","UNKNOWN"),
])
def test_recover_date(values, date, source, method, confidence):
    assert common.recover_date(*values) == (date, source, method, confidence)


@pytest.mark.parametrize("explicit,filename,path,patch,source,method,confidence", [
    ("CUR_00038","","","CUR_00038","EXPLICIT_REGISTRY_FIELD","EXPLICIT_FIELD","HIGH"),
    ("","preview_CUR_00249.png","","CUR_00249","FILENAME_OR_PATH","PATTERN_PARSE","MODERATE"),
    ("","","data/sentinel/patch_curitiba_00350.tif","CUR_00350","FILENAME_OR_PATH","PATTERN_PARSE","MODERATE"),
    ("","","","","NO_PATCH_EVIDENCE","NO_LINK","UNKNOWN"),
])
def test_recover_patch(explicit, filename, path, patch, source, method, confidence):
    assert common.recover_patch(explicit, filename, path) == (patch, source, method, confidence)


@pytest.mark.parametrize("date,patch,visual,dino,expected", [
    ("HIGH","HIGH","HIGH","HIGH","HIGH"),
    ("HIGH","HIGH","HIGH","MODERATE","HIGH"),
    ("MODERATE","HIGH","MODERATE","NOT_LINKED","MODERATE"),
    ("UNKNOWN","HIGH","HIGH","MODERATE","LOW"),
    ("UNKNOWN","MODERATE","MODERATE","NOT_LINKED","LOW"),
    ("UNKNOWN","UNKNOWN","HIGH","HIGH","VERY_LOW"),
    ("UNKNOWN","HIGH","UNKNOWN","MODERATE","VERY_LOW"),
])
def test_confidence(date, patch, visual, dino, expected):
    assert common.confidence(date, patch, visual, dino) == expected


@pytest.mark.parametrize("date,patch,visual,overall,status,ready", [
    ("HIGH","HIGH","READY_FOR_HUMAN_REVIEW","HIGH","CANDIDATE_CROSSWALK_RECOVERED",True),
    ("MODERATE","MODERATE","REFERENCE_ONLY","MODERATE","CANDIDATE_CROSSWALK_RECOVERED",True),
    ("UNKNOWN","HIGH","READY_FOR_HUMAN_REVIEW","LOW","VISUAL_REVIEW_READY_WITH_WEAK_LINK",False),
    ("UNKNOWN","HIGH","NEEDS_ASSET_RENDERING","LOW","NEEDS_ASSET_RENDERING",False),
    ("UNKNOWN","UNKNOWN","MISSING","VERY_LOW","NEEDS_MANUAL_LINEAGE_RESOLUTION",False),
])
def test_crosswalk_update(date, patch, visual, overall, status, ready):
    assert common.crosswalk_update(date, patch, visual, overall) == (status, ready)


@pytest.mark.parametrize("output", common.OUTPUTS + ["v2bf_orchestrator_manifest.csv"])
def test_outputs_exist(output):
    assert os.path.exists(common.dataset_path(output))
    assert common.load_csv(common.dataset_path(output))


def test_lineage_sources_inventory():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[0]))
    assert len(rows) == len(common.LINEAGE_SOURCE_PATHS)
    assert all(r["source_file"] in common.LINEAGE_SOURCE_PATHS for r in rows)
    assert any(r["source_type"] == "LOCAL_RUN_REGISTRY" and r["usable_for_lineage"] == "true" for r in rows)
    assert any(r["source_type"] == "OUTPUT_PUBLIC_TABLE" and r["contains_dino_reference"] == "true" for r in rows)


@pytest.mark.parametrize("field", ["contains_asset_id","contains_patch_id","contains_date","contains_city_region","contains_visual_reference","contains_dino_reference","usable_for_lineage"])
def test_source_boolean_fields(field):
    assert all(r[field] in {"true","false"} for r in common.load_csv(common.dataset_path(common.OUTPUTS[0])))


def test_43_dates_unknown():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[1]))
    assert len(rows) == 43
    assert all(r["recovered_date"] == "" and r["date_confidence"] == "UNKNOWN" for r in rows)
    assert all("timestamps were rejected" in r["date_recovery_note"] for r in rows)


def test_43_asset_patch_links_high():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[2]))
    assert len(rows) == 43
    assert all(r["recovered_patch_id"].startswith("CUR_") and r["patch_link_confidence"] == "HIGH" for r in rows)
    assert all(r["patch_boundary_is_event_geometry"] == "false" for r in rows)


def test_file_reference_resolution():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[3]))
    assert len(rows) == 43
    assert all(r["asset_file_reference"].startswith("data/sentinel/") for r in rows)
    assert all(r["file_exists"] == "false" for r in rows)
    assert sum(r["usable_for_human_review"] == "true" for r in rows) >= 3


@pytest.mark.parametrize("patch", ["CUR_00038","CUR_00249","CUR_00350"])
def test_primary_patch_preview_exists(patch):
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[3]))
    match = next(r for r in rows if patch.lower().replace("cur_","patch_curitiba_") in r["asset_file_reference"])
    assert match["usable_for_human_review"] == "true"
    assert os.path.exists(match["visual_file_reference"])


def test_three_review_assets_ready():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[4]))
    assert len(rows) == 3
    assert all(r["review_asset_status"] == "READY_FOR_HUMAN_REVIEW" for r in rows)
    assert all(r["generated_derivative"] == "false" and r["derivative_is_review_only"] == "true" for r in rows)
    assert all(r["raw_data_versioned"] == "false" for r in rows)


def test_three_dino_links_recovered():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[5]))
    assert len(rows) == 3
    assert all(r["dino_embedding_id"] == "DINO_V1FU_SENTINEL_00001" for r in rows)
    assert all(r["dino_link_confidence"] == "MODERATE" and r["dino_review_signal_available"] == "true" for r in rows)
    assert all(r["dino_is_ground_truth"] == "false" and r["dino_can_create_label"] == "false" for r in rows)


def test_lineage_confidence_low():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[6]))
    assert len(rows) == 3
    assert all(r["date_confidence"] == "UNKNOWN" and r["patch_link_confidence"] == "HIGH" for r in rows)
    assert all(r["visual_reference_confidence"] == "HIGH" and r["dino_link_confidence"] == "MODERATE" for r in rows)
    assert all(r["overall_lineage_confidence"] == "LOW" for r in rows)


def test_status_visual_ready_weak_link():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[7]))
    assert len(rows) == 3
    assert all(r["updated_crosswalk_status"] == "VISUAL_REVIEW_READY_WITH_WEAK_LINK" for r in rows)
    assert all(r["ready_for_candidate_reference_adjudication"] == "false" for r in rows)
    assert all(r["next_action_rank_1"] == "MANUALLY_REPAIR_SENTINEL_PATCH_ASSET_MANIFEST" for r in rows)


def test_remaining_blockers():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[7]))
    required = {"SENTINEL_DATE_MISSING","SEED_TO_PATCH_LINK_MISSING","GEOMETRY_MISSING","HUMAN_REVIEW_PENDING"}
    assert all(set(r["remaining_blockers"].split("|")) == required for r in rows)


def test_three_packets():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[8]))
    assert len(rows) == 3
    assert all(r["updated_crosswalk_status"] == "VISUAL_REVIEW_READY_WITH_WEAK_LINK" for r in rows)


@pytest.mark.parametrize("folder", ["lineage_review_packets","recovered_crosswalk_tables","sentinel_review_assets","evidence_cache"])
def test_doc_folders(folder):
    assert os.path.isdir(common.doc_path(folder))


def test_packet_markdowns():
    files = [f for f in os.listdir(common.doc_path("lineage_review_packets")) if f.endswith(".md")]
    assert len(files) == 3
    for name in files:
        text = open(common.doc_path("lineage_review_packets", name), encoding="utf-8").read()
        assert "Linhagem geral: `LOW`" in text
        assert "nao sao truth ou label" in text


def test_sentinel_review_asset_readme():
    text = open(common.doc_path("sentinel_review_assets", "README.md"), encoding="utf-8").read()
    assert "Nenhum raster Sentinel ou embedding bruto foi copiado ou versionado" in text


@pytest.mark.parametrize("field", ["can_create_ground_truth","can_create_patch_truth","can_create_label","can_create_negative","can_train_model"])
def test_zero_forbidden_outputs(field):
    for output in common.OUTPUTS[:9]:
        assert all(r[field] == "false" for r in common.load_csv(common.dataset_path(output)))


def test_cache_marker():
    assert open(common.doc_path("evidence_cache", ".gitignore"), encoding="utf-8").read() == "*\n!.gitignore\n"


def test_guardrails_pass():
    rows = common.load_csv(common.dataset_path(common.OUTPUTS[9]))
    assert len(rows) == 10 and all(r["status"] == "PASS" and r["violation_count"] == "0" for r in rows)


def test_orchestrator_ok():
    rows = common.load_csv(common.dataset_path("v2bf_orchestrator_manifest.csv"))
    assert len(rows) == 10 and all(r["status"] == "OK" for r in rows)
