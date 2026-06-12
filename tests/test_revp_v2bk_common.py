import csv
import os

import pytest

import scripts.protocolo_c.revp_v2bk_common as common

CANDIDATE = "REC_2022_05_24_30"
PRODUCT = "CH758_RECIFE_20220602_001"
EVENT_PATCH = "FACT_v2at_0005"

GATE_STATUS = {
    "C0_PROVENANCE": "PASS_FOR_REVIEW", "C1_TEMPORALITY": "TEMPORALITY_SUPPORTED_FOR_HUMAN_REVIEW",
    "C2_VALID_SERIES_OR_STATION": "PARTIAL_FOR_HUMAN_REVIEW", "C3_SPATIAL_ANCHOR": "PASS",
    "C4_CANDIDATE_GEOMETRY": "MAP_PRESENT_PENDING_VECTOR_CRS", "C5_HUMAN_REVIEW": "PENDING",
    "C6_CANDIDATE_REFERENCE": "CANDIDATE_REFERENCE_PENDING_HUMAN_REVIEW", "C7_FINAL_GROUND_TRUTH": "BLOCKED",
}

FACTS = {"feature_type_candidate": "LANDSLIDE_SCARS",
         "license_terms": "Includes Pleiades material (c) CNES (2022), Distribution Airbus DS.",
         "product_date": "2022-06-02", "source_html_present": "true"}


def _write(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def seed(workspace, monkeypatch, charter_readiness="MAP_ONLY", crs_present=False,
         ana=True, apac=True, cemaden=False, a301="PRECIP_FULL_GAP", facts=None):
    datasets = os.path.join(workspace, "datasets")
    docs = os.path.join(workspace, "docs_v2bk")
    os.makedirs(datasets, exist_ok=True)
    common.DATASET_DIR = datasets
    common.DOCS_DIR = docs
    common.REFRESH = False
    monkeypatch.setattr(common, "charter_facts", lambda: dict(facts or FACTS))

    _write(os.path.join(datasets, common.INPUTS["reconcile"]),
           ["candidate_id", "gate_id", "reconciled_status", "human_action_required"],
           [{"candidate_id": CANDIDATE, "gate_id": g, "reconciled_status": s, "human_action_required": "ACT"}
            for g, s in GATE_STATUS.items()])
    _write(os.path.join(datasets, common.INPUTS["queue"]),
           ["candidate_id", "reference_status"],
           [{"candidate_id": CANDIDATE, "reference_status": "CANDIDATE_REFERENCE_PENDING_HUMAN_REVIEW"}])
    _write(os.path.join(datasets, common.INPUTS["inmet"]),
           ["station_code", "coverage_status", "usable_as_recife_local_rainfall"],
           [{"station_code": "A301", "coverage_status": a301, "usable_as_recife_local_rainfall": "false"},
            {"station_code": "A320", "coverage_status": "PRECIP_AVAILABLE", "usable_as_recife_local_rainfall": "false"}])
    readiness = {"MAP_ONLY": "PREVIEW_ONLY_NOT_READY", "READY": "CANDIDATE_GEOMETRY_READY_FOR_HUMAN_REVIEW",
                 "NONE": "NO_FILE_AVAILABLE"}[charter_readiness]
    _write(os.path.join(datasets, common.INPUTS["charter_readiness"]),
           ["product_id", "updated_candidate_status"],
           [{"product_id": PRODUCT, "updated_candidate_status": readiness}])
    _write(os.path.join(datasets, common.INPUTS["charter_crs"]),
           ["product_id", "crs_present", "geometry_validity_status"],
           [{"product_id": PRODUCT, "crs_present": "true" if crs_present else "false",
             "geometry_validity_status": "VALID_FOR_HUMAN_REVIEW" if crs_present else "NOT_AVAILABLE"}])
    _write(os.path.join(datasets, common.INPUTS["temporal_metrics"]),
           ["event_patch_package_id", "temporal_status"],
           [{"event_patch_package_id": EVENT_PATCH, "temporal_status": "NO_SERIES_AVAILABLE"}])
    intake = [{"source": "International Charter Activation 758", "file_present": "true"},
              {"source": "APAC monthly accumulated precipitation (May 2022)", "file_present": "true" if apac else "false"},
              {"source": "ANA HidroWeb Capibaribe - Sao Lourenco da Mata (39187800)", "file_present": "true" if ana else "false"},
              {"source": "Cemaden pluviometers Recife/RMR (May 2022)", "file_present": "true" if cemaden else "false"}]
    _write(os.path.join(datasets, common.INPUTS["intake"]), ["source", "file_present"], intake)
    return datasets, docs


@pytest.fixture
def workspace(tmp_path):
    saved = (common.DATASET_DIR, common.DOCS_DIR, common.REFRESH)
    yield str(tmp_path)
    common.DATASET_DIR, common.DOCS_DIR, common.REFRESH = saved


def _load(name):
    with open(common.dataset_path(name), encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("field,expected", list(common.INVARIANTS.items()))
def test_invariants(field, expected):
    assert common.with_invariants({})[field] == expected


def test_forbidden_invariants_are_false():
    for field in ("can_create_ground_truth", "can_create_patch_truth", "can_create_label",
                  "can_create_negative", "can_train_model", "raw_data_versioned"):
        assert common.INVARIANTS[field] == "false"


def test_semantic_invariants_true():
    for field in ("request_pack_is_not_evidence", "human_review_dossier_is_not_ground_truth",
                  "charter_raster_is_not_vector_geometry", "ana_stage_is_not_precipitation",
                  "apac_pdf_is_not_station_series", "inmet_proxy_is_not_local_station",
                  "candidate_reference_is_not_final_truth"):
        assert common.INVARIANTS[field] == "true"


# --------------------------------------------------------------------------- #
# Task 1 - dossier
# --------------------------------------------------------------------------- #

def test_dossier_index_built(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_build_recife_human_review_dossier()
    assert len(rows) == 1
    r = rows[0]
    assert r["recife_package_id"] == common.PACKAGE_ID
    assert r["candidate_status"] == "CANDIDATE_REFERENCE_PENDING_HUMAN_REVIEW"
    assert r["ready_for_human_review"] == "true"


def test_dossier_charter_feature_landslide_scars(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    r = common.run_build_recife_human_review_dossier()[0]
    assert r["charter_feature_type"] == "LANDSLIDE_SCARS"
    assert r["charter_product_status"] == "MAP_RASTER_PRESENT"


def test_dossier_precip_local_empty(workspace, monkeypatch):
    seed(workspace, monkeypatch, a301="PRECIP_FULL_GAP")
    r = common.run_build_recife_human_review_dossier()[0]
    assert r["precipitation_local_status"] == "EMPTY_NOT_USABLE"


def test_dossier_gates_reflected(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    r = common.run_build_recife_human_review_dossier()[0]
    assert r["current_gate_c3"] == "PASS"
    assert r["current_gate_c4"] == "MAP_PRESENT_PENDING_VECTOR_CRS"
    assert r["current_gate_c7"] == "BLOCKED"


@pytest.mark.parametrize("field", ["can_create_ground_truth", "can_create_label", "can_create_negative",
                                   "can_train_model"])
def test_dossier_no_truth_flags(workspace, monkeypatch, field):
    seed(workspace, monkeypatch)
    r = common.run_build_recife_human_review_dossier()[0]
    assert r[field] == "false"


# --------------------------------------------------------------------------- #
# Task 2 - charter request pack
# --------------------------------------------------------------------------- #

def test_charter_request_artifacts(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_build_charter_vector_crs_request_pack()
    artifacts = {r["requested_artifact"] for r in rows}
    assert artifacts == {"VECTOR_FILE", "CRS_METADATA", "FEATURE_DEFINITION", "LICENSE_TERMS",
                         "REDISTRIBUTION_TERMS", "METHOD_DESCRIPTION"}


def test_charter_request_blocks_c4(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_build_charter_vector_crs_request_pack()
    assert all("C4_CANDIDATE_GEOMETRY" in r["blocks_gate"] for r in rows)
    assert all(r["activation_id"] == "758" for r in rows)


def test_charter_request_templates_written(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    common.run_build_charter_vector_crs_request_pack()
    cenad = common.doc_path("request_templates", "request_charter_758_vector_crs_cenad.md")
    charter = common.doc_path("request_templates", "request_charter_758_vector_crs_charter.md")
    assert os.path.exists(cenad) and os.path.exists(charter)
    body = open(cenad, encoding="utf-8").read()
    assert "landslide scars" in body and "CRS" in body and "redistribu" in body.lower()


def test_charter_request_no_ground_truth(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_build_charter_vector_crs_request_pack()
    assert all(r["can_create_ground_truth"] == "false" for r in rows)


# --------------------------------------------------------------------------- #
# Task 3 - temporal request pack
# --------------------------------------------------------------------------- #

def test_temporal_request_targets(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_build_cemaden_apac_temporal_request_pack()
    institutions = {r["target_institution"] for r in rows}
    assert "Cemaden/MCTI" in institutions and "APAC-PE" in institutions


def test_temporal_request_period_and_gate(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_build_cemaden_apac_temporal_request_pack()
    assert all(r["requested_period_start"] == "2022-05-24" for r in rows)
    assert all(r["requested_period_end"] == "2022-06-02" for r in rows)
    assert all("C2_VALID_SERIES_OR_STATION" in r["blocks_gate"] for r in rows)


def test_temporal_request_templates_written(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    common.run_build_cemaden_apac_temporal_request_pack()
    cem = common.doc_path("request_templates", "request_cemaden_recife_rmr_precip_20220524_20220602.md")
    apac = common.doc_path("request_templates", "request_apac_recife_rmr_precip_20220524_20220602.md")
    assert os.path.exists(cem) and os.path.exists(apac)
    body = open(cem, encoding="utf-8").read()
    assert "A301" in body and "timezone" in body.lower()


def test_temporal_request_apac_pdf_not_station_series(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_build_cemaden_apac_temporal_request_pack()
    assert all(r["apac_pdf_is_not_station_series"] == "true" for r in rows)


# --------------------------------------------------------------------------- #
# Task 4 - checklist
# --------------------------------------------------------------------------- #

def test_checklist_has_c5_and_c6(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_build_c5_c6_adjudication_checklist()
    stages = {r["decision_stage"] for r in rows}
    assert stages == {"C5_HUMAN_REVIEW", "C6_CANDIDATE_REFERENCE"}
    assert len(rows) >= 11


def test_checklist_final_ground_truth_is_no(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_build_c5_c6_adjudication_checklist()
    final_q = [r for r in rows if "ground truth final" in r["review_question"].lower()]
    assert final_q and "Nao" in final_q[0]["cannot_infer"]


def test_checklist_acceptable_decisions_enumerated(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_build_c5_c6_adjudication_checklist()
    for r in rows:
        for decision in ("ACCEPT_FOR_CANDIDATE_REFERENCE", "KEEP_PENDING", "REJECT_FOR_NOW",
                         "REQUEST_MORE_EVIDENCE", "MARK_HAZARD_AMBIGUOUS"):
            assert decision in r["acceptable_decision"]


def test_checklist_hazard_ambiguous_for_feature(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_build_c5_c6_adjudication_checklist()
    feature_q = [r for r in rows if "feicao" in r["review_question"].lower()]
    assert feature_q and feature_q[0]["current_recommendation"] == "MARK_HAZARD_AMBIGUOUS"


def test_checklist_markdown_written(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    common.run_build_c5_c6_adjudication_checklist()
    assert os.path.exists(common.doc_path("adjudication_checklists", f"{CANDIDATE}_c5_c6_checklist.md"))


# --------------------------------------------------------------------------- #
# Task 5 - decision matrix
# --------------------------------------------------------------------------- #

def test_decision_matrix_axes(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_build_recife_decision_matrix()
    axes = {r["decision_axis"] for r in rows}
    assert axes == {"TEMPORALITY", "LOCAL_PRECIPITATION", "HYDROLOGICAL_CONTEXT", "CHARTER_SPATIAL_PRODUCT",
                    "GEOMETRY_ACCESS", "HAZARD_TYPING", "HUMAN_REVIEW", "FINAL_TRUTH"}


def test_decision_matrix_final_truth_blocked(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_build_recife_decision_matrix()
    final = next(r for r in rows if r["decision_axis"] == "FINAL_TRUTH")
    assert final["current_status"] == "BLOCKED"
    assert final["promotion_allowed"] == "false"


def test_decision_matrix_no_promotion(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_build_recife_decision_matrix()
    assert all(r["promotion_allowed"] == "false" for r in rows)


def test_decision_matrix_hydro_context_only(workspace, monkeypatch):
    seed(workspace, monkeypatch, ana=True)
    rows = common.run_build_recife_decision_matrix()
    hydro = next(r for r in rows if r["decision_axis"] == "HYDROLOGICAL_CONTEXT")
    assert hydro["current_status"] == "PRESENT_CONTEXT_ONLY"
    assert "not precipitation" in hydro["blocker"]


def test_decision_matrix_markdown_written(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    common.run_build_recife_decision_matrix()
    assert os.path.exists(common.doc_path("decision_matrix", f"{CANDIDATE}_decision_matrix.md"))


# --------------------------------------------------------------------------- #
# Task 6 - markdown + readme
# --------------------------------------------------------------------------- #

def _full_run(workspace, monkeypatch, **kw):
    seed(workspace, monkeypatch, **kw)
    common.run_build_recife_human_review_dossier()
    common.run_build_charter_vector_crs_request_pack()
    common.run_build_cemaden_apac_temporal_request_pack()
    common.run_build_c5_c6_adjudication_checklist()
    common.run_build_recife_decision_matrix()
    common.run_generate_review_ready_markdown()
    common.run_generate_readme()


def test_review_dossier_markdown_sections(workspace, monkeypatch):
    _full_run(workspace, monkeypatch)
    body = open(common.doc_path("dossier", f"{CANDIDATE}_review_dossier.md"), encoding="utf-8").read()
    for section in ("## 1.", "## 7. Gates", "## 11. Decisao recomendada", "## 12. Guardrails"):
        assert section in body
    assert "CANDIDATE_REFERENCE_PENDING_HUMAN_REVIEW" in body
    assert "NAO promover a ground truth final" in body


def test_review_dossier_states_what_sources_do_not_prove(workspace, monkeypatch):
    _full_run(workspace, monkeypatch)
    body = open(common.doc_path("dossier", f"{CANDIDATE}_review_dossier.md"), encoding="utf-8").read()
    assert "NAO prova" in body
    assert "flood extent" in body


def test_readme_written(workspace, monkeypatch):
    _full_run(workspace, monkeypatch)
    body = open(common.doc_path("README.md"), encoding="utf-8").read()
    assert "v2bk" in body and "BLOCKED" in body
    assert "Ground truth final, labels, negativos e treino = 0" in body


# --------------------------------------------------------------------------- #
# Semantic guards (no conversions)
# --------------------------------------------------------------------------- #

def test_raster_not_vector_when_map_only(workspace, monkeypatch):
    seed(workspace, monkeypatch, charter_readiness="MAP_ONLY", crs_present=False)
    r = common.run_build_recife_human_review_dossier()[0]
    assert r["charter_crs_status"] == "ABSENT_OR_UNKNOWN"
    assert r["current_gate_c4"] == "MAP_PRESENT_PENDING_VECTOR_CRS"


def test_vector_ready_allows_review_only(workspace, monkeypatch):
    seed(workspace, monkeypatch, charter_readiness="READY", crs_present=True)
    r = common.run_build_recife_human_review_dossier()[0]
    assert r["charter_crs_status"] == "PRESENT"
    # even with vector/crs, C7 stays blocked
    assert r["current_gate_c7"] == "BLOCKED"


def test_inmet_proxy_not_local(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_build_recife_decision_matrix()
    local = next(r for r in rows if r["decision_axis"] == "LOCAL_PRECIPITATION")
    assert "proxies regional" in local["evidence_supporting"]
    assert local["current_status"] != "PASS"


# --------------------------------------------------------------------------- #
# Guardrail regression
# --------------------------------------------------------------------------- #

def test_guardrail_pass(workspace, monkeypatch):
    _full_run(workspace, monkeypatch)
    rows = common.run_guardrail_regression()
    assert rows and all(r["status"] == "PASS" for r in rows)


def test_guardrail_detects_promotion(workspace, monkeypatch):
    _full_run(workspace, monkeypatch)
    decision = _load(common.OUTPUTS["decision"])
    decision[0]["promotion_allowed"] = "true"
    common.write_csv(common.dataset_path(common.OUTPUTS["decision"]), decision)
    with pytest.raises(ValueError):
        common.run_guardrail_regression()


def test_guardrail_detects_unblocked_final_truth(workspace, monkeypatch):
    _full_run(workspace, monkeypatch)
    decision = _load(common.OUTPUTS["decision"])
    for r in decision:
        if r["decision_axis"] == "FINAL_TRUTH":
            r["current_status"] = "PASS"
    common.write_csv(common.dataset_path(common.OUTPUTS["decision"]), decision)
    with pytest.raises(ValueError):
        common.run_guardrail_regression()


@pytest.mark.parametrize("name_key", ["dossier_index", "charter_request", "temporal_request", "checklist", "decision"])
def test_no_forbidden_true_in_outputs(workspace, monkeypatch, name_key):
    _full_run(workspace, monkeypatch)
    rows = _load(common.OUTPUTS[name_key])
    for r in rows:
        for field in ("can_create_ground_truth", "can_create_label", "can_create_negative", "can_train_model"):
            assert common.clean(r.get(field, "false")).lower() != "true"


# --------------------------------------------------------------------------- #
# Orchestrator end-to-end
# --------------------------------------------------------------------------- #

def test_orchestrator_end_to_end(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    manifest = common.run_orchestrator()
    assert manifest[0]["step_name"] == "refresh_v2bj_v2bi_inputs"
    assert manifest[0]["status"] == "SKIPPED"
    for name in common.OUTPUTS.values():
        assert os.path.exists(common.dataset_path(name))


def test_orchestrator_docs_structure(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    common.run_orchestrator()
    assert os.path.exists(common.doc_path("README.md"))
    assert os.path.exists(common.doc_path("dossier", f"{CANDIDATE}_review_dossier.md"))
    assert os.path.exists(common.doc_path("evidence_cache", ".gitignore"))
    for tpl in ("request_charter_758_vector_crs_cenad.md", "request_charter_758_vector_crs_charter.md",
                "request_cemaden_recife_rmr_precip_20220524_20220602.md",
                "request_apac_recife_rmr_precip_20220524_20220602.md"):
        assert os.path.exists(common.doc_path("request_templates", tpl))


def test_orchestrator_manifest_all_ok(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    manifest = common.run_orchestrator()
    assert all(m["status"] in {"OK", "SKIPPED"} for m in manifest)
    assert len(manifest) == 9


def test_evidence_cache_gitignored(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    common.run_orchestrator()
    body = open(common.doc_path("evidence_cache", ".gitignore"), encoding="utf-8").read()
    assert "*" in body


def test_charter_facts_absent_fail_closed(workspace, monkeypatch):
    seed(workspace, monkeypatch, facts={"feature_type_candidate": "UNKNOWN", "license_terms": "UNKNOWN",
                                         "product_date": "UNKNOWN", "source_html_present": "false"})
    r = common.run_build_recife_human_review_dossier()[0]
    assert r["charter_feature_type"] == "UNKNOWN"


def test_dossier_index_single_candidate(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    common.run_build_recife_human_review_dossier()
    assert len(_load(common.OUTPUTS["dossier_index"])) == 1


def test_temporal_request_variable_precipitation(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_build_cemaden_apac_temporal_request_pack()
    assert all("precipitation" in r["requested_variable"] for r in rows)


def test_charter_request_priority_levels(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_build_charter_vector_crs_request_pack()
    priorities = {r["priority"] for r in rows}
    assert "P0" in priorities
