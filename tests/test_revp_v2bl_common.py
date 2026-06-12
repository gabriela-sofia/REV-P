import csv
import os

import pytest

import scripts.protocolo_c.revp_v2bl_common as common

CANDIDATE = "REC_2022_05_24_30"
PRODUCT = "CH758_RECIFE_20220602_001"
EVENT_PATCH = "FACT_v2at_0005"


def _write(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def seed(workspace, monkeypatch, charter_raster=True, vector=False, crs=False,
         apac=True, ana=True, cemaden=False, a301="PRECIP_FULL_GAP", feature="LANDSLIDE_SCARS",
         registry_present=True):
    datasets = os.path.join(workspace, "datasets")
    docs = os.path.join(workspace, "docs_v2bl")
    public = os.path.join(workspace, "outputs_public")
    os.makedirs(datasets, exist_ok=True)
    common.DATASET_DIR = datasets
    common.DOCS_DIR = docs
    common.PUBLIC_DIR = public
    common.REFRESH = False
    monkeypatch.setattr(common, "charter_feature", lambda: feature)

    if registry_present:
        _write(os.path.join(datasets, common.INPUTS["registry758"]),
               ["charter_activation_id", "activation_date", "requestor", "product_date", "product_type",
                "product_area", "product_title", "hazard_terms", "hazard_scope", "redistribution_status"],
               [{"charter_activation_id": "758", "activation_date": "2022-05-30", "requestor": "CENAD",
                 "product_date": "2022-06-02", "product_type": "MAP_RASTER", "product_area": "Recife/PE",
                 "product_title": "Landslides after effects in Recife/PE - Brazil",
                 "hazard_terms": "landslide", "hazard_scope": "Landslide/Flood", "redistribution_status": "PUBLIC_SOURCE"}])
    else:
        _write(os.path.join(datasets, common.INPUTS["registry758"]),
               ["charter_activation_id", "product_area", "product_title"],
               [{"charter_activation_id": "758", "product_area": "Olinda/PE", "product_title": "Olinda"}])

    intake = [{"source": "International Charter Activation 758", "file_present": "true" if charter_raster else "false"},
              {"source": "APAC monthly accumulated precipitation (May 2022)", "file_present": "true" if apac else "false"},
              {"source": "ANA HidroWeb Capibaribe - Sao Lourenco da Mata (39187800)", "file_present": "true" if ana else "false"},
              {"source": "Cemaden pluviometers Recife/RMR (May 2022)", "file_present": "true" if cemaden else "false"}]
    _write(os.path.join(datasets, common.INPUTS["intake"]), ["source", "file_present"], intake)
    _write(os.path.join(datasets, common.INPUTS["inmet"]),
           ["station_code", "coverage_status", "usable_as_recife_local_rainfall"],
           [{"station_code": "A301", "coverage_status": a301, "usable_as_recife_local_rainfall": "false"},
            {"station_code": "A320", "coverage_status": "PRECIP_AVAILABLE", "usable_as_recife_local_rainfall": "false"},
            {"station_code": "A328", "coverage_status": "PRECIP_PARTIAL", "usable_as_recife_local_rainfall": "false"}])
    readiness = "MAP_PRESENT_PENDING_VECTOR_CRS" if charter_raster else "NO_FILE_AVAILABLE"
    _write(os.path.join(datasets, common.INPUTS["charter_readiness"]),
           ["product_id", "updated_candidate_status"],
           [{"product_id": PRODUCT, "updated_candidate_status": readiness}])
    _write(os.path.join(datasets, common.INPUTS["charter_crs"]),
           ["product_id", "crs_present", "geometry_validity_status"],
           [{"product_id": PRODUCT, "crs_present": "true" if crs else "false",
             "geometry_validity_status": "VALID_FOR_HUMAN_REVIEW" if crs else "NOT_AVAILABLE"}])
    _write(os.path.join(datasets, common.INPUTS["charter_vector"]),
           ["product_id", "vector_file_detected"],
           [{"product_id": PRODUCT, "vector_file_detected": "true" if vector else "false"}])
    _write(os.path.join(datasets, common.INPUTS["temporal_metrics"]),
           ["event_patch_package_id", "temporal_status"],
           [{"event_patch_package_id": EVENT_PATCH, "temporal_status": "NO_SERIES_AVAILABLE"}])
    _write(os.path.join(datasets, common.INPUTS["queue"]),
           ["candidate_id", "reference_status"],
           [{"candidate_id": CANDIDATE, "reference_status": "CANDIDATE_REFERENCE_PENDING_HUMAN_REVIEW"}])
    _write(os.path.join(datasets, common.INPUTS["reconcile"]),
           ["candidate_id", "gate_id", "reconciled_status"],
           [{"candidate_id": CANDIDATE, "gate_id": g, "reconciled_status": "PRIOR"} for g in common.GATES])
    return datasets, docs, public


@pytest.fixture
def workspace(tmp_path):
    saved = (common.DATASET_DIR, common.DOCS_DIR, common.PUBLIC_DIR, common.REFRESH)
    yield str(tmp_path)
    common.DATASET_DIR, common.DOCS_DIR, common.PUBLIC_DIR, common.REFRESH = saved


def _load(name):
    with open(common.dataset_path(name), encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _full(workspace, monkeypatch, **kw):
    seed(workspace, monkeypatch, **kw)
    for fn in (common.run_load_recife_real_evidence_state, common.run_reclassify_non_blocking_limitations,
               common.run_apply_automated_protocol_adjudication, common.run_promote_recife_candidate_reference,
               common.run_build_validated_candidate_reference_registry, common.run_build_protocol_evidence_scorecard,
               common.run_build_reapplication_learning_matrix, common.run_generate_validated_reference_report,
               common.run_guardrail_regression):
        fn()


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("field,expected", list(common.INVARIANTS.items()))
def test_invariants(field, expected):
    assert common.with_invariants({})[field] == expected


def test_forbidden_invariants_false():
    for field in ("can_create_operational_label", "can_create_negative", "can_train_model", "raw_data_versioned"):
        assert common.INVARIANTS[field] == "false"


def test_methodology_invariants_true():
    for field in ("public_source_license_not_blocker", "protocol_validation_replaces_manual_review_step",
                  "raster_cartographic_evidence_can_support_candidate_reference", "raster_is_not_vector_geometry",
                  "landslide_scars_are_not_flood_extent", "ana_stage_is_not_precipitation",
                  "apac_pdf_is_not_hourly_station_series", "inmet_empty_precip_is_instrument_gap_not_absence",
                  "no_supervised_training_target_created", "candidate_reference_is_not_operational_label"):
        assert common.INVARIANTS[field] == "true"


# --------------------------------------------------------------------------- #
# Task 1 - state
# --------------------------------------------------------------------------- #

def test_state_built(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    r = common.run_load_recife_real_evidence_state()[0]
    assert r["recife_package_id"] == common.PACKAGE_ID
    assert r["charter_activation_id"] == "758"
    assert r["charter_product_date"] == "2022-06-02"


def test_state_feature_landslide_scars(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    r = common.run_load_recife_real_evidence_state()[0]
    assert r["charter_feature_type"] == "LANDSLIDE_SCARS"


def test_state_raster_available_vector_not(workspace, monkeypatch):
    seed(workspace, monkeypatch, charter_raster=True, vector=False, crs=False)
    r = common.run_load_recife_real_evidence_state()[0]
    assert r["charter_raster_available"] == "true"
    assert r["charter_vector_available"] == "false"
    assert r["charter_crs_available"] == "false"


def test_state_a301_gap(workspace, monkeypatch):
    seed(workspace, monkeypatch, a301="PRECIP_FULL_GAP")
    r = common.run_load_recife_real_evidence_state()[0]
    assert r["inmet_a301_precip_status"] == "PRECIP_FULL_GAP"


def test_state_proxy_status_recorded(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    r = common.run_load_recife_real_evidence_state()[0]
    assert "A320" in r["proxy_inmet_status"]


# --------------------------------------------------------------------------- #
# Task 2 - reclassification
# --------------------------------------------------------------------------- #

def test_reclass_all_non_blocking_for_reference(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_reclassify_non_blocking_limitations()
    assert rows and all(r["blocks_candidate_reference"] == "false" for r in rows)


def test_reclass_license_not_blocker(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = {r["item"]: r for r in common.run_reclassify_non_blocking_limitations()}
    lic = rows["LICENSE_REDISTRIBUTION_TERMS"]
    assert lic["limitation_type"] == "NON_BLOCKING_PUBLIC_PROVENANCE"
    assert lic["blocks_final_label"] == "false"


def test_reclass_vector_is_technical_limitation(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = {r["item"]: r for r in common.run_reclassify_non_blocking_limitations()}
    assert rows["CHARTER_VECTOR_ABSENCE"]["limitation_type"] == "TECHNICAL_LIMITATION"
    assert rows["CHARTER_CRS_ABSENCE"]["limitation_type"] == "TECHNICAL_LIMITATION"


def test_reclass_a301_gap_not_negative(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = {r["item"]: r for r in common.run_reclassify_non_blocking_limitations()}
    assert "instrument gap" in rows["INMET_A301_EMPTY_PRECIP"]["reclassification_reason"]


@pytest.mark.parametrize("item", ["LICENSE_REDISTRIBUTION_TERMS", "EXTERNAL_CONFIRMATION", "CHARTER_VECTOR_ABSENCE",
                                  "CHARTER_CRS_ABSENCE", "APAC_MONTHLY_PDF", "ANA_RIVER_STAGE",
                                  "INMET_A301_EMPTY_PRECIP", "CHARTER_RASTER_VS_VECTOR"])
def test_reclass_items_present(workspace, monkeypatch, item):
    seed(workspace, monkeypatch)
    items = {r["item"] for r in common.run_reclassify_non_blocking_limitations()}
    assert item in items


# --------------------------------------------------------------------------- #
# Task 3 - adjudication
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("gate,expected", list(common.ADJUDICATED.items()))
def test_adjudication_gate_status(workspace, monkeypatch, gate, expected):
    seed(workspace, monkeypatch)
    rows = {r["gate_id"]: r for r in common.run_apply_automated_protocol_adjudication()}
    assert rows[gate]["updated_gate_status"] == expected


def test_adjudication_c5_auto(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = {r["gate_id"]: r for r in common.run_apply_automated_protocol_adjudication()}
    assert rows["C5_PROTOCOL_VALIDATION"]["updated_gate_status"] == "AUTO_ADJUDICATED_BY_PROTOCOL"


def test_adjudication_c4_raster(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = {r["gate_id"]: r for r in common.run_apply_automated_protocol_adjudication()}
    assert rows["C4_CANDIDATE_GEOMETRY"]["updated_gate_status"] == "PASS_RASTER_CARTOGRAPHIC_EVIDENCE_FOR_REFERENCE"


def test_adjudication_c7_blocked(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = {r["gate_id"]: r for r in common.run_apply_automated_protocol_adjudication()}
    assert rows["C7_OPERATIONAL_LABEL_OR_FINAL_SUPERVISED_TRUTH"]["updated_gate_status"] == "NOT_CREATED_BLOCKED_FOR_TRAINING"


def test_adjudication_no_operational_label(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_apply_automated_protocol_adjudication()
    assert all(r["can_create_operational_label"] == "false" for r in rows)
    assert all(r["can_train_model"] == "false" for r in rows)


def test_adjudication_fail_closed_when_no_evidence(workspace, monkeypatch):
    seed(workspace, monkeypatch, charter_raster=False, apac=False, ana=False, registry_present=False)
    rows = {r["gate_id"]: r for r in common.run_apply_automated_protocol_adjudication()}
    assert rows["C6_CANDIDATE_REFERENCE"]["updated_gate_status"] == "PROTOCOL_VALIDATION_REQUIRED"
    assert rows["C7_OPERATIONAL_LABEL_OR_FINAL_SUPERVISED_TRUTH"]["updated_gate_status"] == "NOT_CREATED_BLOCKED_FOR_TRAINING"


def test_adjudication_confidence_levels(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = {r["gate_id"]: r for r in common.run_apply_automated_protocol_adjudication()}
    assert rows["C3_SPATIAL_ANCHOR"]["confidence_level"] == "HIGH"
    assert rows["C1_TEMPORALITY"]["confidence_level"] == "MODERATE"


def test_adjudication_gt_candidate_only_on_c6(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = {r["gate_id"]: r for r in common.run_apply_automated_protocol_adjudication()}
    assert rows["C6_CANDIDATE_REFERENCE"]["can_create_ground_truth_candidate"] == "true"
    assert rows["C7_OPERATIONAL_LABEL_OR_FINAL_SUPERVISED_TRUTH"]["can_create_ground_truth_candidate"] == "false"


# --------------------------------------------------------------------------- #
# Task 4 - promotion
# --------------------------------------------------------------------------- #

def test_promotion_allowed_with_evidence(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    r = common.run_promote_recife_candidate_reference()[0]
    assert r["promotion_allowed"] == "true"
    assert r["promoted_status"] == "PROTOCOL_VALIDATED_CANDIDATE_REFERENCE"
    assert r["promotion_type"] == "PROTOCOL_LEVEL_REFERENCE"


def test_promotion_not_allowed_without_charter(workspace, monkeypatch):
    seed(workspace, monkeypatch, charter_raster=False, registry_present=False)
    r = common.run_promote_recife_candidate_reference()[0]
    assert r["promotion_allowed"] == "false"
    assert r["promoted_status"] == "PENDING_PROTOCOL_VALIDATION"


def test_promotion_excludes_label_negative_training(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    r = common.run_promote_recife_candidate_reference()[0]
    assert "supervised_label" in r["excluded_interpretations"]
    assert r["can_create_operational_label"] == "false"
    assert r["can_create_negative"] == "false"
    assert r["can_train_model"] == "false"


def test_promotion_not_created_outputs(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    r = common.run_promote_recife_candidate_reference()[0]
    for token in ("operational_label", "negative", "supervised_training_target", "vector_geometry"):
        assert token in r["not_created_outputs"]


# --------------------------------------------------------------------------- #
# Task 5 - registry
# --------------------------------------------------------------------------- #

def test_registry_status_and_scope(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    r = common.run_build_validated_candidate_reference_registry()[0]
    assert r["reference_status"] == "PROTOCOL_VALIDATED_CANDIDATE_REFERENCE"
    assert r["phenomenon_scope"] == "LANDSLIDE_SCARS_WITH_FLOOD_EVENT_CONTEXT"


def test_registry_allowed_use(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    r = common.run_build_validated_candidate_reference_registry()[0]
    for token in ("PROTOCOL_C_REFERENCE_REVIEW", "ARTICLE_EVIDENCE", "PUBLIC_DELIVERY_TABLE"):
        assert token in r["allowed_use"]


def test_registry_forbidden_use(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    r = common.run_build_validated_candidate_reference_registry()[0]
    for token in ("SUPERVISED_LABEL", "NEGATIVE_LABEL", "TRAINING_TARGET", "FLOOD_EXTENT_TRUTH"):
        assert token in r["forbidden_use"]


def test_registry_evidence_score_numeric(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    r = common.run_build_validated_candidate_reference_registry()[0]
    assert 0.0 < float(r["evidence_score"]) <= 1.0


def test_registry_uncertainty_moderate(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    r = common.run_build_validated_candidate_reference_registry()[0]
    assert r["uncertainty_level"] == "MODERATE"


def test_registry_scope_unknown_without_feature(workspace, monkeypatch):
    # feature unknown AND no landslide hazard term in the registry fallback.
    seed(workspace, monkeypatch, feature="UNKNOWN", registry_present=False)
    r = common.run_build_validated_candidate_reference_registry()[0]
    assert r["phenomenon_scope"] == "UNKNOWN"


def test_registry_no_label_flags(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    r = common.run_build_validated_candidate_reference_registry()[0]
    assert r["can_create_operational_label"] == "false"
    assert r["can_train_model"] == "false"


# --------------------------------------------------------------------------- #
# Task 6 - scorecard
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("axis", ["PROVENANCE", "TEMPORALITY", "HYDROLOGICAL_CONTEXT", "SPATIAL_CARTOGRAPHIC_EVIDENCE",
                                  "HAZARD_TYPING", "GEOMETRY_VECTOR_READYNESS", "MODEL_LABEL_READYNESS"])
def test_scorecard_axis_present(workspace, monkeypatch, axis):
    seed(workspace, monkeypatch)
    axes = {r["evidence_axis"] for r in common.run_build_protocol_evidence_scorecard()}
    assert axis in axes


def test_scorecard_model_label_not_supported(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = {r["evidence_axis"]: r for r in common.run_build_protocol_evidence_scorecard()}
    assert rows["MODEL_LABEL_READYNESS"]["supports_operational_label"] == "false"
    assert rows["MODEL_LABEL_READYNESS"]["score"] == "0.0"


def test_scorecard_spatial_high(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = {r["evidence_axis"]: r for r in common.run_build_protocol_evidence_scorecard()}
    assert rows["SPATIAL_CARTOGRAPHIC_EVIDENCE"]["score"] == "1.0"
    assert rows["SPATIAL_CARTOGRAPHIC_EVIDENCE"]["supports_candidate_reference"] == "true"


def test_scorecard_geometry_vector_low(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = {r["evidence_axis"]: r for r in common.run_build_protocol_evidence_scorecard()}
    assert rows["GEOMETRY_VECTOR_READYNESS"]["supports_candidate_reference"] == "false"


def test_scorecard_no_axis_supports_operational_label(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_build_protocol_evidence_scorecard()
    assert all(r["supports_operational_label"] == "false" for r in rows)


# --------------------------------------------------------------------------- #
# Task 7 - learning matrix
# --------------------------------------------------------------------------- #

def test_learning_matrix_regions(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_build_reapplication_learning_matrix()
    regions = {r["applies_to_region"] for r in rows}
    assert {"CURITIBA", "PETROPOLIS", "RECIFE", "ALL"} <= regions


def test_learning_matrix_has_raster_lesson(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_build_reapplication_learning_matrix()
    assert any("raster" in r["lesson"].lower() for r in rows)


def test_learning_matrix_license_not_blocker_lesson(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_build_reapplication_learning_matrix()
    assert any("license" in r["lesson"].lower() and "blocker" in r["lesson"].lower() for r in rows)


def test_learning_matrix_priorities(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_build_reapplication_learning_matrix()
    assert {"P0", "P1"} <= {r["priority"] for r in rows}


# --------------------------------------------------------------------------- #
# Task 8 - report / packet / public
# --------------------------------------------------------------------------- #

def test_report_sections(workspace, monkeypatch):
    _full(workspace, monkeypatch)
    body = open(common.doc_path("reports", "recife_protocol_validated_candidate_reference_report.md"),
                encoding="utf-8").read()
    for section in ("## 1. Resumo", "## 7. O que isso permite", "## 8. O que isso NAO permite", "## 11. Guardrails"):
        assert section in body


def test_packet_written(workspace, monkeypatch):
    _full(workspace, monkeypatch)
    path = common.doc_path("validated_reference_packets", f"{CANDIDATE}_validated_candidate_reference.md")
    assert os.path.exists(path)
    body = open(path, encoding="utf-8").read()
    assert "Forbidden use" in body and "landslide scars" in body


def test_readme_written(workspace, monkeypatch):
    _full(workspace, monkeypatch)
    body = open(common.doc_path("README.md"), encoding="utf-8").read()
    assert "PROTOCOL_VALIDATED_CANDIDATE_REFERENCE" in body
    assert "C7" in body


def test_public_outputs_written(workspace, monkeypatch):
    _full(workspace, monkeypatch)
    for rel in common.PUBLIC_FILES.values():
        assert os.path.exists(common.public_path(rel))


def test_public_outputs_only_csv_md(workspace, monkeypatch):
    _full(workspace, monkeypatch)
    for rel in common.PUBLIC_FILES.values():
        assert os.path.splitext(rel)[1].lower() in {".csv", ".md"}


def test_public_registry_matches_dataset(workspace, monkeypatch):
    _full(workspace, monkeypatch)
    pub = list(csv.DictReader(open(common.public_path(common.PUBLIC_FILES["registry"]), encoding="utf-8-sig")))
    assert pub and pub[0]["reference_status"] == "PROTOCOL_VALIDATED_CANDIDATE_REFERENCE"


# --------------------------------------------------------------------------- #
# Semantic guards (no conversions)
# --------------------------------------------------------------------------- #

def test_raster_not_vector(workspace, monkeypatch):
    seed(workspace, monkeypatch, charter_raster=True, vector=False)
    r = common.run_load_recife_real_evidence_state()[0]
    assert r["charter_raster_available"] == "true" and r["charter_vector_available"] == "false"


def test_crs_absence_blocks_vector_overlay_only(workspace, monkeypatch):
    seed(workspace, monkeypatch, crs=False)
    rows = {r["item"]: r for r in common.run_reclassify_non_blocking_limitations()}
    crs_row = rows["CHARTER_CRS_ABSENCE"]
    assert crs_row["blocks_candidate_reference"] == "false"
    assert crs_row["blocks_final_label"] == "true"


def test_landslide_scars_not_flood_extent(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    r = common.run_build_validated_candidate_reference_registry()[0]
    assert "FLOOD_EXTENT_TRUTH" in r["forbidden_use"]
    assert common.INVARIANTS["landslide_scars_are_not_flood_extent"] == "true"


def test_ana_stage_not_precipitation(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = {r["evidence_axis"]: r for r in common.run_build_protocol_evidence_scorecard()}
    assert "not precipitation" in rows["HYDROLOGICAL_CONTEXT"]["limitation"]


def test_apac_pdf_not_hourly_series(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = {r["item"]: r for r in common.run_reclassify_non_blocking_limitations()}
    assert "hourly" in rows["APAC_MONTHLY_PDF"]["reclassification_reason"].lower()


def test_a301_empty_is_instrument_gap_not_absence(workspace, monkeypatch):
    seed(workspace, monkeypatch, a301="PRECIP_FULL_GAP")
    rows = {r["item"]: r for r in common.run_reclassify_non_blocking_limitations()}
    assert "absence of event" in rows["INMET_A301_EMPTY_PRECIP"]["reclassification_reason"]


def test_proxy_not_local(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    rows = common.run_build_reapplication_learning_matrix()
    assert any("proxy" in r["lesson"].lower() for r in rows)


# --------------------------------------------------------------------------- #
# Guardrail regression
# --------------------------------------------------------------------------- #

def test_guardrail_pass(workspace, monkeypatch):
    _full(workspace, monkeypatch)
    rows = common.run_guardrail_regression()
    assert rows and all(r["status"] == "PASS" for r in rows)


def test_guardrail_detects_operational_label(workspace, monkeypatch):
    _full(workspace, monkeypatch)
    reg = _load(common.OUTPUTS["registry"])
    reg[0]["can_create_operational_label"] = "true"
    common.write_csv(common.dataset_path(common.OUTPUTS["registry"]), reg)
    with pytest.raises(ValueError):
        common.run_guardrail_regression()


def test_guardrail_detects_c7_unblocked(workspace, monkeypatch):
    _full(workspace, monkeypatch)
    adj = _load(common.OUTPUTS["adjudication"])
    for r in adj:
        if r["gate_id"] == "C7_OPERATIONAL_LABEL_OR_FINAL_SUPERVISED_TRUTH":
            r["updated_gate_status"] = "CREATED"
    common.write_csv(common.dataset_path(common.OUTPUTS["adjudication"]), adj)
    with pytest.raises(ValueError):
        common.run_guardrail_regression()


def test_guardrail_detects_training_flag(workspace, monkeypatch):
    _full(workspace, monkeypatch)
    sc = _load(common.OUTPUTS["scorecard"])
    sc[0]["can_train_model"] = "true"
    common.write_csv(common.dataset_path(common.OUTPUTS["scorecard"]), sc)
    with pytest.raises(ValueError):
        common.run_guardrail_regression()


@pytest.mark.parametrize("name_key", ["state", "reclass", "adjudication", "promotion", "registry", "scorecard", "learning"])
def test_no_forbidden_true(workspace, monkeypatch, name_key):
    _full(workspace, monkeypatch)
    rows = _load(common.OUTPUTS[name_key])
    for r in rows:
        for field in ("can_create_operational_label", "can_create_negative", "can_train_model"):
            assert common.clean(r.get(field, "false")).lower() != "true"


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

def test_orchestrator_end_to_end(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    manifest = common.run_orchestrator()
    assert manifest[0]["step_name"] == "refresh_v2bk_v2bj_v2bi_inputs"
    assert manifest[0]["status"] == "SKIPPED"
    for name in common.OUTPUTS.values():
        assert os.path.exists(common.dataset_path(name))


def test_orchestrator_docs_and_public(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    common.run_orchestrator()
    assert os.path.exists(common.doc_path("README.md"))
    assert os.path.exists(common.doc_path("evidence_cache", ".gitignore"))
    for rel in common.PUBLIC_FILES.values():
        assert os.path.exists(common.public_path(rel))


def test_orchestrator_manifest_length(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    manifest = common.run_orchestrator()
    assert len(manifest) == 10
    assert all(m["status"] in {"OK", "SKIPPED"} for m in manifest)


def test_orchestrator_promotes_candidate_reference(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    common.run_orchestrator()
    reg = _load(common.OUTPUTS["registry"])[0]
    assert reg["reference_status"] == "PROTOCOL_VALIDATED_CANDIDATE_REFERENCE"


def test_evidence_cache_gitignore(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    common.run_orchestrator()
    assert "*" in open(common.doc_path("evidence_cache", ".gitignore"), encoding="utf-8").read()


def test_zero_label_negative_training_across_outputs(workspace, monkeypatch):
    seed(workspace, monkeypatch)
    common.run_orchestrator()
    for name in (common.OUTPUTS["adjudication"], common.OUTPUTS["promotion"], common.OUTPUTS["registry"]):
        for r in _load(name):
            assert r.get("can_create_operational_label", "false") == "false"
            assert r.get("can_create_negative", "false") == "false"
            assert r.get("can_train_model", "false") == "false"
